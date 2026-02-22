# depth_warp_vs/engine/trainer_refiner.py
import os, math, yaml, shutil
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from depth_warp_vs.models.refiner import build_refiner
from depth_warp_vs.models.losses.perceptual import VGGPerceptual
from depth_warp_vs.models.losses.photometric import ssim as ssim_map
from depth_warp_vs.models.losses.regularizers import tv as tv_reg
from depth_warp_vs.engine.logger import get_logger
from depth_warp_vs.engine.seed import set_seed
from depth_warp_vs.data.build import build_dataloader
from depth_warp_vs.models.utils.amp import AmpScaler
from depth_warp_vs.models.utils.ema import ModelEMA
from depth_warp_vs.engine.metrics import compute_metrics
from depth_warp_vs.engine.vis_utils import (
    make_tmp_run_dir, finalize_run_dir, save_json, save_csv, plot_curves, save_eval_samples
)

def _charbonnier_l1(x, y, mask=None, eps=1.0e-3):
    diff = x - y
    if mask is not None:
        denom = torch.clamp(mask.sum() * x.shape[1], min=1.0)
        return torch.sqrt(diff.pow(2) + eps*eps).mul(mask).sum() / denom
    return torch.sqrt(diff.pow(2) + eps*eps).mean()

def _masked_mean(x, w):
    denom = torch.clamp(w.sum() * x.shape[1], min=1.0)
    return (x * w).sum() / denom

def _cosine_lr(base_lr, step, warmup, max_steps):
    if step < warmup:
        return base_lr * float(step) / max(1, warmup)
    t = (step - warmup) / max(1, (max_steps - warmup))
    t = min(max(t, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))

def _adjust_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr

def _build_optimizer_with_decay(model, cfg):
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"]["weight_decay"])
    betas = tuple(cfg["optim"]["betas"])
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if (not p.requires_grad):
            continue
        if n.endswith(".bias") or ("norm" in n) or ("bn" in n) or ("gn" in n):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=lr, betas=betas)

def _save_ckpt(path, model, optimizer, scaler, step, cfg, ema=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.scaler.state_dict() if scaler is not None else None,
        "step": step,
        "cfg": cfg
    }
    if ema is not None and hasattr(ema, "ema"):
        obj["model_ema"] = ema.ema.state_dict()
    torch.save(obj, path)

def _clean_state(state):
    return { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }

# 兼容预训练加载与恢复训练的通用ckpt加载器（含EMA）
def _load_ckpt(path, model: torch.nn.Module, optimizer=None, scaler=None, map_location="cpu", ema: ModelEMA=None, load_ema=True,
    resume=False, strict=False
):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Refiner ckpt not found: {path}")

    ck = torch.load(path, map_location=map_location, weights_only=True)
    step = int(ck.get("step", 0))

    # 1) 加载model参数（过滤不匹配形状的键）
    raw_state = ck.get("model", ck)
    cleaned = _clean_state(raw_state)
    msd = model.state_dict()
    to_load = {}
    skipped_mismatch = []
    for k, v in cleaned.items():
        if k in msd and msd[k].shape == v.shape:
            to_load[k] = v
        else:
            if k in msd:
                skipped_mismatch.append((k, tuple(v.shape), tuple(msd[k].shape)))
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    if missing:
        print(f"提示：模型缺少以下键（将使用默认初始化）：{missing}")
    if unexpected:
        print(f"提示：权重中存在未使用的键：{unexpected}")
    if skipped_mismatch:
        print("提示：以下键因形状不匹配已跳过加载：")
        for k, vs, ms in skipped_mismatch[:20]:
            print(f"  - {k}: ckpt{vs} != model{ms}")
        if len(skipped_mismatch) > 20:
            print(f"  ... 共跳过 {len(skipped_mismatch)} 个键")

    # 2) EMA
    if load_ema and ema is not None and hasattr(ema, "ema"):
        if "model_ema" in ck and isinstance(ck["model_ema"], dict):
            ema_raw = _clean_state(ck["model_ema"])
            esd = ema.ema.state_dict()
            ema_to_load = {}
            ema_skipped = []
            for k, v in ema_raw.items():
                if k in esd and esd[k].shape == v.shape:
                    ema_to_load[k] = v
                else:
                    if k in esd:
                        ema_skipped.append((k, tuple(v.shape), tuple(esd[k].shape)))
            ema_missing, ema_unexpected = ema.ema.load_state_dict(ema_to_load, strict=False)
            if ema_missing:
                print(f"提示：EMA缺少以下键（将使用默认初始化）：{ema_missing}")
            if ema_unexpected:
                print(f"提示：EMA权重中存在未使用的键：{ema_unexpected}")
            if ema_skipped:
                print("提示：以下EMA键因形状不匹配已跳过加载：")
                for k, vs, ms in ema_skipped[:20]:
                    print(f"  - {k}: ckpt{vs} != ema{ms}")
                if len(ema_skipped) > 20:
                    print(f"  ... 共跳过 {len(ema_skipped)} 个键")
        else:
            # 若ckpt无EMA，则用当前model参数初始化EMA
            ema.ema.load_state_dict(model.state_dict(), strict=False)

    # 3) 恢复优化器与scaler（仅在resume模式）
    if resume:
        if optimizer is not None and ck.get("optimizer", None) is not None:
            try:
                optimizer.load_state_dict(ck["optimizer"])
            except Exception as e:
                print(f"警告：加载optimizer失败，已忽略。错误信息：{e}")
        if scaler is not None and ck.get("scaler", None) is not None:
            try:
                scaler.scaler.load_state_dict(ck["scaler"])
            except Exception as e:
                print(f"警告：加载scaler失败，已忽略。错误信息：{e}")

    return step

@torch.no_grad()
def evaluate_refiner(model, dl, device, max_batches=20):
    model.eval()
    total = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    samples = []
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        x, gt, mk_union = batch
        x, gt, mk_union = x.to(device), gt.to(device), mk_union.to(device)
        Iw = x[:, :3]
        pred = model(x)
        pred_final = mk_union * pred + (1.0 - mk_union) * Iw
        mets = compute_metrics(pred_final.clamp(0,1), gt.clamp(0,1), mask=None)
        total["l1"] += mets["l1"]; total["psnr"] += mets["psnr"]; total["ssim"] += mets["ssim"]
        count += 1
        if len(samples) < 8:
            samples.append((
                Iw[0].detach().cpu().permute(1,2,0).numpy(),
                mk_union[0].detach().cpu().permute(1,2,0).numpy(),
                pred_final[0].detach().cpu().permute(1,2,0).numpy(),
                gt[0].detach().cpu().permute(1,2,0).numpy()
            ))
    if count == 0:
        return {"l1": None, "psnr": None, "ssim": None}, []
    return {k: v/count for k,v in total.items()}, samples

def _downsample_like(img, size, mode="bilinear"):
    return torch.nn.functional.interpolate(img, size=size, mode=mode, align_corners=True if mode=="bilinear" else None)

def train_refiner(cfg_path: str):
    logger = get_logger()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # 构建训练/验证/测试数据加载器
    train_split = cfg.get("data", {}).get("split", "train")
    dl = build_dataloader(cfg, split=train_split)

    def _build_loader_for_split(split_name):
        try:
            return build_dataloader(cfg, split=split_name)
        except Exception:
            return None

    data_cfg = cfg.get("data", {})
    val_split = data_cfg.get("val_split", "validation")
    test_split = data_cfg.get("test_split", "test")

    dl_val = _build_loader_for_split(val_split)
    dl_test = _build_loader_for_split(test_split)

    # 构建模型与工具
    model = build_refiner(cfg).to(device)
    rcfg = cfg.get("model", {}).get("refiner", {})
    model_name = rcfg.get("type", "InpaintRefiner")
    per_loss = VGGPerceptual().to(device)
    opt = _build_optimizer_with_decay(model, cfg)
    scaler = AmpScaler(enabled=bool(cfg.get("amp", True)))
    ema = ModelEMA(model, decay=float(cfg.get("model", {}).get("ema_decay", 0.999)))

    # 日志与目录
    base_ckpt_dir = cfg.get("log", {}).get("ckpt_dir", "./checkpoints")
    run_tmp_dir = make_tmp_run_dir(base_ckpt_dir, model_name)
    os.makedirs(run_tmp_dir, exist_ok=True)
    try:
        with open(os.path.join(run_tmp_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
    except Exception:
        pass

    # 预训练/恢复
    start_step = 0
    resume = cfg.get("log", {}).get("resume", "")
    pretrained = cfg.get("log", {}).get("pretrained", "")
    if resume and os.path.isfile(resume):
        start_step = _load_ckpt(
            resume, model, optimizer=opt, scaler=scaler,
            map_location=device, ema=ema, load_ema=True, resume=True, strict=False
        )
        logger.info(f"Resumed from {resume} at step={start_step}")
    elif pretrained and os.path.isfile(pretrained):
        _ = _load_ckpt(
            pretrained, model, optimizer=None, scaler=None,
            map_location=device, ema=ema, load_ema=False, resume=False, strict=False
        )
        logger.info(f"Loaded pretrained weights from {pretrained}")

    # 训练配置
    model.train()
    max_steps = int(cfg["optim"].get("max_steps", 200000))
    warmup = int(cfg["optim"].get("warmup_steps", 1000))
    base_lr = float(cfg["optim"]["lr"])
    save_interval = int(cfg["log"].get("save_interval", 2000))
    val_interval = int(cfg["log"].get("val_interval", 5000))
    meter_interval = int(cfg.get("log", {}).get("meter_interval", 200))
    accum_steps = int(cfg["optim"].get("accum_steps", 1))
    grad_clip_norm = float(cfg["optim"].get("grad_clip_norm", 0.0))
    val_max_batches = int(cfg.get("data", {}).get("val_max_batches", 50))
    test_max_batches = int(cfg.get("data", {}).get("test_max_batches", 100))

    step = start_step
    use_charb = bool(cfg["loss"].get("use_charbonnier", True))
    charb_eps = float(cfg["loss"].get("charbonnier_eps", 1e-3))
    hole_w_start = float(cfg["loss"].get("hole_weight_start", cfg["loss"].get("hole_weight", 3.0)))
    hole_w_end   = float(cfg["loss"].get("hole_weight_end",   cfg["loss"].get("hole_weight", 3.0)))
    hole_w_ramp  = int(cfg["loss"].get("hole_weight_ramp",    20000))
    pollute_w    = float(cfg["loss"].get("pollute_weight", 2.0))
    valid_w      = float(cfg["loss"].get("valid_weight", 1.0))
    lam_ssim = float(cfg["loss"].get("ssim", 0.2))
    lam_vgg  = float(cfg["loss"].get("vgg", 0.1))
    lam_tv   = float(cfg["loss"].get("tv", 0.0))
    ms_scales = cfg["loss"].get("multiscale", [1.0, 0.5])
    ms_weights = cfg["loss"].get("ms_weights", None)
    if ms_weights is None or len(ms_weights) != len(ms_scales):
        ms_weights = [1.0 for _ in ms_scales]

    def hole_weight_at(s):
        if hole_w_ramp <= 0: return hole_w_end
        t = min(1.0, max(0.0, (s - start_step) / float(hole_w_ramp)))
        return hole_w_start + (hole_w_end - hole_w_start) * t

    history = {"step": [], "loss": [], "lr": [], "psnr": [], "ssim": []}
    pbar = tqdm(total=max_steps - step, initial=0, desc=f"Refiner Training ({model_name})")

    best_psnr = -1.0

    opt.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch in dl:
            lr_now = _cosine_lr(base_lr, step + 1, warmup, max_steps) if cfg["optim"].get("cosine", True) else base_lr
            _adjust_lr(opt, lr_now)

            x, gt, mk_union = batch
            x, gt, mk_union = x.to(device), gt.to(device), mk_union.to(device)
            Iw = x[:, :3]
            mk_all = x[:, 3:]
            mk_hole = mk_all[:, :1] if mk_all.shape[1] >= 1 else torch.zeros_like(mk_union)
            mk_poll = mk_all[:, 1:2] if mk_all.shape[1] >= 2 else torch.zeros_like(mk_union)

            with scaler.autocast(dtype=torch.float16 if cfg.get("precision","fp16")=="fp16" else torch.bfloat16):
                pred = model(x)
                loss_ms = x.new_tensor(0.0)
                for s_i, (scale, w) in enumerate(zip(ms_scales, ms_weights)):
                    if scale == 1.0:
                        Iw_s, gt_s, pred_s = Iw, gt, pred
                        mk_u_s, mk_h_s, mk_p_s = mk_union, mk_hole, mk_poll
                    else:
                        Hs = int(round(Iw.shape[-2] * float(scale)))
                        Ws = int(round(Iw.shape[-1] * float(scale)))
                        Iw_s  = _downsample_like(Iw,  (Hs, Ws), mode="bilinear")
                        gt_s  = _downsample_like(gt,  (Hs, Ws), mode="bilinear")
                        pred_s= _downsample_like(pred,(Hs, Ws), mode="bilinear")
                        mk_u_s= _downsample_like(mk_union, (Hs, Ws), mode="nearest")
                        mk_h_s= _downsample_like(mk_hole,  (Hs, Ws), mode="nearest")
                        mk_p_s= _downsample_like(mk_poll,  (Hs, Ws), mode="nearest")

                    pred_final_s = mk_u_s * pred_s + (1.0 - mk_u_s) * Iw_s

                    if use_charb:
                        L1_hole    = _charbonnier_l1(pred_final_s, gt_s, mask=mk_h_s, eps=charb_eps)
                        L1_pollute = _charbonnier_l1(pred_final_s, gt_s, mask=mk_p_s, eps=charb_eps)
                        L1_valid   = _charbonnier_l1(pred_final_s, gt_s, mask=(1.0 - mk_u_s), eps=charb_eps)
                    else:
                        l1 = torch.abs(pred_final_s - gt_s)
                        L1_hole    = _masked_mean(l1, mk_h_s)
                        L1_pollute = _masked_mean(l1, mk_p_s)
                        L1_valid   = _masked_mean(l1, 1.0 - mk_u_s)

                    L_ssim = _masked_mean(ssim_map(pred_final_s, gt_s), mk_u_s)
                    loss_ms = loss_ms + w * (hole_weight_at(step+1) * L1_hole + pollute_w * L1_pollute + valid_w * L1_valid + lam_ssim * L_ssim)

                pred_final = mk_union * pred + (1.0 - mk_union) * Iw
                L_vgg = per_loss(pred_final, gt, mask=mk_union) * lam_vgg if lam_vgg > 0 else pred_final.new_tensor(0.0)
                L_tv  = tv_reg(pred_final) * lam_tv if lam_tv > 0 else pred_final.new_tensor(0.0)

                loss = loss_ms + L_vgg + L_tv

            scaler.scale(loss / max(1, accum_steps)).backward()

            do_step = ((step + 1) % accum_steps == 0)
            if do_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)

            step += 1
            pbar.set_postfix({"step": step, "lr": lr_now, "loss": float(loss.item())})
            pbar.update(1)

            if step % meter_interval == 0 or step == max_steps:
                with torch.no_grad():
                    mets = compute_metrics(pred_final.clamp(0,1), gt.clamp(0,1), mask=None)
                history["step"].append(step)
                history["loss"].append(float(loss.item()))
                history["lr"].append(lr_now)
                history["psnr"].append(mets["psnr"])
                history["ssim"].append(mets["ssim"])

            if step % save_interval == 0 or step == max_steps:
                _save_ckpt(os.path.join(run_tmp_dir, f"refiner_step{step}.pth"), model, opt, scaler, step, cfg, ema=ema)
                _save_ckpt(os.path.join(run_tmp_dir, "refiner_latest.pth"), model, opt, scaler, step, cfg, ema=ema)
                torch.save({"model": ema.ema.state_dict(), "step": step, "cfg": cfg}, os.path.join(run_tmp_dir, "refiner_ema_latest.pth"))

            # 验证
            if dl_val is not None and (step % val_interval == 0 or step == max_steps):
                avg_mets, samples = evaluate_refiner(ema.ema, dl_val, device, max_batches=val_max_batches)
                save_json(avg_mets, os.path.join(run_tmp_dir, f"val_metrics_step{step}.json"))
                if samples:
                    save_eval_samples(samples, os.path.join(run_tmp_dir, f"val_samples_step{step}.png"))
                # 保存best（按PSNR）
                cur_psnr = avg_mets.get("psnr", -1.0) or -1.0
                if cur_psnr > best_psnr:
                    best_psnr = cur_psnr
                    _save_ckpt(os.path.join(run_tmp_dir, "refiner_best.pth"), model, opt, scaler, step, cfg, ema=ema)
                    torch.save({"model": ema.ema.state_dict(), "step": step, "cfg": cfg, "psnr": best_psnr}, os.path.join(run_tmp_dir, "refiner_ema_best.pth"))

            if step >= max_steps:
                break
    pbar.close()

    # 结束时评估（优先使用验证集，其次测试集，最后训练集）
    eval_loader = None
    if dl_val is not None:
        eval_loader = dl_val
    elif dl_test is not None:
        eval_loader = dl_test
    else:
        eval_loader = dl

    avg_mets, samples = evaluate_refiner(ema.ema, eval_loader, device, max_batches=val_max_batches)
    save_json(avg_mets, os.path.join(run_tmp_dir, "final_eval_metrics.json"))
    if samples:
        save_eval_samples(samples, os.path.join(run_tmp_dir, "final_eval_samples.png"))

    # 单独对测试集评估（若存在）
    if dl_test is not None:
        test_mets, test_samples = evaluate_refiner(ema.ema, dl_test, device, max_batches=test_max_batches)
        save_json(test_mets, os.path.join(run_tmp_dir, "test_metrics.json"))
        if test_samples:
            save_eval_samples(test_samples, os.path.join(run_tmp_dir, "test_samples.png"))

    plot_curves(history, out_dir=run_tmp_dir)
    save_csv(history, os.path.join(run_tmp_dir, "train_log.csv"))

    final_dir = finalize_run_dir(run_tmp_dir, base_ckpt_dir, model_name)
    logger.info(f"Refiner training completed. All artifacts saved to: {final_dir}")
