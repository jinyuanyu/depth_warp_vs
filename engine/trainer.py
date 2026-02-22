# depth_warp_vs/engine/trainer.py
import os, math, yaml
import torch
from torch import optim
from tqdm import tqdm
from models.route_a_model import RouteAModel
from models.losses.photometric import recon_loss
from models.losses.perceptual import VGGPerceptual
from models.losses.regularizers import tv, edge_aware
from engine.logger import get_logger
from engine.seed import set_seed
from data.build import build_dataloader
from models.utils.amp import AmpScaler

def build_optimizer(params, cfg):
    return optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"], betas=tuple(cfg["optim"]["betas"]))

def cosine_lr(base_lr, step, warmup, max_steps):
    if step < warmup:
        return base_lr * float(step) / max(1, warmup)
    t = (step - warmup) / max(1, (max_steps - warmup))
    t = min(max(t, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))

def adjust_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr

def save_ckpt(path, model, optimizer, scaler, step, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.scaler.state_dict() if scaler is not None else None,
        "step": step,
        "cfg": cfg
    }, path)

def load_ckpt(path, model, optimizer=None, scaler=None, map_location="cpu", strict=False):
    ck = torch.load(path, map_location=map_location)
    state = ck.get("model", ck)
    new_state = {}
    for k,v in state.items():
        new_state[k[7:]] = v if k.startswith("module.") else v
    model.load_state_dict(new_state, strict=strict)
    if optimizer is not None and ck.get("optimizer", None) is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler", None) is not None:
        scaler.scaler.load_state_dict(ck["scaler"])
    step = ck.get("step", 0)
    return step

def train(cfg_path: str):
    logger = get_logger()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Model / losses / opt / amp
    model = RouteAModel(cfg).to(device)
    per_loss = VGGPerceptual().to(device)
    opt = build_optimizer(model.parameters(), cfg)
    scaler = AmpScaler(enabled=bool(cfg.get("amp", True)))

    # Data
    dl = build_dataloader(cfg, split=cfg["data"].get("split","train"))

    # Resume or pretrained
    start_step = 0
    resume = cfg.get("log", {}).get("resume", "")
    pretrained = cfg.get("log", {}).get("pretrained", "")
    if resume and os.path.isfile(resume):
        start_step = load_ckpt(resume, model, optimizer=opt, scaler=scaler, map_location=device)
        logger.info(f"Resumed from {resume} at step={start_step}")
    elif pretrained and os.path.isfile(pretrained):
        load_ckpt(pretrained, model, map_location=device, strict=False)
        logger.info(f"Loaded pretrained weights from {pretrained}")

    # Train
    model.train()
    max_steps = int(cfg["optim"].get("max_steps", 100000))
    warmup = int(cfg["optim"].get("warmup_steps", 1000))
    base_lr = float(cfg["optim"]["lr"])
    save_interval = int(cfg["log"].get("save_interval", 2000))
    ckpt_dir = cfg["log"].get("ckpt_dir", "./checkpoints")
    step = start_step
    pbar = tqdm(total=max_steps - step, initial=0, desc="Training")

    while step < max_steps:
        for batch in dl:
            step += 1
            Is, Ds, Ks, Kt, dT, It_gt = batch
            Is, Ds, Ks, Kt, dT, It_gt = Is.to(device), Ds.to(device), Ks.to(device), Kt.to(device), dT.to(device), It_gt.to(device)

            lr_now = cosine_lr(base_lr, step, warmup, max_steps) if cfg["optim"].get("cosine", True) else base_lr
            adjust_lr(opt, lr_now)
            opt.zero_grad(set_to_none=True)

            with scaler.autocast(dtype=torch.float16 if cfg.get("precision","fp16")=="fp16" else torch.bfloat16):
                It, aux = model(Is, Ds, Ks, Kt, dT)
                # 全图监督：网络需对空洞完成填补
                mask = None
                L_photo = recon_loss(It, It_gt, mask=mask, lam_l1=cfg["loss"]["l1"], lam_ssim=cfg["loss"]["ssim"])
                L_vgg = per_loss(It, It_gt, mask)
                L_tv = tv(aux["dgrid"])
                L_edge = edge_aware(aux["dgrid"], Is)
                loss = L_photo + cfg["loss"]["vgg"]*L_vgg + cfg["loss"]["tv"]*L_tv + cfg["loss"]["edgeaware"]*L_edge

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix({"step": step, "lr": lr_now, "loss": float(loss.item())})
            pbar.update(1)

            if step % save_interval == 0 or step == max_steps:
                path = os.path.join(ckpt_dir, f"routea_step{step}.pth")
                save_ckpt(path, model, opt, scaler, step, cfg)
                save_ckpt(os.path.join(ckpt_dir, "routea_latest.pth"), model, opt, scaler, step, cfg)

            if step >= max_steps:
                break
    pbar.close()
    logger.info("Training completed.")

if __name__ == "__main__":
    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {"train_dataset":"mannequin","root":"./MannequinChallenge","split":"train","img_size":[128,128],
                 "neighbors":[-1,1], "window":3, "use_pose_pnp": False, "batch_size": 2},
        "model": {
            "residual_flow_net":{"in_ch":5,"base_ch":16},
            "refiner":{"in_ch":5,"out_ch":3,"depth":3,"base_ch":16},
            "softmax_splat":{"temperature":10.0,"normalize":True},
            "grid":{"align_corners":True}
        },
        "loss":{"l1":1.0,"ssim":0.2,"vgg":0.1,"tv":0.05,"edgeaware":0.05},
        "optim":{"lr":1e-4,"weight_decay":0.01,"betas":[0.9,0.999], "max_steps":10}
    }
    import yaml
    with open("temp_train.yaml","w") as f: yaml.dump(cfg,f)
    train("temp_train.yaml")
