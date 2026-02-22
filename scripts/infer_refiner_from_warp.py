# depth_warp_vs/scripts/infer_refiner_from_warp.py
import argparse, os, cv2, yaml, torch, numpy as np, re
from typing import Dict, List, Tuple
from contextlib import nullcontext

from depth_warp_vs.models.refiner.inpaint_refiner import InpaintRefiner

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _read_rgb(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return ten

def _resize_mask(m, size):
    if size is not None:
        m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return m

def _to_binary01(m):
    # 输入可为0~255或0/1，统一转为0/1 uint8
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    m = (m > 127).astype(np.uint8)
    return m

def _remove_small_components(bin01, min_area, target_value=1, connectivity=4):
    # 对值==target_value的连通域，移除面积小于min_area的斑点
    if min_area <= 0:
        return bin01
    work = (bin01 == target_value).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=connectivity)
    for lab in range(1, num_labels):  # 0是背景
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < min_area:
            bin01[labels == lab] = 1 - target_value
    return bin01

def _process_hole_mask(hole01, ksize=5, close_iters=1, open_iters=0, rm_white_area=0, rm_black_area=0):
    # hole01: 0/1，1表示空洞
    hole01 = hole01.astype(np.uint8)
    ksize = max(1, int(ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if close_iters > 0:
        hole01 = cv2.morphologyEx(hole01, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    if open_iters > 0:
        hole01 = cv2.morphologyEx(hole01, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))
    hole01 = _remove_small_components(hole01, min_area=int(rm_white_area), target_value=1, connectivity=4)
    inv = 1 - hole01
    inv = _remove_small_components(inv, min_area=int(rm_black_area), target_value=1, connectivity=4)
    hole01 = 1 - inv
    return hole01.astype(np.uint8)

def _parse_s_index(filename: str) -> int:
    """
    从文件名中解析 s 后面的序号。例如:
    - warp_49320000_s00_tx-0.25105.png -> 返回 0
    - hole_s12_tx-0.1.png -> 返回 12
    匹配模式：_*s(\d+)[_\.]
    """
    m = re.search(r'_s(\d+)(?:_|\.|$)', filename)
    if not m:
        return -1
    return int(m.group(1))

def _list_pairs(input_dir: str) -> List[Tuple[int, str, str]]:
    """
    遍历 input_dir，寻找 warp_ 和 hole_ 文件，按 s 序号匹配成对。
    返回列表：[(s_idx, warp_path, hole_path), ...]，按 s_idx 升序。
    """
    warp_map: Dict[int, str] = {}
    hole_map: Dict[int, str] = {}

    for name in os.listdir(input_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in VALID_EXTS:
            continue
        full = os.path.join(input_dir, name)
        if not os.path.isfile(full):
            continue

        s_idx = _parse_s_index(name)
        if s_idx < 0:
            continue
        if base.startswith("warp_"):
            warp_map[s_idx] = full
        elif base.startswith("hole_"):
            hole_map[s_idx] = full

    common = sorted(set(warp_map.keys()) & set(hole_map.keys()))
    pairs = [(i, warp_map[i], hole_map[i]) for i in common]
    missing_warp = sorted(set(hole_map.keys()) - set(warp_map.keys()))
    missing_hole = sorted(set(warp_map.keys()) - set(hole_map.keys()))
    if missing_warp:
        print(f"警告：以下 s 序号缺少 warp_ 文件：{missing_warp}")
    if missing_hole:
        print(f"警告：以下 s 序号缺少 hole_ 文件：{missing_hole}")
    return pairs

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _load_refiner_weights(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Refiner ckpt not found: {ckpt_path}")
    # 兼容不同torch版本，不强制使用 weights_only
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)

    # 兼容直接保存state_dict或trainer保存的dict
    state = ck.get("model", ck)

    # 安全去除可能的"module."前缀
    cleaned = {}
    for k, v in state.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"警告：权重中缺少以下键：{missing}")
    if unexpected:
        print(f"警告：权重中存在未使用的键：{unexpected}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 配置文件")
    ap.add_argument("--input_dir", required=True, help="包含 warp_ 与 hole_ 成对图片的输入文件夹")
    ap.add_argument("--out_dir", required=True, help="输出文件夹（自动命名保存）")
    ap.add_argument("--ckpt", default="", help="Refiner 权重文件路径，可覆盖 config 中的 infer.ckpt")

    # mask处理参数
    ap.add_argument("--mask_white_is_hole", type=int, default=1, help="输入mask里白色是否表示空洞(1是,0否)")
    ap.add_argument("--mask_ksize", type=int, default=5, help="形态学核大小")
    ap.add_argument("--mask_close_iters", type=int, default=1, help="闭运算迭代次数(填补白色空洞中的小黑洞)")
    ap.add_argument("--mask_open_iters", type=int, default=0, help="开运算迭代次数(去除零星白点)")
    ap.add_argument("--mask_rm_white_area", type=int, default=0, help="移除面积小于阈值的白色连通域(空洞斑点)")
    ap.add_argument("--mask_rm_black_area", type=int, default=0, help="填平空洞内部面积小于阈值的黑色小洞")
    ap.add_argument("--save_processed_mask_dir", default="", help="可选: 处理后的空洞mask输出文件夹(逐张保存png)")

    args = ap.parse_args()

    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    size = cfg.get("infer", {}).get("img_size", [0, 0])
    size = None if (not isinstance(size, (list, tuple)) or size[0] == 0 or size[1] == 0) else tuple(size)

    torch.backends.cudnn.benchmark = True if str(device).startswith("cuda") else False

    # 列出成对图片
    pairs = _list_pairs(args.input_dir)
    if not pairs:
        raise FileNotFoundError("在输入文件夹中未找到任何 warp_/hole_ 成对图片（按 s 序号匹配）。")

    _ensure_dir(args.out_dir)
    if args.save_processed_mask_dir:
        _ensure_dir(args.save_processed_mask_dir)

    # 加载网络
    mcfg = cfg["model"]["refiner"]
    net = InpaintRefiner(
        in_ch=mcfg.get("in_ch", 4),
        base_ch=mcfg.get("base_ch", 48),
        depth=mcfg.get("depth", 5)
    ).to(device).eval()

    ckpt = args.ckpt if args.ckpt else cfg.get("infer", {}).get("ckpt", "")
    _load_refiner_weights(net, ckpt, device=device)

    # AMP 设置
    infer_cfg = cfg.get("infer", {})
    use_amp = bool(infer_cfg.get("amp", True)) and str(device).startswith("cuda")
    prec = infer_cfg.get("precision", "fp16")
    amp_dtype = torch.float16 if prec == "fp16" else torch.bfloat16
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    print(f"共匹配到 {len(pairs)} 对图片，开始推理...")
    for idx, (s_idx, warp_path, hole_path) in enumerate(pairs, 1):
        base_warp = os.path.basename(warp_path)
        stem_warp, _ = os.path.splitext(base_warp)

        # 自动输出命名：warp_xxx.png -> refine_xxx.png；否则加后缀
        if stem_warp.startswith("warp_"):
            out_name = "refine_" + stem_warp[len("warp_"):] + ".png"
        else:
            out_name = stem_warp + "_refine.png"
        out_path = os.path.join(args.out_dir, out_name)

        # 读入 warp
        Iw = _read_rgb(warp_path, size=size)

        # 读入并处理 mask
        mgray = cv2.imread(hole_path, cv2.IMREAD_GRAYSCALE)
        if mgray is None:
            print(f"[{idx}/{len(pairs)}][s{s_idx:02d}] 跳过：无法读取mask {hole_path}")
            continue
        mgray = _resize_mask(mgray, size)
        mbin = _to_binary01(mgray)  # 0/1，当前白=1
        hole01 = mbin if args.mask_white_is_hole else (1 - mbin)  # 统一 holes=1
        hole01 = _process_hole_mask(
            hole01,
            ksize=args.mask_ksize,
            close_iters=args.mask_close_iters,
            open_iters=args.mask_open_iters,
            rm_white_area=args.mask_rm_white_area,
            rm_black_area=args.mask_rm_black_area
        )

        # 可选保存处理后的mask
        if args.save_processed_mask_dir:
            mask_name = f"maskproc_s{s_idx:02d}.png"
            cv2.imwrite(os.path.join(args.save_processed_mask_dir, mask_name), (hole01 * 255).astype(np.uint8))

        # 转为tensor并融合
        Mk = torch.from_numpy(hole01.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        Iin = Iw * (1.0 - Mk)  # 空洞区域置黑
        x = torch.cat([Iin, Mk], dim=1).to(device)

        # 推理
        with torch.no_grad():
            with amp_ctx:
                It = net(x)

        out = (It.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out_bgr)
        print(f"[{idx}/{len(pairs)}][s{s_idx:02d}] Saved: {out_path}")

    print("全部处理完成。")

if __name__ == "__main__":
    main()


    # state = torch.load(ckpt, map_location=device, weights_only=True)
    # missing, unexpected = net.load_state_dict(state['model'], strict=True)
    #
    # # 兼容module.前缀
    # state = state.get("model", state)
    # new_state = {}
    # for k, v in state.items():
    #     new_state[k[7:]] = v if k.startswith("module.") else v
    # net.load_state_dict(new_state, strict=False)
    #
    # with torch.no_grad():
    #     use_amp = bool(cfg.get("infer",{}).get("amp", True)) and device.startswith("cuda")
    #     with torch.amp.autocast('cuda', enabled=use_amp):
    #         It = net(x)
    #
    # out = (It.clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
    # out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(args.out, out_bgr)
    # print(f"Saved refined image to {args.out}")

if __name__ == "__main__":
    main()
# (DWvs) D:\naked-eye 3D\video call>python -m depth_warp_vs.scripts.infer_refiner_from_warp --config depth_warp_vs/configs/infer_refiner.yaml --warp "D:\naked-eye 3D\video call\depth_warp_vs\data\datasets\MannequinChallenge\test\00c4a2d23c90fbc9\sim_warp\warp_157924000.png" --mask "D:\naked-eye 3D\video call\depth_warp_vs\data\datasets\MannequinChallenge\test\00c4a2d23c90fbc9\hole_mask\hole_157924000.png" --out out.png
# (DWvs) D:\naked-eye 3D\video call>python -m depth_warp_vs.scripts.infer_refiner_from_warp --config depth_warp_vs/configs/infer_refiner.yaml --warp "D:\naked-eye 3D\video call\depth_warp_vs\output\warp_49320000_s00_tx-0.25105.png" --mask "D:\naked-eye 3D\video call\depth_warp_vs\output\hole_s00_tx-0.25105.png" --out s00_tx-0.25105.png --save_processed_mask "D:\naked-eye 3D\video call\depth_warp_vs\output\phole_s00_tx-0.25105.png"