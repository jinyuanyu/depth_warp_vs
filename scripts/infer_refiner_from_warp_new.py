# depth_warp_vs/scripts/infer_refiner_from_warp_new.py
import argparse, os, cv2, yaml, torch, numpy as np, re
from typing import Dict, List, Tuple
from contextlib import nullcontext
from depth_warp_vs.models.refiner import build_refiner

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
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    m = (m > 127).astype(np.uint8)
    return m

def _morph_kernel(shape: str, ksize: int):
    k = max(1, int(ksize))
    s = str(shape).strip().lower()
    if s == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    elif s == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def _process_hole_mask(hole01, ksize=5, close_iters=0, open_iters=0):
    hole01 = hole01.astype(np.uint8)
    ksize = max(1, int(ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if close_iters and close_iters > 0:
        hole01 = cv2.morphologyEx(hole01, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    if open_iters and open_iters > 0:
        hole01 = cv2.morphologyEx(hole01, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))
    return hole01.astype(np.uint8)

def _build_ring_band(hole01: np.ndarray, band_px: int, shape: str = "ellipse") -> np.ndarray:
    # 基于膨胀-差分的环形污染带：band = dilate(hole) \ hole
    band_px = max(0, int(band_px))
    if band_px == 0:
        return np.zeros_like(hole01, dtype=np.uint8)
    # 使用直径为 2*band_px+1 的核，近似生成“半径=band_px”的环形带
    ksize = 2 * band_px + 1
    kernel = _morph_kernel(shape, ksize)
    dilated = cv2.dilate(hole01.astype(np.uint8), kernel, iterations=1)
    band = ((dilated > 0) & (hole01 == 0)).astype(np.uint8)
    return band

def _parse_s_index(filename: str) -> int:
    m = re.search(r'_s(\d+)(?:_|\.|$)', filename)
    if not m:
        return -1
    return int(m.group(1))

def _list_pairs(input_dir: str) -> List[Tuple[int, str, str]]:
    warp_map, hole_map = {}, {}
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
    return [(i, warp_map[i], hole_map[i]) for i in common]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _load_refiner_weights(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Refiner ckpt not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ck.get("model", ck)
    cleaned = {}
    for k, v in state.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned[new_k] = v
    res = model.load_state_dict(cleaned, strict=False)
    if getattr(res, "missing_keys", None):
        print(f"警告：权重中缺少以下键：{res.missing_keys}")
    if getattr(res, "unexpected_keys", None):
        print(f"警告：权重中存在未使用的键：{res.unexpected_keys}")

def _save_mask_png(path: str, m: np.ndarray, expect_hw: Tuple[int, int] = None):
    if m is None:
        return
    m8 = m.astype(np.uint8)
    if m8.ndim == 3 and m8.shape[2] == 1:
        m8 = m8[:, :, 0]
    if m8.max() <= 1:
        m8 = (m8 * 255).astype(np.uint8)
    if expect_hw is not None and (m8.shape[0] != expect_hw[0] or m8.shape[1] != expect_hw[1]):
        m8 = cv2.resize(m8, (expect_hw[1], expect_hw[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, m8)

def _save_rgb_or_bgr_png(path: str, img_rgb_u8: np.ndarray, to_bgr: bool = True):
    if img_rgb_u8 is None:
        return
    if img_rgb_u8.dtype != np.uint8:
        img_rgb_u8 = img_rgb_u8.astype(np.uint8)
    if img_rgb_u8.ndim == 2:
        img_rgb_u8 = cv2.cvtColor(img_rgb_u8, cv2.COLOR_GRAY2RGB)
    if img_rgb_u8.shape[2] == 4:
        img_rgb_u8 = img_rgb_u8[:, :, :3]
    out = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR) if to_bgr else img_rgb_u8
    cv2.imwrite(path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--mask_white_is_hole", type=int, default=1, help="1: 白色为空洞; 0: 黑色为空洞")

    # mask形态学清理（空洞内部）
    ap.add_argument("--mask_ksize", type=int, default=5, help="开闭运算核大小")
    ap.add_argument("--mask_close_iters", type=int, default=0, help="闭运算迭代次数")
    ap.add_argument("--mask_open_iters", type=int, default=0, help="开运算迭代次数")

    # 环形污染带参数（不再区分方向）
    ap.add_argument("--band_px", type=int, default=6, help="污染带半径（像素），将围绕空洞四周生成环形带")

    # 可选空洞膨胀/腐蚀（谨慎）
    ap.add_argument("--enable_dilate", type=int, default=0)
    ap.add_argument("--dilate_ksize", type=int, default=3)
    ap.add_argument("--dilate_iters", type=int, default=1)
    ap.add_argument("--dilate_shape", type=str, default="ellipse")
    ap.add_argument("--enable_erode", type=int, default=0)
    ap.add_argument("--erode_ksize", type=int, default=3)
    ap.add_argument("--erode_iters", type=int, default=1)
    ap.add_argument("--erode_shape", type=str, default="ellipse")

    # 调整污染带在推理中的处理方式
    ap.add_argument("--band_as_edit", type=int, default=0, help="1: 将污染带纳入编辑区域并完全替换; 0: 仅作为引导，可选软融合")
    ap.add_argument("--band_alpha", type=float, default=0.0, help="污染带区域的软融合系数(0~1)，0表示不改动原图，1表示完全采用网络输出")

    ap.add_argument("--save_debug", type=int, default=0)
    ap.add_argument("--debug_dir", type=str, default="")

    args = ap.parse_args()

    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    size = cfg.get("infer", {}).get("img_size", [0, 0])
    size = None if (not isinstance(size, (list, tuple)) or size[0] == 0 or size[1] == 0) else tuple(size)

    pairs = _list_pairs(args.input_dir)
    if not pairs:
        raise FileNotFoundError("未找到 warp_ 与 hole_ 成对图片。")
    _ensure_dir(args.out_dir)

    save_debug = int(args.save_debug) == 1
    debug_dir = args.debug_dir if args.debug_dir else os.path.join(args.out_dir, "debug")
    if save_debug:
        _ensure_dir(debug_dir)

    net = build_refiner(cfg).to(device).eval()
    ckpt = args.ckpt if args.ckpt else cfg.get("infer", {}).get("ckpt", "")
    _load_refiner_weights(net, ckpt, device=device)

    use_amp = bool(cfg.get("infer", {}).get("amp", True)) and str(device).startswith("cuda")
    dtype = torch.float16 if cfg.get("precision", "fp16") == "fp16" else torch.bfloat16
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()

    print(f"共 {len(pairs)} 对，开始推理...")
    for idx, (s_idx, warp_path, hole_path) in enumerate(pairs, 1):
        Iw = _read_rgb(warp_path, size=size).to(device)
        mgray = cv2.imread(hole_path, cv2.IMREAD_GRAYSCALE)
        if mgray is None:
            print(f"[{idx}] 跳过：无法读取mask: {hole_path}")
            continue
        mgray = _resize_mask(mgray, size)
        hole01 = _to_binary01(mgray)

        H, W = Iw.shape[-2], Iw.shape[-1]
        stem = os.path.splitext(os.path.basename(warp_path))[0]

        if save_debug:
            _save_mask_png(os.path.join(debug_dir, f"{stem}_mask_gray.png"), mgray, expect_hw=(H, W))
            _save_mask_png(os.path.join(debug_dir, f"{stem}_mask_bin.png"), hole01, expect_hw=(H, W))

        if int(args.mask_white_is_hole) == 0:
            hole01 = 1 - hole01

        # 1) 空洞内部清理（先闭后开）
        hole01 = _process_hole_mask(hole01, ksize=args.mask_ksize,
                                    close_iters=args.mask_close_iters,
                                    open_iters=args.mask_open_iters)
        if save_debug:
            _save_mask_png(os.path.join(debug_dir, f"{stem}_mask_morph.png"), hole01, expect_hw=(H, W))

        # 2) 可选膨胀/腐蚀（谨慎）：仅对空洞做
        if int(args.enable_dilate) == 1:
            kernel = _morph_kernel(args.dilate_shape, args.dilate_ksize)
            hole01 = (cv2.dilate(hole01.astype(np.uint8), kernel, iterations=int(args.dilate_iters)) > 0).astype(np.uint8)
            if save_debug:
                _save_mask_png(os.path.join(debug_dir, f"{stem}_mask_dilate.png"), hole01, expect_hw=(H, W))
        if int(args.enable_erode) == 1:
            kernel = _morph_kernel(args.erode_shape, args.erode_ksize)
            hole01 = (cv2.erode(hole01.astype(np.uint8), kernel, iterations=int(args.erode_iters)) > 0).astype(np.uint8)
            if save_debug:
                _save_mask_png(os.path.join(debug_dir, f"{stem}_mask_erode.png"), hole01, expect_hw=(H, W))

        # 3) 基于最终空洞生成“环形污染带”
        band01 = _build_ring_band(hole01, band_px=int(args.band_px), shape="ellipse")
        if save_debug:
            _save_mask_png(os.path.join(debug_dir, f"{stem}_mask_band.png"), band01, expect_hw=(H, W))

        # 4) 并集（用于 band_as_edit=1 的旧行为）
        union01 = ((hole01 > 0) | (band01 > 0)).astype(np.uint8)
        if save_debug:
            _save_mask_png(os.path.join(debug_dir, f"{stem}_mask_union.png"), union01, expect_hw=(H, W))

        # 构建mask张量（空洞与污染带分别为独立通道，引导网络注意力）
        Mk_h = torch.from_numpy(hole01.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        Mk_p = torch.from_numpy(band01.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        Mk_u = torch.from_numpy(union01.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        # 仅对“编辑区域”清零输入：默认只清零空洞
        band_as_edit = int(args.band_as_edit) == 1
        Mk_edit = Mk_u if band_as_edit else Mk_h

        Iw_proc = (1.0 - Mk_edit) * Iw
        if save_debug:
            Iw_proc_np = (Iw_proc.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            _save_rgb_or_bgr_png(os.path.join(debug_dir, f"{stem}_Iw_proc.png"), Iw_proc_np, to_bgr=True)

        # 网络输入：[Iw_proc, Mk_h, Mk_p] => 5通道
        x = torch.cat([Iw_proc, Mk_h, Mk_p], dim=1)
        with torch.no_grad():
            with amp_ctx:
                Ipred = net(x)

        # 融合策略：
        # - 默认：仅在空洞内强制替换
        # - band_as_edit=1：空洞并集强制替换（与旧脚本一致）
        # - 可选：对污染带使用软融合（alpha）
        if band_as_edit:
            Ifinal = Mk_u * Ipred + (1.0 - Mk_u) * Iw
        else:
            # 仅空洞内替换
            Ifinal = Mk_h * Ipred + (1.0 - Mk_h) * Iw
            # 污染带软融合（可选）
            alpha = float(args.band_alpha)
            if alpha > 0.0:
                alpha = max(0.0, min(1.0, alpha))
                band3 = Mk_p.expand(-1, 3, -1, -1)
                soft_band = alpha * Ipred + (1.0 - alpha) * Iw
                Ifinal = band3 * soft_band + (1.0 - band3) * Ifinal

        out = (Ifinal.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out_name = "refine_" + stem[5:] + ".png" if stem.startswith("warp_") else stem + "_refine.png"
        cv2.imwrite(os.path.join(args.out_dir, out_name), out_bgr)
        print(f"[{idx}/{len(pairs)}] Saved: {out_name}")
    print("完成。")

if __name__ == "__main__":
    main()
