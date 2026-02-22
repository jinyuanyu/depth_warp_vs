# depth_warp_vs/scripts/prepare_simwarp_cli.py
import os
import argparse
import cv2
import numpy as np
from typing import Tuple, List

def read_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img  # BGR uint8

import numpy as np
import cv2
import matplotlib

def _color_depth_to_scalar_bgr(dep_bgr: np.ndarray, method: str = "pca") -> np.ndarray:
    dep_bgr = dep_bgr.astype(np.uint8)
    h, w, _ = dep_bgr.shape

    if method == "r":
        v = dep_bgr[..., 2] / 255.0
    elif method == "g":
        v = dep_bgr[..., 1] / 255.0
    elif method == "b":
        v = dep_bgr[..., 0] / 255.0
    elif method == "luma":
        v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    elif method == "rgb24":
        R = dep_bgr[..., 2].astype(np.float32)
        G = dep_bgr[..., 1].astype(np.float32)
        B = dep_bgr[..., 0].astype(np.float32)
        v = (R * 65536.0 + G * 256.0 + B) / (255.0 * 65536.0 + 255.0 * 256.0 + 255.0)
    elif method == "auto":
        dep_bgr_f = dep_bgr.astype(np.float32)
        diff_rg = np.mean(np.abs(dep_bgr_f[..., 2] - dep_bgr_f[..., 1]))
        diff_gb = np.mean(np.abs(dep_bgr_f[..., 1] - dep_bgr_f[..., 0]))
        diff_rb = np.mean(np.abs(dep_bgr_f[..., 2] - dep_bgr_f[..., 0]))
        if (diff_rg + diff_gb + diff_rb) / 3.0 < 2.0:
            v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            method = "pca"

    if method == "pca":
        dep_bgr_f = dep_bgr.astype(np.float32)
        step = max(1, int(np.ceil(np.sqrt((h * w) / 10000.0))))
        sample = dep_bgr_f[::step, ::step, :].reshape(-1, 3)
        mean = sample.mean(axis=0, keepdims=True)
        X = sample - mean
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, np.argmax(eigvals)]
        Xfull = dep_bgr_f.reshape(-1, 3) - mean
        proj = Xfull @ pc1
        proj = proj.reshape(h, w).astype(np.float32)
        pmin = float(np.min(proj)); pmax = float(np.max(proj))
        if pmax - pmin < 1e-6:
            v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            v = (proj - pmin) / (pmax - pmin)

    if method == "vda":
        cm = matplotlib.colormaps.get_cmap("inferno")
        if hasattr(cm, "colors"):
            pal = np.array(cm.colors)
        else:
            pal = cm(np.linspace(0.0, 1.0, 256))[:, :3]
        if pal.shape[1] == 4:
            pal = pal[:, :3]
        palette_rgb_u8 = (pal * 255.0).round().astype(np.uint8)
        palette_bgr_u8 = palette_rgb_u8[:, ::-1]

        pixels = dep_bgr.reshape(-1, 3)

        palette_pack = (palette_bgr_u8[:, 0].astype(np.uint32) << 16) | \
                       (palette_bgr_u8[:, 1].astype(np.uint32) << 8)  | \
                        palette_bgr_u8[:, 2].astype(np.uint32)
        pixels_pack =  (pixels[:, 0].astype(np.uint32) << 16) | \
                       (pixels[:, 1].astype(np.uint32) << 8)  | \
                        pixels[:, 2].astype(np.uint32)
        order = np.argsort(palette_pack)
        palette_sorted = palette_pack[order]
        pos = np.searchsorted(palette_sorted, pixels_pack)
        match_mask = (pos < palette_sorted.size) & (palette_sorted[pos] == pixels_pack)

        out_idx = np.empty(pixels.shape[0], dtype=np.int32)
        out_idx[match_mask] = order[pos[match_mask]]

        if np.any(~match_mask):
            palette_f = palette_bgr_u8.astype(np.float32)
            p_norm2 = np.sum(palette_f * palette_f, axis=1)
            pixels_f = pixels[~match_mask].astype(np.float32)
            nearest_idx = np.empty(pixels_f.shape[0], dtype=np.int32)
            block_size = 32768
            k = 0
            while k < pixels_f.shape[0]:
                block = pixels_f[k:k + block_size]
                scores = block @ palette_f.T
                scores -= 0.5 * p_norm2[None, :]
                nearest_idx[k:k + block_size] = np.argmax(scores, axis=1)
                k += block_size
            out_idx[~match_mask] = nearest_idx

        v = (out_idx.astype(np.float32) / float(palette_bgr_u8.shape[0] - 1)).reshape(h, w)

    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    return v

def read_depth_auto(path: str,
                    mode: str = "auto",
                    scale: float = 10.0,
                    color_decode: str = "auto",
                    depth_min_m: float = 2.0,
                    depth_max_m: float = 10.0) -> np.ndarray:
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(path)

    dm = float(depth_min_m); dM = float(depth_max_m)
    if dM <= dm: dM = dm + 1.0

    if dep.ndim == 3:
        v01 = _color_depth_to_scalar_bgr(dep, method=color_decode)
        if mode == "metric":
            dep_f = dm + v01 * (dM - dm)
        elif mode == "normalized":
            dep_f = dm + np.clip(v01, 1e-6, 1.0) * (dM - dm)
        else:
            dep_f = dm + np.clip(v01, 1e-6, 1.0) * (dM - dm)
        return dep_f.astype(np.float32)

    dep = dep.astype(np.float32)
    if mode == "metric":
        if dep.max() > 255.0:
            dep = dep / 1000.0
        dep = np.clip(dep, dm, dM)
    elif mode == "normalized":
        if dep.max() > 1.0:
            dep = dep / 255.0
        dep = dm + np.clip(dep, 1e-6, 1.0) * (dM - dm)
    else:
        if dep.max() > 255.0:
            dep = dep / 1000.0
        else:
            if dep.max() > 1.0:
                dep = dep / 255.0
            dep = dm + np.clip(dep, 1e-6, 1.0) * (dM - dm)
    return dep.astype(np.float32)

def robust_threshold(diff_abs: np.ndarray, k: float, min_tau: float) -> float:
    med = np.median(diff_abs)
    tau = max(min_tau, k * med)
    return float(tau)

def _nms1d_along_x(mag: np.ndarray, rel_margin: float = 0.0) -> np.ndarray:
    H, Wm1 = mag.shape
    nms = np.zeros_like(mag, dtype=bool)
    if Wm1 < 3:
        return nms
    c = mag[:, 1:-1]
    l = mag[:, :-2]
    r = mag[:, 2:]
    thr_l = (1.0 + float(rel_margin)) * l
    thr_r = (1.0 + float(rel_margin)) * r
    nms[:, 1:-1] = (c >= thr_l) & (c >= thr_r)
    return nms

def _detect_abrupt_edges_horiz(depth: np.ndarray,
                               edge_k: float,
                               min_tau: float,
                               curv_k: float,
                               curv_min_tau: float,
                               nms_rel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = depth.shape
    dl = depth[:, :-1]
    dr = depth[:, 1:]
    diff = dr - dl
    mag = np.abs(diff)

    tau_grad = robust_threshold(mag, k=edge_k, min_tau=min_tau)
    strong_grad = mag > tau_grad

    kernel = np.array([[1.0, -2.0, 1.0]], dtype=np.float32)
    curv = cv2.filter2D(depth, cv2.CV_32F, kernel, borderType=cv2.BORDER_DEFAULT)
    curv_edge = np.maximum(curv[:, :-1], curv[:, 1:])
    curv_edge = np.abs(curv_edge)
    tau_curv = robust_threshold(curv_edge, k=curv_k, min_tau=curv_min_tau)
    strong_curv = curv_edge > tau_curv

    nms_ok = _nms1d_along_x(mag, rel_margin=nms_rel)
    if nms_ok.shape[1] >= 1:
        nms_ok[:, 0] = strong_grad[:, 0] & strong_curv[:, 0]
        nms_ok[:, -1] = strong_grad[:, -1] & strong_curv[:, -1]

    strong_mask = strong_grad & strong_curv & nms_ok
    diff_far = diff
    return strong_mask, mag, diff_far

def simulate_horizontal_holes_with_pollution(
    rgb_bgr: np.ndarray,
    depth: np.ndarray,
    add_direction: str = "right",
    min_px: int = 2,
    max_px: int = 24,
    band_min_px: int = 1,
    band_max_px: int = 8,
    band_ratio: float = 0.5,
    edge_k: float = 6.0,
    min_tau: float = 1e-3,
    curv_k: float = 6.0,
    curv_min_tau: float = 1e-6,
    nms_rel: float = 0.0,
    dilate_px: int = 1,
    depth_greater_is_far: bool = True,
    smooth_ksize: int = 3,
    smooth_iters: int = 1,
    pollute_alpha: float = 0.8,
    pollute_blur_ksize: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      - sim_bgr: 模拟warp后的图像（空洞置黑，沿方向的污染带已施加颜色）
      - hole_mask: uint8 {0,255}
      - pollute_mask: uint8 {0,255}，为环形带（围绕空洞四周），宽度近似等于模拟时的污染带宽度
      - edit_mask: uint8 {0,255} = hole ∪ pollute_ring
    """
    assert add_direction in ("left", "right")
    H, W = depth.shape

    dep_valid = depth[np.isfinite(depth)]
    if dep_valid.size == 0:
        dep_valid = depth.reshape(-1)
    p10 = np.percentile(dep_valid, 10.0)
    p90 = np.percentile(dep_valid, 90.0)
    if p90 <= p10:
        p90 = p10 + 1e-3

    strong_mask, mag, diff_far = _detect_abrupt_edges_horiz(
        depth, edge_k=edge_k, min_tau=min_tau, curv_k=curv_k, curv_min_tau=curv_min_tau, nms_rel=nms_rel
    )
    diff_far = diff_far if depth_greater_is_far else -diff_far

    dl = depth[:, :-1]
    dr = depth[:, 1:]
    if depth_greater_is_far:
        near_depth = np.minimum(dl, dr)
        near_norm = np.clip((p90 - near_depth) / (p90 - p10), 0.0, 1.0)
    else:
        near_depth = np.maximum(dl, dr)
        near_norm = np.clip((near_depth - p10) / (p90 - p10), 0.0, 1.0)

    p99 = np.percentile(mag, 99.0) if mag.size > 0 else (min_tau + 1.0)
    p99 = max(p99, min_tau + 1e-6)
    tau_grad = robust_threshold(mag, k=edge_k, min_tau=min_tau)
    contrast_norm = np.clip((mag - tau_grad) / (p99 - tau_grad), 0.0, 1.0)

    width = np.rint(min_px + (max_px - min_px) * near_norm * contrast_norm).astype(np.int32)
    width = np.clip(width, min_px, max_px)
    width = width * strong_mask.astype(np.int32)

    hole_mask = np.zeros((H, W), dtype=np.uint8)

    maxw = int(max_px)
    if depth_greater_is_far:
        ge_ok = np.greater_equal
    else:
        ge_ok = np.less_equal

    if add_direction == "right":
        right_far = strong_mask & (diff_far > 0)
        cumulative_clear = np.ones_like(right_far, dtype=bool)
        for k in range(1, maxw + 1):
            col_range = W - k
            if col_range <= 0:
                break
            step_ok = ge_ok(depth[:, k:], near_depth[:, :col_range])
            cumulative_clear[:, :col_range] &= step_ok
            cond_k = right_far[:, :col_range] & (width[:, :col_range] >= k) & cumulative_clear[:, :col_range]
            if not np.any(cond_k):
                continue
            mline = np.zeros_like(hole_mask, dtype=bool)
            mline[:, k:] = cond_k
            hole_mask[mline] = 255
    else:
        left_far = strong_mask & (diff_far < 0)
        cumulative_clear = np.ones_like(left_far, dtype=bool)
        for k in range(1, maxw + 1):
            col_range = W - k
            if col_range <= 0:
                break
            step_ok = ge_ok(depth[:, :col_range], near_depth[:, (k - 1):])
            cumulative_clear[:, (k - 1):] &= step_ok
            cond_k = left_far & (width >= k) & cumulative_clear
            cond_k = cond_k[:, (k - 1):]
            if not np.any(cond_k):
                continue
            mline = np.zeros_like(hole_mask, dtype=bool)
            mline[:, :col_range] = cond_k
            hole_mask[mline] = 255

    # 可选水平膨胀增强连贯性
    if dilate_px > 0:
        kernel = np.ones((1, int(dilate_px) * 2 + 1), np.uint8)
        hole_mask = cv2.dilate(hole_mask, kernel, iterations=1)

    # 开/闭平滑使边缘更平滑
    if smooth_ksize > 0 and smooth_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(smooth_ksize), int(smooth_ksize)))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, k, iterations=int(smooth_iters))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN,  k, iterations=int(smooth_iters))

    # 颜色污染（保持原行为：仅沿方向添加），并记录每行污染带宽度的最大值
    hole01 = (hole_mask > 0).astype(np.uint8)
    band_min_px = max(1, int(band_min_px))
    band_max_px = max(band_min_px, int(band_max_px))
    sim = rgb_bgr.copy()

    # 仅用于颜色模糊的方向性污染mask
    pollute_mask_dir = np.zeros((H, W), dtype=np.uint8)
    # 每行最大污染带宽度（为后续环形带宽度参考）
    row_band_w = np.zeros((H,), dtype=np.int32)

    for i in range(H):
        row = hole01[i]
        idx = np.where(row > 0)[0]
        if idx.size == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0]
        start_ids = np.insert(idx[splits + 1], 0, idx[0])
        end_ids   = np.append(idx[splits], idx[-1])
        for a, b in zip(start_ids, end_ids):
            hole_w = int(b - a + 1)
            band_w = int(np.clip(int(round(hole_w * float(band_ratio))), band_min_px, band_max_px))
            if band_w <= 0:
                continue
            # 记录该行的最大污染带宽度
            row_band_w[i] = max(row_band_w[i], band_w)

            if add_direction == "right":
                c0 = b + 1
                c1 = min(W - 1, b + band_w)
                if c0 > c1:
                    continue
                pollute_mask_dir[i, c0:c1+1] = 255
                src_col = max(0, a - 1)
                src_color = sim[i, src_col:src_col+1, :]
                for c in range(c0, c1 + 1):
                    dst_color = sim[i, c, :]
                    new_color = (pollute_alpha * src_color[0] + (1.0 - pollute_alpha) * dst_color).astype(np.uint8)
                    sim[i, c, :] = new_color
            else:
                c1 = max(0, a - 1)
                c0 = max(0, a - band_w)
                if c0 > c1:
                    continue
                pollute_mask_dir[i, c0:c1+1] = 255
                src_col = min(W - 1, b + 1)
                src_color = sim[i, src_col:src_col+1, :]
                for c in range(c0, c1 + 1):
                    dst_color = sim[i, c, :]
                    new_color = (pollute_alpha * src_color[0] + (1.0 - pollute_alpha) * dst_color).astype(np.uint8)
                    sim[i, c, :] = new_color

    # 对方向性污染带做可选模糊，仅作用于实际污染区域
    if pollute_blur_ksize and pollute_blur_ksize > 1:
        ksize = int(pollute_blur_ksize)
        blurred = cv2.GaussianBlur(sim, (ksize, ksize), 0)
        mask3_dir = cv2.cvtColor((pollute_mask_dir > 0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        sim = np.where(mask3_dir > 0, blurred, sim)

    # 空洞置黑
    sim[hole_mask > 0] = 0

    # 环形污染带：围绕空洞四周，厚度近似为每行最大 band_w
    # 思路：对空洞的外部做距离变换，阈值为 row_band_w（逐行广播），得到 dist in (0, row_band_w] 的环
    outside = (hole01 == 0).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3).astype(np.float32)
    # 保障每行至少有最小宽度
    row_band_w = np.maximum(row_band_w, band_min_px).astype(np.float32)
    thr_map = np.repeat(row_band_w.reshape(H, 1), W, axis=1)
    pollute_ring01 = ((dist > 0.0) & (dist <= thr_map + 1e-6)).astype(np.uint8)
    pollute_mask = (pollute_ring01.astype(np.uint8) * 255)

    edit_mask = ((hole01 > 0) | (pollute_ring01 > 0)).astype(np.uint8) * 255
    return sim, hole_mask, pollute_mask, edit_mask

def _choose_direction(index: int, total: int, strategy: str) -> str:
    if strategy == "alternate":
        return "left" if (index % 2 == 0) else "right"
    elif strategy == "first_half_left":
        left_cnt = (total + 1) // 2
        return "left" if index < left_cnt else "right"
    else:
        return "right"

def process_clip(clip_dir: str,
                 depth_dir_name="depth",
                 min_px=2,
                 max_px=24,
                 band_min_px=1,
                 band_max_px=8,
                 band_ratio=0.5,
                 edge_k=6.0,
                 min_tau=1e-3,
                 curv_k=6.0,
                 curv_min_tau=1e-6,
                 nms_rel=0.0,
                 dilate_px=1,
                 depth_mode="metric",
                 depth_scale=10.0,
                 depth_color_decode="auto",
                 assign_strategy="alternate",
                 far_value: str = "larger",
                 depth_min_m: float = 2.0,
                 depth_max_m: float = 10.0,
                 smooth_ksize: int = 3,
                 smooth_iters: int = 1,
                 pollute_alpha: float = 0.8,
                 pollute_blur_ksize: int = 0):
    depth_dir = os.path.join(clip_dir, depth_dir_name)
    if not os.path.isdir(depth_dir):
        return 0
    frames = [f for f in os.listdir(clip_dir) if f.lower().startswith("frame_") and f.lower().endswith((".jpg", ".png", ".jpeg"))]
    frames.sort()

    valid: List[str] = []
    depth_paths: List[str] = []
    for f in frames:
        stem = os.path.splitext(f)[0]
        idv = stem.replace("frame_", "")
        dep_path = os.path.join(depth_dir, f"depth_{idv}.png")
        if os.path.isfile(dep_path):
            valid.append(f)
            depth_paths.append(dep_path)
    n_valid = len(valid)
    if n_valid == 0:
        return 0

    os.makedirs(os.path.join(clip_dir, "sim_warp"), exist_ok=True)
    os.makedirs(os.path.join(clip_dir, "hole_mask"), exist_ok=True)
    os.makedirs(os.path.join(clip_dir, "pollute_mask"), exist_ok=True)
    os.makedirs(os.path.join(clip_dir, "edit_mask"), exist_ok=True)

    greater_is_far = (far_value == "larger")

    count = 0
    for idx, (f, dep_path) in enumerate(zip(valid, depth_paths)):
        stem = os.path.splitext(f)[0]
        idv = stem.replace("frame_", "")
        out_warp = os.path.join(clip_dir, "sim_warp",   f"warp_{idv}.png")
        out_hole = os.path.join(clip_dir, "hole_mask",  f"hole_{idv}.png")
        out_poll = os.path.join(clip_dir, "pollute_mask", f"pollute_{idv}.png")
        out_edit = os.path.join(clip_dir, "edit_mask",  f"edit_{idv}.png")

        add_dir = _choose_direction(idx, n_valid, assign_strategy)

        try:
            rgb = read_rgb(os.path.join(clip_dir, f))
            dep = read_depth_auto(dep_path, mode=depth_mode, color_decode=depth_color_decode,
                                  depth_min_m=depth_min_m, depth_max_m=depth_max_m)
            sim, hole_mask, pollute_mask, edit_mask = simulate_horizontal_holes_with_pollution(
                rgb, dep,
                add_direction=add_dir,
                min_px=min_px, max_px=max_px,
                band_min_px=band_min_px, band_max_px=band_max_px, band_ratio=band_ratio,
                edge_k=edge_k, min_tau=min_tau,
                curv_k=curv_k, curv_min_tau=curv_min_tau, nms_rel=nms_rel,
                dilate_px=dilate_px, depth_greater_is_far=greater_is_far,
                smooth_ksize=smooth_ksize, smooth_iters=smooth_iters,
                pollute_alpha=pollute_alpha, pollute_blur_ksize=pollute_blur_ksize
            )

            cv2.imwrite(out_warp, sim)
            cv2.imwrite(out_hole, hole_mask)
            cv2.imwrite(out_poll, pollute_mask)  # 注意：这里保存的是环形污染带
            cv2.imwrite(out_edit, edit_mask)      # 注意：这里是 hole ∪ ring
            count += 1
        except Exception as e:
            print(f"[Warn] Failed on {clip_dir}/{f}: {e}")
    return count

def main():
    np.random.seed(42)
    ap = argparse.ArgumentParser(description="基于深度突变边界生成模拟水平空洞及（环形）污染带。图像仍做定向污染，mask改为空洞四周的一圈。")
    ap.add_argument("--root", required=True, default="/media/a1/16THDD/Zhan/depth_warp_vs/data/datasets/MannequinChallenge", help="数据根目录")
    ap.add_argument("--splits", default="train,validation,test", help="处理的子目录，逗号分隔")
    ap.add_argument("--min_px", type=int, default=2, help="远处最小空洞宽度（像素）")
    ap.add_argument("--max_px", type=int, default=24, help="近处最大空洞宽度（像素）")
    ap.add_argument("--band_min_px", type=int, default=1, help="污染带最小宽度")
    ap.add_argument("--band_max_px", type=int, default=8, help="污染带最大宽度")
    ap.add_argument("--band_ratio", type=float, default=0.5, help="污染带宽度相对空洞宽度比例")
    ap.add_argument("--edge_k", type=float, default=6.0, help="突变边界的一阶差分阈值系数")
    ap.add_argument("--min_tau", type=float, default=1e-3, help="一阶差分最小阈值")
    ap.add_argument("--curv_k", type=float, default=24.0, help="二阶差分阈值系数")
    ap.add_argument("--curv_min_tau", type=float, default=1e-6, help="二阶差分最小阈值")
    ap.add_argument("--nms_rel", type=float, default=0.1, help="梯度NMS相对裕量")
    ap.add_argument("--dilate_px", type=int, default=0, help="洞mask水平方向膨胀像素")
    ap.add_argument("--depth_mode", default="metric", choices=["auto", "metric", "normalized"], help="深度单位模式")
    ap.add_argument("--depth_color_decode", default="auto",
                    choices=["auto", "pca", "luma", "rgb24", "r", "g", "b", "vda"], help="彩色深度解码策略")
    ap.add_argument("--assign_strategy", default="alternate",
                    choices=["alternate", "first_half_left"], help="方向分配策略")
    ap.add_argument("--far_value", default="larger", choices=["larger", "smaller"], help="深度值大小与远近对应关系")
    ap.add_argument("--depth_min_m", type=float, default=2.0, help="深度下限（米）")
    ap.add_argument("--depth_max_m", type=float, default=10.0, help="深度上限（米）")
    ap.add_argument("--smooth_ksize", type=int, default=3, help="开闭平滑核大小")
    ap.add_argument("--smooth_iters", type=int, default=1, help="开闭平滑迭代次数")
    ap.add_argument("--pollute_alpha", type=float, default=0.8, help="污染带颜色混合系数（靠近近景）")
    ap.add_argument("--pollute_blur_ksize", type=int, default=0, help="污染带模糊核大小（0=不模糊）")

    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    total = 0
    for sp in splits:
        split_dir = os.path.join(args.root, sp)
        if not os.path.isdir(split_dir):
            print(f"[Skip] split not found: {split_dir}")
            continue
        items = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for d in items:
            clip_dir = os.path.join(split_dir, d)
            n = process_clip(
                clip_dir,
                min_px=args.min_px, max_px=args.max_px,
                band_min_px=args.band_min_px, band_max_px=args.band_max_px, band_ratio=args.band_ratio,
                edge_k=args.edge_k, min_tau=args.min_tau,
                curv_k=args.curv_k, curv_min_tau=args.curv_min_tau, nms_rel=args.nms_rel,
                dilate_px=args.dilate_px,
                depth_mode=args.depth_mode, depth_color_decode=args.depth_color_decode,
                assign_strategy=args.assign_strategy, far_value=args.far_value,
                depth_min_m=args.depth_min_m, depth_max_m=args.depth_max_m,
                smooth_ksize=args.smooth_ksize, smooth_iters=args.smooth_iters,
                pollute_alpha=args.pollute_alpha, pollute_blur_ksize=args.pollute_blur_ksize
            )
            if n > 0:
                print(f"[OK] {clip_dir}: processed {n} frames.")
            total += n
    print(f"All done. Total processed frames: {total}")

if __name__ == "__main__":
    main()
