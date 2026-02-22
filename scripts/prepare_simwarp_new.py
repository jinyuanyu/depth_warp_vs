# depth_warp_vs/scripts/prepare_simwarp_new.py
import os
import argparse
from typing import Tuple, List, Dict

import numpy as np
import cv2
from PIL import Image

import torch

# 使用你提供的新方法（请确保该模块按你给的路径已添加到工程中）
from depth_warp_vs.models.splatting.inverse_splat_holes import (
    generate_hole_mask,
    apply_holes_to_image,
    add_pollution_band,
    add_floating_pixels,
)


# ---------------- I/O helpers ----------------

def _read_rgb_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img  # BGR uint8


def _bgr_to_tchw01(bgr: np.ndarray) -> torch.Tensor:
    # 不改变颜色通道顺序（保持BGR），仅做到[0,1]并转为TCHW；新的方法对颜色本身不依赖RGB次序
    # 注意：此函数仍返回 CPU Tensor；在调用处统一 .to(device, non_blocking=True)
    bgr_f = (bgr.astype(np.float32) / 255.0)
    t = torch.from_numpy(bgr_f).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1x3xHxW
    return t


def _tchw01_to_bgr_u8(t: torch.Tensor) -> np.ndarray:
    arr = (t.clamp(0, 1) * 255.0).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return arr  # HxWx3 BGR uint8


# ---------------- Depth loader (new method) ----------------

def _load_depth(path: str,
                expected_hw: Tuple[int, int],
                depth_big_is: str,
                depth_scale: float,
                inv_depth_scale: float,
                inv_eps: float) -> torch.Tensor:
    """
    与 add_warp_like_holes 脚本一致的深度读取和尺度转换。
    - depth_big_is='far'  : 输入为度量深度（大=远）；.png使用depth_scale（meters = raw * depth_scale）
    - depth_big_is='near' : 输入为逆深度/视差（大=近）；meters = inv_depth_scale / max(inv_eps, raw)
    - 对 .npy 不使用 depth_scale 以保持行为一致
    """
    H, W = expected_hw
    is_npy = path.lower().endswith(".npy")

    if is_npy:
        d = np.load(path)
        if d.ndim == 2:
            pass
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        else:
            raise ValueError("Unsupported depth npy shape")
        if d.shape != (H, W):
            raise ValueError(f"Depth shape {d.shape} != image shape {(H, W)}")
        d = d.astype(np.float32)

        if depth_big_is == "far":
            d_m = d  # 假定已是米
        else:
            d_m = float(inv_depth_scale) / np.maximum(float(inv_eps), d)
    else:
        # 用PIL保证16位深度读取
        im = Image.open(path)
        im = im.convert("I;16") if im.mode != "I;16" else im
        d = np.array(im).astype(np.float32)
        if d.shape != (H, W):
            raise ValueError(f"Depth shape {d.shape} != image shape {(H, W)}")

        if depth_big_is == "far":
            d_m = d * float(depth_scale)
        else:
            d_m = float(inv_depth_scale) / np.maximum(float(inv_eps), d)

    Dt = torch.from_numpy(d_m).unsqueeze(0).unsqueeze(0)  # 1x1xHxW (meters)
    return Dt


# ---------------- Camera intrinsics/motion (new method) ----------------

def _build_intrinsics(fx: float, fy: float, cx: float, cy: float, batch: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    # 直接在目标 device 构造，避免后续一次拷贝
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1, 1)
    return K


def _rotz(a, device):
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def _roty(a, device):
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[ c, 0.0, s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0, c]], dtype=torch.float32, device=device)


def _rotx(a, device):
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[1.0, 0.0, 0.0],
                         [0.0,  c, -s],
                         [0.0,  s,  c]], dtype=torch.float32, device=device)


def _build_cam_motion(tx: float, ty: float, tz: float, rx_deg: float, ry_deg: float, rz_deg: float, device: torch.device) -> torch.Tensor:
    # 显式6DoF：Rz * Ry * Rx；直接在目标 device 上构造
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg]).tolist()
    R = _rotz(rz, device) @ _roty(ry, device) @ _rotx(rx, device)
    t = torch.tensor([tx, ty, tz], dtype=torch.float32, device=device).view(3, 1)
    dT = torch.eye(4, dtype=torch.float32, device=device)
    dT[:3, :3] = R
    dT[:3, 3:] = t
    return dT.unsqueeze(0)  # 1x4x4


def _build_lookat_rotation_from_campos(C: np.ndarray, target: np.ndarray, up_hint: np.ndarray = None) -> np.ndarray:
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd = (target.reshape(3,) - C.reshape(3,)).astype(np.float32)
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        fwd = fwd / n
    if abs(np.dot(fwd, up_hint) / (np.linalg.norm(up_hint) + 1e-8)) > 0.999:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(fwd[2]) < 0.999 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = np.cross(up_hint, fwd); nx = np.linalg.norm(x_axis)
    x_axis = x_axis / nx if nx > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = np.cross(fwd, x_axis); ny = np.linalg.norm(y_axis)
    y_axis = y_axis / ny if ny > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    R = np.stack([x_axis, y_axis, fwd], axis=1).astype(np.float32)
    return R


def _build_convergent_camera_motion(tx: float, focus_z: float, device: torch.device) -> torch.Tensor:
    # toe-in：沿X平移tx，同时旋转使Z=focus_z的点保持对准
    C = np.array([tx, 0.0, 0.0], dtype=np.float32)
    P_focus = np.array([0.0, 0.0, max(1e-6, float(focus_z))], dtype=np.float32)
    R_cam = _build_lookat_rotation_from_campos(C, P_focus, up_hint=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    dT = np.eye(4, dtype=np.float32)
    dT[:3, :3] = R_cam
    dT[:3, 3] = C
    dT_t = torch.from_numpy(dT).to(device=device).unsqueeze(0)  # 1x4x4
    return dT_t


# 简单结构元素缓存，避免每帧重复构造
_kernel_cache: Dict[Tuple[int, int], np.ndarray] = {}


def _get_ellipse_kernel(dpx: int) -> np.ndarray:
    key = (dpx, 1)
    k = _kernel_cache.get(key)
    if k is None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dpx + 1, 2 * dpx + 1))
        _kernel_cache[key] = k
    return k


# ---------------- Batch processing ----------------

def _choose_direction(index: int, total: int, strategy: str) -> str:
    if strategy == "alternate":
        return "left" if (index % 2 == 0) else "right"
    elif strategy == "first_half_left":
        left_cnt = (total + 1) // 2
        return "left" if index < left_cnt else "right"
    else:
        return "right"


def process_clip(
    clip_dir: str,
    depth_dir_name: str,
    assign_strategy: str,
    # New method parameters
    tx_mag: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
    focus_z: float or None,
    eps: float,
    max_band_px: float,
    float_prob: float,
    # Depth reading per new method
    depth_big_is: str,
    depth_scale: float,
    inv_depth_scale: float,
    inv_eps: float,
    # Pollute ring mask (dilation ring)
    pollute_dilate_px: int,
) -> int:
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
        if os.path.isfile(dep_path) or os.path.isfile(dep_path.replace(".png", ".npy")):
            # 允许两种后缀
            if not os.path.isfile(dep_path):
                dep_path = dep_path.replace(".png", ".npy")
            valid.append(f)
            depth_paths.append(dep_path)

    n_valid = len(valid)
    if n_valid == 0:
        return 0

    os.makedirs(os.path.join(clip_dir, "sim_warp"), exist_ok=True)
    os.makedirs(os.path.join(clip_dir, "hole_mask"), exist_ok=True)
    os.makedirs(os.path.join(clip_dir, "pollute_mask"), exist_ok=True)
    os.makedirs(os.path.join(clip_dir, "edit_mask"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    non_blocking = (device.type == "cuda")

    count = 0
    for idx, (f, dep_path) in enumerate(zip(valid, depth_paths)):
        stem = os.path.splitext(f)[0]
        idv = stem.replace("frame_", "")
        out_warp = os.path.join(clip_dir, "sim_warp",   f"warp_{idv}.png")
        out_hole = os.path.join(clip_dir, "hole_mask",  f"hole_{idv}.png")
        out_poll = os.path.join(clip_dir, "pollute_mask", f"pollute_{idv}.png")
        out_edit = os.path.join(clip_dir, "edit_mask",  f"edit_{idv}.png")

        add_dir = _choose_direction(idx, n_valid, assign_strategy)
        # 为每张图片在 [0.03, 0.06] 之间采样 tx 幅值，并由方向决定符号
        tx_mag_i = float(np.random.uniform(0.0001, 0.00025))
        tx_signed = float(tx_mag_i if add_dir == "right" else -tx_mag_i)

        try:
            # 读取图像/深度
            bgr = _read_rgb_bgr(os.path.join(clip_dir, f))
            H, W = bgr.shape[:2]
            It = _bgr_to_tchw01(bgr)
            if non_blocking:
                It = It.pin_memory()
            It = It.to(device, non_blocking=non_blocking)  # 1x3xHxW

            Dt = _load_depth(
                dep_path,
                (H, W),
                depth_big_is=str(depth_big_is),
                depth_scale=float(depth_scale),
                inv_depth_scale=float(inv_depth_scale),
                inv_eps=float(inv_eps),
            )
            if non_blocking:
                Dt = Dt.pin_memory()
            Dt = Dt.to(device, non_blocking=non_blocking)  # 1x1xHxW, meters

            # 内参（与单图脚本一致，直接在 device 上构造）
            fx = fy = float(max(H, W))
            cx = float((W - 1) * 0.5)
            cy = float((H - 1) * 0.5)
            K = _build_intrinsics(fx, fy, cx, cy, batch=1, device=device)

            # 相机运动：toe-in优先，否则6DoF（直接在 device 上构造）
            if focus_z is not None:
                dT = _build_convergent_camera_motion(tx_signed, float(focus_z), device=device)
            else:
                dT = _build_cam_motion(tx_signed, float(ty), float(tz), float(rx), float(ry), float(rz), device=device)

            # 生成空洞mask（新方法）
            hole_mask = generate_hole_mask(It, Dt, K, dT, hard_z_epsilon=float(eps))  # 1x1xHxW, {0,1}

            # 先在图像边上按方向添加污染带颜色（新方法，非mask；mask单独按膨胀生成）
            It_with_band = add_pollution_band(It, hole_mask, Dt, tx=float(tx_signed), max_band_px=float(max_band_px))

            # 空洞置黑
            Ih = apply_holes_to_image(It_with_band, hole_mask, fill_value=0.0).clamp_(0, 1)

            # 悬浮2x2像素（返回游离像素mask）
            Ih, float_mask = add_floating_pixels(Ih, hole_mask, It, tx=float(tx_signed), float_prob=float(float_prob))

            # 基于“剔除游离像素”的空洞mask（白=空洞；游离像素=黑）
            hole_mask_clean = hole_mask * (~float_mask).to(hole_mask.dtype)

            # 保存模拟图像
            sim_bgr = _tchw01_to_bgr_u8(Ih)
            cv2.imwrite(out_warp, sim_bgr)

            # 保存空洞mask（8位，已剔除游离像素）
            hole_u8 = (hole_mask_clean * 255.0).to(torch.uint8).squeeze(0).squeeze(0).cpu().numpy()
            cv2.imwrite(out_hole, hole_u8)

            # 生成污染带mask（按要求：在“修改后的空洞mask”基础上膨胀再减去空洞）
            dpx = max(1, int(pollute_dilate_px))
            k = _get_ellipse_kernel(dpx)
            dil = cv2.dilate(hole_u8, k, iterations=1)
            ring = cv2.subtract(dil, hole_u8)
            cv2.imwrite(out_poll, ring)

            # edit_mask = hole_clean ∪ ring
            edit = cv2.bitwise_or(hole_u8, ring)
            cv2.imwrite(out_edit, edit)

            count += 1
        except Exception as e:
            print(f"[Warn] Failed on {clip_dir}/{f}: {e}")
    return count


def main():
    np.random.seed(42)
    ap = argparse.ArgumentParser(description="批量用逆向前向投影生成warp空洞与污染带；污染带mask为(膨胀空洞 - 空洞)。")
    ap.add_argument("--root", required=True, default="/media/a1/16THDD/Zhan/depth_warp_vs/data/datasets/MannequinChallenge", help="数据根目录（与原方法一致）")
    ap.add_argument("--splits", default="train,validation,test", help="处理的子目录，逗号分隔（与原方法一致）")
    ap.add_argument("--depth_dir_name", default="depth", help="深度子目录名（与原方法一致）")
    ap.add_argument("--assign_strategy", default="alternate",
                    choices=["alternate", "first_half_left"], help="方向分配策略（与原方法一致）")

    # 新方法相关参数（与 add_warp_like_holes 一致/兼容）
    ap.add_argument("--tx_mag", type=float, default=0.05, help="X方向平移的幅值，方向由assign_strategy决定")
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--tz", type=float, default=0.0)
    ap.add_argument("--rx", type=float, default=0.0)
    ap.add_argument("--ry", type=float, default=0.0)
    ap.add_argument("--rz", type=float, default=0.0)
    ap.add_argument("--focus_z", type=float, default=None, help="toe-in 聚焦深度(米)，设置后覆盖ty,tz,rx,ry,rz")
    ap.add_argument("--eps", type=float, default=1e-3, help="z-buffer epsilon")
    ap.add_argument("--max_band_px", type=float, default=3.0, help="污染带最大宽度（像素，用于上色，不用于mask）")
    ap.add_argument("--float_prob", type=float, default=0.005, help="悬浮2x2像素概率")

    # 深度读取（新方法）
    ap.add_argument("--depth_big_is", choices=["far", "near"], default="far",
                    help="输入深度含义：'far'度量深度(大=远)；'near'逆深度(大=近)")
    ap.add_argument("--depth_scale", type=float, default=1.0,
                    help="对 .png 且 depth_big_is=far：meters = raw * depth_scale；.npy忽略")
    ap.add_argument("--inv_depth_scale", type=float, default=1.0,
                    help="depth_big_is=near：meters = inv_depth_scale / raw（.png/.npy均适用）")
    ap.add_argument("--inv_eps", type=float, default=1e-6, help="逆深度取倒时的最小值保护")

    # 污染带mask宽度（按‘膨胀-空洞’生成）
    ap.add_argument("--pollute_dilate_px", type=int, default=4, help="污染带mask的膨胀像素半径")

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
                clip_dir=clip_dir,
                depth_dir_name=args.depth_dir_name,
                assign_strategy=args.assign_strategy,
                tx_mag=args.tx_mag, ty=args.ty, tz=args.tz, rx=args.rx, ry=args.ry, rz=args.rz,
                focus_z=args.focus_z, eps=args.eps, max_band_px=args.max_band_px, float_prob=args.float_prob,
                depth_big_is=args.depth_big_is, depth_scale=args.depth_scale,
                inv_depth_scale=args.inv_depth_scale, inv_eps=args.inv_eps,
                pollute_dilate_px=args.pollute_dilate_px,
            )
            if n > 0:
                print(f"[OK] {clip_dir}: processed {n} frames.")
            total += n
    print(f"All done. Total processed frames: {total}")


if __name__ == "__main__":
    main()
# 示例：
# python depth_warp_vs/scripts/prepare_simwarp_new.py --root /path/to/datasets/MannequinChallenge --splits train,validation,test --depth_big_is near --tx_mag 0.0002 --focus_z 0.0074 --pollute_dilate_px 2
