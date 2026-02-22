# depth_warp_vs/test_warp_cli.py
import argparse
import os
import re
import cv2
import glob
import numpy as np
import torch

from depth_warp_vs.models.splatting.softmax_splat import softmax_splat

def extract_timestamp_from_path(path: str):
    base = os.path.basename(path)
    m = re.search(r"frame_(\d+)", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def read_meta_txt(meta_path: str, pose_convention: str = "w2c"):
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Meta txt not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Meta file has no frame lines: {meta_path}")
    frames = {}
    for line in lines[1:]:
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        if len(parts) < 19:
            continue
        try:
            ts = int(parts[0])
            fx, fy, cx, cy = map(float, parts[1:5])
            pose_vals = list(map(float, parts[7:7+12]))
        except Exception:
            continue
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        P = np.array(pose_vals, dtype=np.float32).reshape(3, 4)
        T = np.eye(4, dtype=np.float32)
        T[:3, :4] = P
        if pose_convention.lower() == "c2w":
            T = np.linalg.inv(T).astype(np.float32)
        frames[ts] = (K, T)
    if not frames:
        raise RuntimeError(f"No valid frame entries parsed in {meta_path}")
    return frames

def is_normalized_K(K: np.ndarray, thresh: float = 4.0) -> bool:
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    if any(np.isnan([fx, fy, cx, cy])):
        return False
    return (abs(fx) <= thresh) and (abs(fy) <= thresh) and (abs(cx) <= thresh) and (abs(cy) <= thresh)

def scale_intrinsics(K: np.ndarray, H: int, W: int):
    Kp = K.copy().astype(np.float32)
    Kp[0, 0] *= float(W)
    Kp[0, 2] *= float(W)
    Kp[1, 1] *= float(H)
    Kp[1, 2] *= float(H)
    Kp[2, 2] = 1.0
    return Kp

def _resize_intrinsics(K: np.ndarray, src_hw, dst_hw):
    H0, W0 = src_hw
    H1, W1 = dst_hw
    sx = float(W1) / max(1.0, float(W0))
    sy = float(H1) / max(1.0, float(H0))
    K = K.copy().astype(np.float32)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K

def unify_intrinsics(K_meta: np.ndarray, native_hw, final_hw):
    if is_normalized_K(K_meta):
        return scale_intrinsics(K_meta, final_hw[0], final_hw[1])
    else:
        if (native_hw is None) or (tuple(native_hw) == tuple(final_hw)):
            return K_meta.copy().astype(np.float32)
        else:
            return _resize_intrinsics(K_meta, native_hw, final_hw)

def load_rgb(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    H0, W0 = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return ten, (H0, W0)

def _color_depth_to_scalar_bgr(dep_bgr: np.ndarray, method: str = "auto") -> np.ndarray:
    dep_bgr = dep_bgr.astype(np.float32)
    h, w, _ = dep_bgr.shape
    if method in ["r", "g", "b", "luma", "rgb24"]:
        pass
    elif method == "auto":
        diff_rg = np.mean(np.abs(dep_bgr[..., 2] - dep_bgr[..., 1]))
        diff_gb = np.mean(np.abs(dep_bgr[..., 1] - dep_bgr[..., 0]))
        diff_rb = np.mean(np.abs(dep_bgr[..., 2] - dep_bgr[..., 0]))
        method = "luma" if (diff_rg + diff_gb + diff_rb) / 3.0 < 2.0 else "pca"

    if method == "r":
        v = dep_bgr[..., 2] / 255.0
    elif method == "g":
        v = dep_bgr[..., 1] / 255.0
    elif method == "b":
        v = dep_bgr[..., 0] / 255.0
    elif method == "luma":
        v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    elif method == "rgb24":
        R = dep_bgr[..., 2]; G = dep_bgr[..., 1]; B = dep_bgr[..., 0]
        v = (R * 65536.0 + G * 256.0 + B) / (255.0 * 65536.0 + 255.0 * 256.0 + 255.0)
    else:
        step = max(1, int(np.ceil(np.sqrt((h * w) / 10000.0))))
        sample = dep_bgr[::step, ::step, :].reshape(-1, 3)
        mean = sample.mean(axis=0, keepdims=True)
        X = sample - mean
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, np.argmax(eigvals)]
        proj = (dep_bgr.reshape(-1, 3) - mean) @ pc1
        proj = proj.reshape(h, w).astype(np.float32)
        pmin = float(np.min(proj)); pmax = float(np.max(proj))
        if pmax - pmin < 1e-6:
            v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            v = (proj - pmin) / (pmax - pmin)
    return np.clip(v, 0.0, 1.0).astype(np.float32)

def decode_depth_array(dep_arr, size=None, mode="auto", scale=10.0, color_decode="auto", far_value="larger"):
    dep = dep_arr
    if dep is None:
        raise ValueError("Empty depth array")
    H0, W0 = dep.shape[:2]
    if dep.ndim == 3:
        v01 = _color_depth_to_scalar_bgr(dep, method=color_decode)
        if size is not None:
            v01 = cv2.resize(v01, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        depth = v01.copy()
        if mode == "metric":
            depth = depth * float(scale)
            depth = np.clip(depth, 1e-6, 1e6)
        elif mode == "normalized":
            depth = np.clip(depth, 1e-6, 1.0) * float(scale)
        else:
            depth = np.clip(depth, 1e-6, 1.0) * float(scale)
    else:
        dep = dep.astype(np.float32)
        if size is not None:
            dep = cv2.resize(dep, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        if mode == "metric":
            if dep.max() > 255.0:
                dep = dep / 1000.0
            depth = np.clip(dep, 1e-3, 1e6)
        elif mode == "normalized":
            if dep.max() > 1.0:
                dep = dep / 255.0
            depth = np.clip(dep, 1e-4, 1.0) * float(scale)
        else:
            if dep.max() > 255.0:
                dep = dep / 1000.0
                depth = np.clip(dep, 1e-3, 1e6)
            else:
                if dep.max() > 1.0:
                    dep = dep / 255.0
                depth = np.clip(dep, 1e-4, 1.0) * float(scale)
    if far_value == "smaller":
        dmin = float(np.min(depth)); dmax = float(np.max(depth))
        depth = (dmin + dmax) - depth
        depth = np.clip(depth, 1e-6, 1e9)
    ten = torch.from_numpy(depth.astype(np.float32)).float().unsqueeze(0).unsqueeze(0)
    return ten, (H0, W0)

def load_depth(path, size=None, mode="auto", scale=10.0, color_decode="auto", far_value="larger"):
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(path)
    return decode_depth_array(dep, size=size, mode=mode, scale=scale, color_decode=color_decode, far_value=far_value)

def euler_xyz_to_matrix(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    return R.astype(np.float32)

def build_camera_motion_from_se3(tx, ty, tz, rx=0.0, ry=0.0, rz=0.0):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_xyz_to_matrix(rx, ry, rz)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T

def compute_tx_from_disp(fx_px: float, depth_ref: float, disp_px: float) -> float:
    return float(disp_px) * float(depth_ref) / max(1e-6, float(fx_px))

def is_video_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv"]

def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)]

class StreamReader:
    def __init__(self, path: str, is_depth: bool = False):
        self.path = path
        self.is_depth = is_depth
        self.cap = None
        self.files = None
        self.idx = 0
        self.fps = None

        if os.path.isdir(path):
            exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(path, e)))
            files = sorted(files, key=natural_key)
            if not files:
                raise FileNotFoundError(f"No image files found in directory: {path}")
            self.files = files
            self.frame_count = len(self.files)
            self.fps = None
        elif os.path.isfile(path) and is_video_file(path):
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if self.fps <= 0.0:
                self.fps = None
        else:
            raise FileNotFoundError(f"Not a video file or directory: {path}")

    def read(self):
        if self.files is not None:
            if self.idx >= len(self.files):
                return False, None
            fp = self.files[self.idx]
            self.idx += 1
            flag = cv2.IMREAD_UNCHANGED if self.is_depth else cv2.IMREAD_COLOR
            img = cv2.imread(fp, flag)
            if img is None:
                return False, None
            return True, img
        else:
            ok, frame = self.cap.read()
            if not ok:
                return False, None
            return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()

def parse_manual_K(s: str):
    vals = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    if len(vals) != 4:
        raise ValueError("manual_K must be 'fx,fy,cx,cy'")
    K = np.array([[vals[0], 0.0, vals[2]],
                  [0.0, vals[1], vals[3]],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def build_offsets(tx_max: float, num_per_side: int, spacing: str):
    N = int(max(0, num_per_side))
    if N == 0:
        return [0.0]
    k = np.arange(-N, N + 1, dtype=np.float32)
    if spacing == "linear":
        s = k / max(1, N)
    else:
        s = np.sin(0.5 * np.pi * (k / max(1, N)))
    return (s * tx_max).tolist()

def _visualize_depth_for_click(depth_np: np.ndarray) -> np.ndarray:
    d = depth_np.copy().astype(np.float32)
    d = d[np.isfinite(d)]
    if d.size == 0:
        dmin, dmax = 0.0, 1.0
    else:
        dmin, dmax = float(np.percentile(d, 1.0)), float(np.percentile(d, 99.0))
        if not np.isfinite(dmin): dmin = float(np.min(depth_np))
        if not np.isfinite(dmax): dmax = float(np.max(depth_np))
        if dmax <= dmin:
            dmax = dmin + 1e-6
    v = np.clip((depth_np - dmin) / (dmax - dmin), 0.0, 1.0)
    v8 = (v * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(v8, cv2.COLORMAP_TURBO)
    return cm

def pick_focus_depth_gui(depth_np: np.ndarray, win_name: str = "深度图 - 鼠标左键点击选择汇聚深度，ESC/Enter确认") -> float:
    H, W = depth_np.shape
    show = _visualize_depth_for_click(depth_np)
    clicked = {"z": None}
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            z = float(depth_np[y, x])
            if not np.isfinite(z) or z <= 1e-6:
                return
            clicked["z"] = z
            disp = show.copy()
            cv2.circle(disp, (x, y), 5, (0, 0, 255), 2)
            txt = f"z={z:.4f}  (x={x}, y={y})  Enter/ESC to confirm"
            cv2.putText(disp, txt, (10, max(25, H-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(win_name, disp)
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, show)
    cv2.setMouseCallback(win_name, cb)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, 13):
            break
    cv2.destroyWindow(win_name)
    if clicked["z"] is not None:
        return clicked["z"]
    valid = depth_np[np.isfinite(depth_np) & (depth_np > 1e-6)]
    if valid.size == 0:
        return float(max(1e-3, np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0).mean()))
    return float(np.median(valid))

def build_lookat_rotation_from_campos(C: np.ndarray, target: np.ndarray, up_hint: np.ndarray = None) -> np.ndarray:
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
    x_axis = np.cross(up_hint, fwd)
    nx = np.linalg.norm(x_axis)
    if nx < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_axis = x_axis / nx
    y_axis = np.cross(fwd, x_axis)
    ny = np.linalg.norm(y_axis)
    if ny < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        y_axis = y_axis / ny
    R = np.stack([x_axis, y_axis, fwd], axis=1).astype(np.float32)
    return R

def build_convergent_camera_motion(tx: float, focus_z: float) -> np.ndarray:
    C = np.array([tx, 0.0, 0.0], dtype=np.float32)
    P_focus = np.array([0.0, 0.0, max(1e-6, float(focus_z))], dtype=np.float32)
    R_cam = build_lookat_rotation_from_campos(C, P_focus, up_hint=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    dT = np.eye(4, dtype=np.float32)
    dT[:3, :3] = R_cam
    dT[:3, 3] = C
    return dT

def rescale_depth_tensor_linear(Ds: torch.Tensor, rng=(2.0, 10.0), percentiles=(1.0, 99.0)) -> torch.Tensor:
    d = Ds.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    v = d[np.isfinite(d) & (d > 1e-8)]
    if v.size < 16:
        return Ds
    plo, phi = float(percentiles[0]), float(percentiles[1])
    z0 = float(np.percentile(v, plo))
    z1 = float(np.percentile(v, phi))
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        return Ds
    a, b = float(rng[0]), float(rng[1])
    d2 = a + (d - z0) * (b - a) / (z1 - z0)
    d2 = np.clip(d2, min(a,b), max(a,b)).astype(np.float32)
    return torch.from_numpy(d2).view(1,1,*d.shape).to(Ds.dtype)

def _compute_hole_mask_from_visibility(V_tensor: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    Vn = V_tensor.detach().cpu().squeeze(0).squeeze(0).numpy().astype(np.float32)
    hole = (~np.isfinite(Vn)) | (Vn <= eps)
    hole_img = (np.stack([hole, hole, hole], axis=-1).astype(np.uint8) * 255)
    return hole_img

def run_single_image(args):
    Hs, Ws = [int(x) for x in args.img_size.replace(",", " ").split()]
    resize_to = None if (Hs == 0 or Ws == 0) else (Hs, Ws)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Is, (H0_s, W0_s) = load_rgb(args.image, size=resize_to)
    Ds, _ = load_depth(args.depth, size=resize_to, mode=args.depth_mode, scale=args.depth_scale,
                       color_decode=args.depth_color_decode, far_value=args.far_value)
    if args.rescale_depth == "linear":
        rmin, rmax = [float(x) for x in args.rescale_range.replace(",", " ").split()]
        plo, phi = [float(x) for x in args.rescale_percentiles.replace(",", " ").split()]
        Ds = rescale_depth_tensor_linear(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
    Is = Is.to(device)
    Ds = Ds.to(device)
    B, _, H, W = Is.shape

    if args.manual_K is not None:
        K_manual = parse_manual_K(args.manual_K)
        if args.K_units == "normalized" or (args.K_units == "auto" and is_normalized_K(K_manual)):
            Ks_np_final = scale_intrinsics(K_manual, H, W)
        else:
            Ks_np_final = _resize_intrinsics(K_manual, (H0_s, W0_s), (H, W))
    else:
        if args.meta is None:
            raise ValueError("Single image mode: either --manual_K or --meta must be provided.")
        meta = read_meta_txt(args.meta, pose_convention=args.pose_convention)
        src_ts = extract_timestamp_from_path(args.image)
        if src_ts is None or src_ts not in meta:
            raise KeyError(f"Source timestamp not found in meta or filename has no 'frame_<timestamp>': {args.image}")
        Ks_meta, _ = meta[src_ts]
        Ks_np_final = unify_intrinsics(Ks_meta, native_hw=(H0_s, W0_s), final_hw=(H, W))

    Ks = torch.from_numpy(Ks_np_final).float().unsqueeze(0).to(device)
    Kt = Ks.clone()

    fx_px = float(Ks_np_final[0, 0])
    Dnp = Ds.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    valid = Dnp[np.isfinite(Dnp) & (Dnp > 1e-6)]
    if valid.size < 16:
        z_ref = float(np.maximum(Dnp.mean(), 1e-3))
    else:
        q = float(np.clip(args.disp_ref_percentile, 0.0, 1.0))
        z_ref = float(np.quantile(valid, q))
        z_ref = max(z_ref, 1e-6)
    if args.tx_max > 0.0:
        tx_max = float(args.tx_max)
    else:
        tx_max = compute_tx_from_disp(fx_px, z_ref, args.max_disp_px)
    offsets = build_offsets(tx_max, args.num_per_side, args.spacing)

    if args.focus_depth > 0.0:
        z_focus = float(args.focus_depth)
    else:
        try:
            Ds_gui, _ = load_depth(args.depth, size=None, mode=args.depth_mode, scale=args.depth_scale,
                                   color_decode=args.depth_color_decode, far_value=args.far_value)
            if args.rescale_depth == "linear":
                Ds_gui = rescale_depth_tensor_linear(Ds_gui, rng=(rmin, rmax), percentiles=(plo, phi))
            Dnp_gui = Ds_gui.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
            z_focus = pick_focus_depth_gui(Dnp_gui, "Depth - Left click to pick focus depth; Enter/ESC to confirm")
        except Exception:
            z_focus = float(np.median(valid) if valid.size > 0 else z_ref)
    z_focus = max(z_focus, 1e-6)

    out_dir = args.out_dir or os.path.dirname(args.image)
    os.makedirs(out_dir, exist_ok=True)

    # 中心索引（tx==0）直接保存原图与纯黑hole，避免softmax splat引入微小空洞
    center_idx = len(offsets) // 2 if (len(offsets) % 2 == 1) else None
    orig_np_rgb = (Is.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    black_hole = np.zeros((H, W, 3), dtype=np.uint8)
    if args.save_vis:
        full_vis = np.full((H, W, 3), 255, dtype=np.uint8)

    with torch.no_grad():
        for idx, tx in enumerate(offsets):
            if (center_idx is not None) and (idx == center_idx):
                # 直接输出原图和纯黑hole
                out_warp = os.path.join(out_dir, f"warp_s{idx:02d}_tx{tx:+.5f}.png")
                cv2.imwrite(out_warp, cv2.cvtColor(orig_np_rgb, cv2.COLOR_RGB2BGR))
                if args.save_hole:
                    out_hole = os.path.join(out_dir, f"hole_s{idx:02d}_tx{tx:+.5f}.png")
                    cv2.imwrite(out_hole, black_hole)
                if args.save_vis:
                    out_vis = os.path.join(out_dir, f"vis_s{idx:02d}_tx{tx:+.5f}.png")
                    cv2.imwrite(out_vis, full_vis)
                continue

            dT_np = build_convergent_camera_motion(tx=tx, focus_z=z_focus)
            dT = torch.from_numpy(dT_np).float().unsqueeze(0).to(device)
            Iw, V = softmax_splat(
                Is, Ds, Ks, Kt, dT,
                temperature=args.temperature,
                normalize=True,
                occlusion=args.occlusion_mode,
                hard_z_epsilon=args.hard_z_epsilon
            )
            Iw_np = (Iw.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            out_warp = os.path.join(out_dir, f"warp_s{idx:02d}_tx{tx:+.5f}.png")
            cv2.imwrite(out_warp, cv2.cvtColor(Iw_np, cv2.COLOR_RGB2BGR))

            if args.save_vis:
                Vn = V.cpu().squeeze(0).squeeze(0).numpy()
                Vn = Vn - Vn.min()
                if Vn.max() > 1e-8:
                    Vn = Vn / Vn.max()
                Vc = (np.stack([Vn, Vn, Vn], axis=-1) * 255).astype(np.uint8)
                out_vis = os.path.join(out_dir, f"vis_s{idx:02d}_tx{tx:+.5f}.png")
                cv2.imwrite(out_vis, Vc)

            if args.save_hole:
                hole_img = _compute_hole_mask_from_visibility(V, eps=1e-6)
                out_hole = os.path.join(out_dir, f"hole_s{idx:02d}_tx{tx:+.5f}.png")
                cv2.imwrite(out_hole, hole_img)

def run_video(args):
    if args.video is None or args.depth_video is None:
        raise ValueError("Video mode requires --video and --depth_video.")
    if args.manual_K is None:
        raise ValueError("Video mode requires --manual_K。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb_reader = StreamReader(args.video, is_depth=False)
    depth_reader = StreamReader(args.depth_video, is_depth=True)
    fps = args.fps if args.fps > 0 else (rgb_reader.fps or 30.0)

    ok_rgb, rgb0 = rgb_reader.read()
    ok_dep, dep0 = depth_reader.read()
    if not ok_rgb or not ok_dep:
        rgb_reader.release(); depth_reader.release()
        raise RuntimeError("Failed to read first frame from video/depth_video.")

    H0_s, W0_s = rgb0.shape[:2]
    Hs, Ws = [int(x) for x in args.img_size.replace(",", " ").split()]
    resize_to = None if (Hs == 0 or Ws == 0) else (Hs, Ws)
    H_final, W_final = (H0_s, W0_s) if resize_to is None else (resize_to[0], resize_to[1])

    K_manual = parse_manual_K(args.manual_K)
    if args.K_units == "normalized" or (args.K_units == "auto" and is_normalized_K(K_manual)):
        Ks_np_final = scale_intrinsics(K_manual, H_final, W_final)
    else:
        Ks_np_final = _resize_intrinsics(K_manual, (H0_s, W0_s), (H_final, W_final))
    Ks = torch.from_numpy(Ks_np_final).float().unsqueeze(0).to(device)
    Kt = Ks.clone()

    Ds0, _ = decode_depth_array(dep0, size=resize_to, mode=args.depth_mode, scale=args.depth_scale,
                                color_decode=args.depth_color_decode, far_value=args.far_value)
    if args.rescale_depth == "linear":
        rmin, rmax = [float(x) for x in args.rescale_range.replace(",", " ").split()]
        plo, phi = [float(x) for x in args.rescale_percentiles.replace(",", " ").split()]
        Ds0 = rescale_depth_tensor_linear(Ds0, rng=(rmin, rmax), percentiles=(plo, phi))
    fx_px = float(Ks_np_final[0, 0])
    Dnp0 = Ds0.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    valid = Dnp0[np.isfinite(Dnp0) & (Dnp0 > 1e-6)]
    if valid.size < 16:
        z_ref = float(np.maximum(Dnp0.mean(), 1e-3))
    else:
        q = float(np.clip(args.disp_ref_percentile, 0.0, 1.0))
        z_ref = float(np.quantile(valid, q))
        z_ref = max(z_ref, 1e-6)
    if args.tx_max > 0.0:
        tx_max = float(args.tx_max)
    else:
        tx_max = compute_tx_from_disp(fx_px, z_ref, args.max_disp_px)
    offsets = build_offsets(tx_max, args.num_per_side, args.spacing)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.video), "sweep_video")
    os.makedirs(out_dir, exist_ok=True)

    writers_warp = []
    writers_vis = []
    writers_hole = []

    for idx, tx in enumerate(offsets):
        fn_warp = os.path.join(out_dir, f"warp_s{idx:02d}_tx{tx:+.5f}.mp4")
        w_warp = cv2.VideoWriter(fn_warp, fourcc, fps, (W_final, H_final), isColor=True)
        if not w_warp.isOpened():
            raise RuntimeError(f"Cannot open writer: {fn_warp}")
        writers_warp.append(w_warp)

        if args.save_vis:
            fn_vis = os.path.join(out_dir, f"vis_s{idx:02d}_tx{tx:+.5f}.mp4")
            w_vis = cv2.VideoWriter(fn_vis, fourcc, fps, (W_final, H_final), isColor=True)
            if not w_vis.isOpened():
                raise RuntimeError(f"Cannot open writer: {fn_vis}")
            writers_vis.append(w_vis)

        if args.save_hole:
            fn_hole = os.path.join(out_dir, f"hole_s{idx:02d}_tx{tx:+.5f}.mp4")
            w_hole = cv2.VideoWriter(fn_hole, fourcc, fps, (W_final, H_final), isColor=True)
            if not w_hole.isOpened():
                raise RuntimeError(f"Cannot open writer: {fn_hole}")
            writers_hole.append(w_hole)

    def rgb_to_tensor(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if resize_to is not None:
            img = cv2.resize(img, (W_final, H_final), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return ten

    center_idx = len(offsets) // 2 if (len(offsets) % 2 == 1) else None
    black_hole = np.zeros((H_final, W_final, 3), dtype=np.uint8)
    if args.save_vis:
        full_vis = np.full((H_final, W_final, 3), 255, dtype=np.uint8)

    def process_one(rgb_bgr, dep_img):
        # 准备原图（按最终尺寸）
        if resize_to is not None:
            orig_bgr_resized = cv2.resize(rgb_bgr, (W_final, H_final), interpolation=cv2.INTER_AREA)
        else:
            orig_bgr_resized = rgb_bgr.copy()

        Is = rgb_to_tensor(rgb_bgr).to(device)
        Ds, _ = decode_depth_array(dep_img, size=resize_to, mode=args.depth_mode, scale=args.depth_scale,
                                   color_decode=args.depth_color_decode, far_value=args.far_value)
        if args.rescale_depth == "linear":
            Ds = rescale_depth_tensor_linear(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
        Ds = Ds.to(device)

        with torch.no_grad():
            for idx, tx in enumerate(offsets):
                if (center_idx is not None) and (idx == center_idx):
                    # 中心视角：直接输出原图和纯黑hole，避免微小空洞
                    writers_warp[idx].write(orig_bgr_resized)
                    if args.save_vis:
                        writers_vis[idx].write(full_vis)
                    if args.save_hole:
                        writers_hole[idx].write(black_hole)
                    continue

                dT_np = build_convergent_camera_motion(tx=tx, focus_z=z_focus)
                dT = torch.from_numpy(dT_np).float().unsqueeze(0).to(device)
                Iw, V = softmax_splat(
                    Is, Ds, Ks, Kt, dT,
                    temperature=args.temperature,
                    normalize=True,
                    occlusion=args.occlusion_mode,
                    hard_z_epsilon=args.hard_z_epsilon
                )
                Iw_np = (Iw.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                writers_warp[idx].write(cv2.cvtColor(Iw_np, cv2.COLOR_RGB2BGR))

                if args.save_vis:
                    Vn = V.cpu().squeeze(0).squeeze(0).numpy()
                    Vn = Vn - Vn.min()
                    if Vn.max() > 1e-8:
                        Vn = Vn / Vn.max()
                    Vc = (np.stack([Vn, Vn, Vn], axis=-1) * 255).astype(np.uint8)
                    writers_vis[idx].write(Vc)

                if args.save_hole:
                    hole_img = _compute_hole_mask_from_visibility(V, eps=1e-6)
                    writers_hole[idx].write(hole_img)

    if args.focus_depth > 0.0:
        z_focus = float(args.focus_depth)
    else:
        try:
            Ds0_gui, _ = decode_depth_array(dep0, size=None, mode=args.depth_mode, scale=args.depth_scale,
                                            color_decode=args.depth_color_decode, far_value=args.far_value)
            if args.rescale_depth == "linear":
                Ds0_gui = rescale_depth_tensor_linear(Ds0_gui, rng=(rmin, rmax), percentiles=(plo, phi))
            Dnp0_gui = Ds0_gui.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
            z_focus = pick_focus_depth_gui(Dnp0_gui, "Depth - Left click to pick focus depth; Enter/ESC to confirm")
        except Exception:
            z_focus = float(np.median(valid) if valid.size > 0 else z_ref)
    z_focus = max(z_focus, 1e-6)

    process_one(rgb0, dep0)
    while True:
        ok_rgb, rgb = rgb_reader.read()
        ok_dep, dep = depth_reader.read()
        if not ok_rgb or not ok_dep:
            break
        process_one(rgb, dep)

    rgb_reader.release()
    depth_reader.release()
    for w in writers_warp:
        w.release()
    for w in writers_vis:
        w.release()
    for w in writers_hole:
        w.release()
    print(f"Done. Saved {len(offsets)} warped videos to {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="图像/视频水平相机平移视角合成（支持彩色深度、深度视频/序列、手动相机参数）")
    ap.add_argument("--image", help="源图像，文件名最好包含 frame_<timestamp>")
    ap.add_argument("--depth", help="对应的深度图（可为彩色或灰度；metric uint16 或 0~255归一化均可）")
    ap.add_argument("--video", help="源视频（文件路径）")
    ap.add_argument("--depth_video", help="深度视频（视频文件或深度图序列目录）")

    ap.add_argument("--meta", help="对应clip的 .txt 参数文件（含所有帧的内参与位姿）")
    ap.add_argument("--pose_convention", default="w2c", choices=["w2c", "c2w"])

    ap.add_argument("--manual_K", default=None, help="手动相机内参：'fx,fy,cx,cy'")
    ap.add_argument("--K_units", default="auto", choices=["auto", "pixel", "normalized"])
    ap.add_argument("--img_size", default="0,0", help="H,W (0,0 表示保持原尺寸)")

    ap.add_argument("--depth_mode", default="metric", choices=["auto", "metric", "normalized"])
    ap.add_argument("--depth_scale", type=float, default=10.0)
    ap.add_argument("--depth_color_decode", default="auto", choices=["auto", "pca", "luma", "rgb24", "r", "g", "b"])
    ap.add_argument("--far_value", default="larger", choices=["larger", "smaller"])

    ap.add_argument("--rescale_depth", default="linear", choices=["none","linear"])
    ap.add_argument("--rescale_range", default="2,10")
    ap.add_argument("--rescale_percentiles", default="1,99")

    ap.add_argument("--tx_max", type=float, default=0.0)
    ap.add_argument("--max_disp_px", type=float, default=48.0)
    ap.add_argument("--disp_ref_percentile", type=float, default=0.5)
    ap.add_argument("--num_per_side", type=int, default=4)
    ap.add_argument("--spacing", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--save_center", action="store_true")  # 保留该参数以兼容，但当前逻辑在中心视角始终保存原图

    ap.add_argument("--temperature", type=float, default=30.0)
    ap.add_argument("--occlusion_mode", default="hard", choices=["soft", "hard"])
    ap.add_argument("--hard_z_epsilon", type=float, default=1.0e-3)

    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--save_hole", action="store_true")
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--focus_depth", type=float, default=0.0)

    args = ap.parse_args()
    torch.set_grad_enabled(False)

    is_image_mode = (args.image is not None) and (args.depth is not None) and (args.video is None) and (args.depth_video is None)
    is_video_mode = (args.video is not None) and (args.depth_video is not None) and (args.image is None) and (args.depth is None)

    if not (is_image_mode or is_video_mode):
        raise ValueError("选择一种模式：图像(--image --depth) 或 视频(--video --depth_video)。")

    if is_image_mode:
        run_single_image(args)
    else:
        run_video(args)

if __name__ == "__main__":
    main()
