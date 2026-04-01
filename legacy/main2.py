# depth_warp_vs/main.py
import os, re, cv2, glob, time, argparse, math, subprocess, shutil, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F


try:
    from depth_warp_vs.models.splatting.softmax_splat import softmax_splat
    from depth_warp_vs.models.refiner import build_refiner
except Exception:
    from models.splatting.softmax_splat import softmax_splat
    from depth_warp_vs.models.refiner import build_refiner

import threading
from queue import Queue

# NEW: acceleration helpers
try:
    import torch_tensorrt
    _HAS_TRT = True
except Exception:
    _HAS_TRT = False

_IDX_BGR_CACHE = {}
def _get_idx_bgr(device):
    key = str(device)
    t = _IDX_BGR_CACHE.get(key, None)
    if t is None:
        t = torch.tensor([2, 1, 0], device=device)
        _IDX_BGR_CACHE[key] = t
    return t

class CudaGraphWrapper:
    """
    用于捕获固定shape+dtype的前向图（Refiner）。要求：
    - shape固定（B,4,H,W）、dtype固定、device固定
    - 建议：channels_last 保持一致
    """
    def __init__(self, model: torch.nn.Module, example_x: torch.Tensor, amp_enabled: bool, channels_last: bool):
        assert example_x.is_cuda, "CudaGraph 仅支持CUDA张量"
        self.model = model
        self.amp_enabled = bool(amp_enabled)
        self.channels_last = bool(channels_last)
        # 静态输入/输出缓冲
        self.static_x = example_x.clone()
        if self.channels_last:
            self.static_x = self.static_x.contiguous(memory_format=torch.channels_last)
        self.static_y = None
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.autocast(device_type="cuda", enabled=self.amp_enabled):
                self.static_y = self.model(self.static_x)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 要求 x 与 static_x 同shape/dtype/device/memory_format
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_y


VALID_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VALID_VID_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv")

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def is_video_file(path: str) -> bool:
    return os.path.isfile(path) and (os.path.splitext(path)[1].lower() in VALID_VID_EXTS)

def parse_manual_K(s: str):
    vals = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    if len(vals) != 4:
        raise ValueError("manual_K 必须为 'fx,fy,cx,cy'")
    K = np.array([[vals[0], 0.0, vals[2]],
                  [0.0, vals[1], vals[3]],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def is_normalized_K(K: np.ndarray, thresh: float = 4.0) -> bool:
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    if any([np.isnan(fx), np.isnan(fy), np.isnan(cx), np.isnan(cy)]):
        return False
    return (abs(fx) <= thresh) and (abs(fy) <= thresh) and (abs(cx) <= thresh) and (abs(cy) <= thresh)

def scale_intrinsics(K: np.ndarray, H: int, W: int):
    Kp = K.copy().astype(np.float32)
    Kp[0,0] *= float(W); Kp[0,2] *= float(W)
    Kp[1,1] *= float(H); Kp[1,2] *= float(H)
    Kp[2,2] = 1.0
    return Kp

def resize_intrinsics(K: np.ndarray, src_hw, dst_hw):
    H0, W0 = src_hw; H1, W1 = dst_hw
    sx = float(W1) / max(1.0, float(W0))
    sy = float(H1) / max(1.0, float(H0))
    Kp = K.copy().astype(np.float32)
    Kp[0,0] *= sx; Kp[1,1] *= sy; Kp[0,2] *= sx; Kp[1,2] *= sy; Kp[2,2] = 1.0
    return Kp

def default_K(H: int, W: int):
    fx = fy = 0.9 * W
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

def load_rgb(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    H0, W0 = img.shape[:2]
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0)/255.0
    return ten, (H0, W0)

def color_depth_to_scalar_bgr(dep_bgr: np.ndarray, method: str = "auto") -> np.ndarray:
    dep_bgr = dep_bgr.astype(np.float32)
    h, w, _ = dep_bgr.shape
    if method in ["r", "g", "b", "luma", "rgb24"]:
        pass
    elif method == "auto":
        diff_rg = np.mean(np.abs(dep_bgr[..., 2] - dep_bgr[..., 1]))
        diff_gb = np.mean(np.abs(dep_bgr[..., 1] - dep_bgr[..., 0]))
        diff_rb = np.mean(np.abs(dep_bgr[..., 2] - dep_bgr[..., 0]))
        method = "luma" if (diff_rg + diff_gb + diff_rb)/3.0 < 2.0 else "pca"
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
        v = (R*65536.0 + G*256.0 + B) / (255.0*65536.0 + 255.0*256.0 + 255.0)
    else:
        step = max(1, int(np.ceil(np.sqrt((h*w)/10000.0))))
        sample = dep_bgr[::step, ::step, :].reshape(-1, 3)
        mean = sample.mean(axis=0, keepdims=True)
        X = sample - mean
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, np.argmax(eigvals)]
        proj = (dep_bgr.reshape(-1,3) - mean) @ pc1
        proj = proj.reshape(h, w).astype(np.float32)
        pmin, pmax = float(np.min(proj)), float(np.max(proj))
        if pmax - pmin < 1e-6:
            v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY)/255.0
        else:
            v = (proj - pmin) / (pmax - pmin)
    return np.clip(v, 0.0, 1.0).astype(np.float32)

def decode_depth_array(dep, size=None, mode="auto", scale=10.0, color_decode="auto", far_value="larger"):
    if dep is None: raise ValueError("Empty depth array")
    H0, W0 = dep.shape[:2]
    if dep.ndim == 3:
        v01 = color_depth_to_scalar_bgr(dep, method=color_decode)
        if size is not None:
            v01 = cv2.resize(v01, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        if mode == "metric":
            depth = np.clip(v01*float(scale), 1e-6, 1e6)
        elif mode == "normalized":
            depth = np.clip(v01, 1e-6, 1.0)*float(scale)
        else:
            depth = np.clip(v01, 1e-6, 1.0)*float(scale)
    else:
        dep = dep.astype(np.float32)
        if size is not None:
            dep = cv2.resize(dep, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        if mode == "metric":
            if dep.max() > 255.0: dep = dep / 1000.0
            depth = np.clip(dep, 1e-3, 1e6)
        elif mode == "normalized":
            if dep.max() > 1.0: dep = dep / 255.0
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
        dmin, dmax = float(depth.min()), float(depth.max())
        depth = (dmin + dmax) - depth
        depth = np.clip(depth, 1e-6, 1e9)
    ten = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
    return ten, (H0, W0)

def load_depth(path, size=None, mode="auto", scale=10.0, color_decode="auto", far_value="larger"):
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None: raise FileNotFoundError(path)
    return decode_depth_array(dep, size=size, mode=mode, scale=scale, color_decode=color_decode, far_value=far_value)

def _visualize_depth_for_click(depth_np: np.ndarray) -> np.ndarray:
    d = depth_np.copy().astype(np.float32)
    vals = d[np.isfinite(d)]
    if vals.size == 0:
        dmin, dmax = 0.0, 1.0
    else:
        dmin = float(np.percentile(vals, 1.0))
        dmax = float(np.percentile(vals, 99.0))
        if dmax <= dmin: dmax = dmin + 1e-6
    v = np.clip((depth_np - dmin) / (dmax - dmin), 0.0, 1.0)
    v8 = (v*255.0).astype(np.uint8)
    cm = cv2.applyColorMap(v8, cv2.COLORMAP_TURBO)
    return cm

def pick_focus_depth_gui(depth_np: np.ndarray, win_name: str = "深度图 - 点击选择对焦深度，ESC/Enter确认") -> float:
    H, W = depth_np.shape
    show = _visualize_depth_for_click(depth_np)
    clicked = {"z": None}
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            z = float(depth_np[y, x])
            if not np.isfinite(z) or z <= 1e-6: return
            clicked["z"] = z
            disp = show.copy()
            cv2.circle(disp, (x, y), 5, (0,0,255), 2)
            txt = f"z={z:.4f}  (x={x}, y={y})  Enter/ESC确认"
            cv2.putText(disp, txt, (10, max(25, H-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(win_name, disp)
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, show)
    cv2.setMouseCallback(win_name, cb)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, 13): break
    cv2.destroyWindow(win_name)
    if clicked["z"] is not None:
        return clicked["z"]
    valid = depth_np[np.isfinite(depth_np) & (depth_np > 1e-6)]
    return float(np.median(valid) if valid.size > 0 else 1.0)

def euler_xyz_to_matrix(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(rx_deg); ry = np.deg2rad(ry_deg); rz = np.deg2rad(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def build_lookat_rotation_from_campos(C: np.ndarray, target: np.ndarray, up_hint: np.ndarray = None) -> np.ndarray:
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd = (target.reshape(3,) - C.reshape(3,)).astype(np.float32)
    n = np.linalg.norm(fwd)
    if n < 1e-8: fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else: fwd = fwd / n
    if abs(np.dot(fwd, up_hint)/(np.linalg.norm(up_hint)+1e-8)) > 0.999:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(fwd[2]) < 0.999 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = np.cross(up_hint, fwd); nx = np.linalg.norm(x_axis); x_axis = x_axis/nx if nx>1e-8 else np.array([1.0,0.0,0.0], np.float32)
    y_axis = np.cross(fwd, x_axis); ny = np.linalg.norm(y_axis); y_axis = y_axis/ny if ny>1e-8 else np.array([0.0,1.0,0.0], np.float32)
    R = np.stack([x_axis, y_axis, fwd], axis=1).astype(np.float32)
    return R

def build_convergent_camera_motion(tx: float, focus_z: float) -> np.ndarray:
    C = np.array([tx, 0.0, 0.0], dtype=np.float32)
    P_focus = np.array([0.0, 0.0, max(1e-6, float(focus_z))], dtype=np.float32)
    R_cam = build_lookat_rotation_from_campos(C, P_focus, up_hint=np.array([0.0,1.0,0.0], dtype=np.float32))
    dT = np.eye(4, dtype=np.float32)
    dT[:3, :3] = R_cam
    dT[:3, 3] = C
    return dT

def compute_tx_from_disp(fx_px: float, depth_ref: float, disp_px: float) -> float:
    return float(disp_px) * float(depth_ref) / max(1e-6, float(fx_px))

def build_offsets(tx_max: float, num_per_side: int, spacing: str):
    N = int(max(0, num_per_side))
    if N == 0: return [0.0]
    k = np.arange(-N, N+1, dtype=np.float32)
    if spacing == "linear":
        s = k / max(1, N)
    else:
        s = np.sin(0.5 * np.pi * (k/max(1,N)))
    return (s * float(tx_max)).tolist()

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
            for e in exts: files.extend(glob.glob(os.path.join(path, e)))
            files = sorted(files, key=natural_key)
            if not files: raise FileNotFoundError(f"目录为空: {path}")
            self.files = files
            self.frame_count = len(self.files)
        elif is_video_file(path):
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened(): raise RuntimeError(f"无法打开视频: {path}")
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if self.fps <= 0.0: self.fps = None
        else:
            raise FileNotFoundError(f"不是视频文件或目录: {path}")
    def read(self):
        if self.files is not None:
            if self.idx >= len(self.files): return False, None
            fp = self.files[self.idx]; self.idx += 1
            flag = cv2.IMREAD_UNCHANGED if self.is_depth else cv2.IMREAD_COLOR
            img = cv2.imread(fp, flag)
            if img is None: return False, None
            return True, img
        else:
            ok, frame = self.cap.read()
            if not ok: return False, None
            return True, frame
    def release(self):
        if self.cap is not None: self.cap.release()

def get_mask(rows, cols, theta, N, X=4.66666, koff=5):
    y, x = np.mgrid[0:rows, 0:cols*3].astype(np.float32)
    r = (np.mod((x + koff - 3.0*y*np.tan(np.float32(theta))), np.float32(X))) * (np.float32(N)/np.float32(X))
    pattern = np.floor(r)
    # 使用更大的类型避免较大N时溢出
    mask = np.zeros((rows, cols, 3), dtype=np.int32)
    mask[:, :, 0] = pattern[:, 0::3].astype(np.int32)
    mask[:, :, 1] = pattern[:, 1::3].astype(np.int32)
    mask[:, :, 2] = pattern[:, 2::3].astype(np.int32)
    return mask

def create_3d_img(mask, views):
    """
    融合函数（GPU优先）：
    - views: torch.Tensor [N,H,W,3] (BGR, uint8，device=CPU/CUDA) 或 List[np.uint8 HxWx3 BGR]
    - mask:  numpy(int32) HxWx3 或 torch(long) HxWx3（建议：预先在与views相同device、dtype=torch.long缓存，避免重复拷贝）
    返回: numpy.uint8 HxWx3 BGR
    """
    if isinstance(views, torch.Tensor):
        assert views.dim() == 4 and views.size(-1) == 3, "views 需为 (N,H,W,3)"
        N, H, W, _ = views.shape
        device = views.device

        # 优先走已在正确device/long类型的快速路径
        if isinstance(mask, torch.Tensor) and (mask.device == device) and (mask.dtype == torch.long):
            m = torch.clamp(mask, 0, max(0, N-1))
        else:
            if isinstance(mask, np.ndarray):
                m = torch.from_numpy(mask.astype(np.int64, copy=False)).to(device)
            elif isinstance(mask, torch.Tensor):
                m = mask.to(device=device, dtype=torch.long)
            else:
                raise TypeError("mask 需为 numpy.ndarray 或 torch.Tensor")
            m = torch.clamp(m, 0, max(0, N-1))

        V = views.permute(3, 0, 1, 2).contiguous()  # (3,N,H,W), uint8
        out_chs = []
        for c in range(3):
            out_c = torch.gather(V[c], dim=0, index=m[..., c].unsqueeze(0)).squeeze(0)  # (H,W), uint8
            out_chs.append(out_c)
        out = torch.stack(out_chs, dim=-1)  # (H,W,3), BGR, uint8
        return out.detach().cpu().numpy()

    # Numpy回退路径
    views_np = np.stack(views, axis=0)  # (N,H,W,3), uint8, BGR
    N, H, W, _ = views_np.shape
    out = np.empty((H, W, 3), dtype=np.uint8)
    m = np.clip(mask, 0, N-1).astype(np.int64)
    for c in range(3):
        out[..., c] = np.take_along_axis(views_np[..., c], m[..., c][None, ...], axis=0)[0]
    return out

def rescale_depth_tensor_linear_torch(Ds: torch.Tensor, rng=(2.0, 10.0), percentiles=(1.0, 99.0)) -> torch.Tensor:
    """
    纯GPU版本，避免CPU往返和同步：
    Ds: (1,1,H,W) on device
    """
    if not torch.is_floating_point(Ds):
        Ds = Ds.float()
    m = torch.isfinite(Ds) & (Ds > 1e-6)
    if m.sum() < 16:
        return Ds
    plo, phi = float(percentiles[0]), float(percentiles[1])
    v = Ds[m]
    z0 = torch.quantile(v, plo / 100.0)
    z1 = torch.quantile(v, phi / 100.0)
    if not torch.isfinite(z0) or not torch.isfinite(z1) or (z1 <= z0):
        return Ds
    a, b = float(rng[0]), float(rng[1])
    d2 = a + (Ds - z0) * (b - a) / (z1 - z0)
    d2 = torch.clamp(d2, min(a, b), max(a, b))
    return d2.to(Ds.dtype)

_mask_device_cache = {}
def get_cached_mask_torch(H, W, N, device, slope, X, koff):
    key = (H, W, N, float(slope), float(X), float(koff), str(device))
    if key in _mask_device_cache:
        return _mask_device_cache[key]
    theta = math.atan(float(slope))
    m_np = get_mask(H, W, theta, N, X=float(X), koff=float(koff)).astype(np.int64, copy=False)
    m = torch.from_numpy(m_np).to(device=device, dtype=torch.long)
    _mask_device_cache[key] = m
    return m


def _maybe_to_channels_last(x: torch.Tensor, enable: bool) -> torch.Tensor:
    if enable and x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x

def warp_and_refine_batch(Is, Ds, Ks, focus_z, offsets, device, splat_cfg, refiner_model, amp_enabled,
                          chunk_size=0, precomp=None, channels_last=False, sync_timing=False,
                          band_px=6, band_as_edit=False, band_alpha=0.0, mask_eps=1e-8):
    N = len(offsets)
    if N == 0:
        return torch.empty(0), 0.0, 0.0
    B, C, H, W = Is.shape

    if precomp is not None:
        Ks_b = precomp.get("Ks_b", None)
        Kt_b = precomp.get("Kt_b", None)
        dT_b = precomp.get("dT_b", None)
    else:
        Ks_b = None; Kt_b = None; dT_b = None

    if Ks_b is None:
        Ks_b = Ks.expand(N, -1, -1)
    if Kt_b is None:
        Kt_b = Ks_b
    if dT_b is None:
        dT_list = [build_convergent_camera_motion(tx=o, focus_z=focus_z) for o in offsets]
        dT_b = torch.from_numpy(np.stack(dT_list, axis=0)).float().to(device)

    if chunk_size is None or chunk_size <= 0:
        chunk = N
    else:
        chunk = int(max(1, min(chunk_size, N)))

    chunks = []
    t_warp_total = 0.0
    t_refine_total = 0.0

    def build_ring_band(hole_bool, radius):
        if radius <= 0:
            return torch.zeros_like(hole_bool)
        k = int(radius)
        kernel = 2 * k + 1
        pad = k
        dilated = F.max_pool2d(hole_bool.float(), kernel_size=kernel, stride=1, padding=pad)
        ring = (dilated > 0) & (~hole_bool)
        return ring.to(hole_bool.dtype)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Is_c = Is.expand(e - s, -1, -1, -1)
        Ds_c = Ds.expand(e - s, -1, -1, -1)
        Ks_c = Ks_b[s:e]
        Kt_c = Kt_b[s:e]
        dT_c = dT_b[s:e]

        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                Iw, V = softmax_splat(
                    Is_c, Ds_c, Ks_c, Kt_c, dT_c,
                    temperature=float(splat_cfg["temperature"]),
                    normalize=True,
                    occlusion=splat_cfg["occlusion"],
                    hard_z_epsilon=float(splat_cfg["hard_z_epsilon"])
                )
        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_warp_total += (time.time() - t0)

        hole_bool = (V <= mask_eps)
        hole = hole_bool.to(Iw.dtype)
        band = build_ring_band(hole_bool, int(band_px)).to(Iw.dtype)
        union = ((hole_bool | (band > 0)) if band_px > 0 else hole_bool).to(Iw.dtype)

        edit_mask = union if band_as_edit else hole
        Iw_proc = (1.0 - edit_mask) * Iw

        x = torch.cat([Iw_proc, hole, band], dim=1)
        x = _maybe_to_channels_last(x, channels_last)

        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                Ipred = refiner_model(x)
        if Ipred.dtype != Iw.dtype:
            Ipred = Ipred.to(Iw.dtype)

        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_refine_total += (time.time() - t1)

        if band_as_edit:
            out_mask = union
            Ifinal = out_mask * Ipred + (1.0 - out_mask) * Iw
        else:
            Ifinal = hole * Ipred + (1.0 - hole) * Iw
            if band_alpha > 0.0:
                alpha = float(max(0.0, min(1.0, band_alpha)))
                band3 = band.expand(-1, 3, -1, -1)
                soft_band = alpha * Ipred + (1.0 - alpha) * Iw
                Ifinal = band3 * soft_band + (1.0 - band3) * Ifinal
        Ifinal = Ifinal.clamp_(0, 1)

        Ifinal_u8 = (Ifinal * 255.0).to(torch.uint8)
        views_rgb = Ifinal_u8.permute(0, 2, 3, 1).contiguous()
        idx_bgr = _get_idx_bgr(views_rgb.device)
        views_bgr = views_rgb.index_select(-1, idx_bgr)

        chunks.append(views_bgr)

        del Iw, V, hole, band, union, edit_mask, Iw_proc, x, Ipred, Ifinal, Ifinal_u8, views_rgb, views_bgr

    views_stacked = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    return views_stacked, t_warp_total, t_refine_total



def read_meta_txt(meta_path: str, pose_convention: str = "w2c"):
    if not meta_path or (not os.path.isfile(meta_path)):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    frames = {}
    for line in lines[1:]:
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        if len(parts) < 19: continue
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
        T = np.eye(4, dtype=np.float32); T[:3, :4] = P
        if pose_convention.lower() == "c2w":
            T = np.linalg.inv(T).astype(np.float32)
        frames[ts] = (K, T)
    return frames

def extract_timestamp_from_frame_path(p):
    base = os.path.basename(p)
    m = re.search(r"frame_(\d+)", base)
    if not m: return None
    try: return int(m.group(1))
    except Exception: return None

def choose_lossless_writer(path: str, fps: float, size_wh):
    W, H = size_wh
    base, ext = os.path.splitext(path)
    trials = []
    trials.append(("FFV1", base + ".mkv"))
    trials.append(("HFYU", base + ".avi"))
    trials.append(("LAGS", base + ".avi"))
    trials.append(("MJPG", base + ".avi"))
    for fourcc_str, outp in trials:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w = cv2.VideoWriter(outp, fourcc, float(fps), (W, H), isColor=True)
        if w.isOpened():
            return w, outp, fourcc_str
    raise RuntimeError("无法创建无损视频写入器（FFV1/HFYU/LAGS 均不可用）。请安装带FFmpeg支持的OpenCV或改用PNG序列。")

class FFmpegPipeWriter:
    def __init__(self, out_path: str, fps: float, size_wh, close_timeout: float = 300.0):
        self.W, self.H = size_wh
        self.proc = None
        self.out_path = out_path
        self.fps = float(fps)
        self.close_timeout = float(close_timeout)
        self._start()

    @staticmethod
    def _has_ffmpeg():
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return True
        except Exception:
            return False

    def _start(self):
        if not FFmpegPipeWriter._has_ffmpeg():
            raise RuntimeError("未检测到 ffmpeg。请安装后使用 --ffmpeg_lossless。")
        base, ext = os.path.splitext(self.out_path)
        outp = self.out_path if ext else (base + ".mkv")
        cmd_try = [
            ["ffmpeg", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pixel_format", "bgr24",
             "-video_size", f"{self.W}x{self.H}", "-framerate", str(self.fps),
             "-i", "-", "-c:v", "libx265", "-preset", "medium",
             "-x265-params", "lossless=1", "-pix_fmt", "yuv444p", outp],
            ["ffmpeg", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pixel_format", "bgr24",
             "-video_size", f"{self.W}x{self.H}", "-framerate", str(self.fps),
             "-i", "-", "-c:v", "libx264", "-preset", "veryslow",
             "-crf", "0", "-pix_fmt", "yuv444p", outp],
        ]
        err = None
        for cmd in cmd_try:
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                self.out_path = outp
                return
            except Exception as e:
                err = e
                self.proc = None
        raise RuntimeError(f"启动 ffmpeg 失败: {err}")

    def write(self, frame_bgr_uint8: np.ndarray):
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("FFmpeg 管道未启动")
        if frame_bgr_uint8.shape[0] != self.H or frame_bgr_uint8.shape[1] != self.W:
            frame_bgr_uint8 = cv2.resize(frame_bgr_uint8, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        self.proc.stdin.write(frame_bgr_uint8.tobytes())

    def release(self):
        if self.proc is None:
            return
        try:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=self.close_timeout)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    try:
                        self.proc.wait(timeout=10)
                    except Exception:
                        pass
        finally:
            self.proc = None

class FFmpegH264Writer:
    def __init__(self, out_path: str, fps: float, size_wh, qp: int = 23, preset: str = "medium",
                 pix_fmt: str = "yuv420p", codec: str = "libx264", close_timeout: float = 120.0):
        self.W, self.H = size_wh
        self.proc = None
        self.out_path = out_path
        self.fps = float(fps)
        self.close_timeout = float(close_timeout)
        self.qp = int(max(0, min(51, qp)))
        self.preset = str(preset)
        self.pix_fmt_req = str(pix_fmt)
        self.pix_fmt = self.pix_fmt_req
        self.codec = str(codec)
        if self.pix_fmt in ("yuv420p", "yuv422p") and ((self.W % 2) or (self.H % 2)):
            print(f"[H264] 检测到尺寸为 {self.W}x{self.H}（含奇数），已将 pix_fmt 从 {self.pix_fmt} 自动改为 yuv444p 以避免编码失败")
            self.pix_fmt = "yuv444p"
        self._start()

    def _start(self):
        if not FFmpegPipeWriter._has_ffmpeg():
            raise RuntimeError("未检测到 ffmpeg。请安装后使用 --ffmpeg_h264。")
        base, ext = os.path.splitext(self.out_path)
        outp = self.out_path if ext else (base + ".mp4")
        cmd = [
            "ffmpeg", "-loglevel", "error", "-hide_banner", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{self.W}x{self.H}", "-framerate", str(self.fps),
            "-i", "-", "-an",
            "-c:v", self.codec,
            "-preset", self.preset,
            "-qp", str(self.qp),
            "-pix_fmt", self.pix_fmt,
            outp
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.out_path = outp
        except Exception as e:
            raise RuntimeError(f"启动 ffmpeg({self.codec}) 失败: {e}")

    def write(self, frame_bgr_uint8: np.ndarray):
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("FFmpeg(h264) 管道未启动")
        if frame_bgr_uint8.shape[0] != self.H or frame_bgr_uint8.shape[1] != self.W:
            frame_bgr_uint8 = cv2.resize(frame_bgr_uint8, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        self.proc.stdin.write(frame_bgr_uint8.tobytes())

    def release(self):
        if self.proc is None:
            return
        try:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=self.close_timeout)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    try:
                        self.proc.wait(timeout=10)
                    except Exception:
                        pass
        finally:
            self.proc = None

class AsyncWriter:
    def __init__(self, writer, max_queue: int = 32):
        self.writer = writer
        self.q = Queue(maxsize=max_queue)
        self._stop = False
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                self.writer.write(item)
            except Exception as e:
                print(f"[AsyncWriter] 写入错误: {e}")
            finally:
                self.q.task_done()

    def write(self, frame_bgr_uint8: np.ndarray):
        self.q.put(frame_bgr_uint8, block=True)

    def release(self):
        try:
            self.q.put(None)
            self.th.join()
        finally:
            try:
                self.writer.release()
            except Exception:
                pass

def iter_video_pairs_prefetch(rgb_reader: StreamReader, dep_reader: StreamReader, start_idx: int = 1, prefetch: int = 16):
    q = Queue(maxsize=prefetch)
    SENTINEL = object()

    def _read_loop():
        idx = start_idx
        while True:
            ok1, r = rgb_reader.read()
            ok2, d = dep_reader.read()
            if (not ok1) or (not ok2):
                break
            q.put((idx, r, d))
            idx += 1
        q.put(SENTINEL)

    th = threading.Thread(target=_read_loop, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        yield item


def maybe_resize_views(views, fuse_size, resize_mode="stretch"):
    if fuse_size is None:
        return views
    Hf, Wf = fuse_size
    resize_mode = (resize_mode or "stretch").lower()

    # Torch路径：views 为 (N,H,W,3) uint8 BGR
    if isinstance(views, torch.Tensor):
        assert views.dim() == 4 and views.size(-1) == 3, "views 需为 (N,H,W,3)"
        N, H, W, C = views.shape

        if (H == Hf) and (W == Wf):
            # 已与目标一致，直接返回
            return views

        if resize_mode == "keep_ar_pad":
            # 等比缩放到不超过目标尺寸，然后居中黑边填充
            s = min(float(Hf) / max(1.0, float(H)), float(Wf) / max(1.0, float(W)))
            H1 = max(1, int(round(H * s)))
            W1 = max(1, int(round(W * s)))

            # 先resize到(H1,W1)
            x = views.permute(0, 3, 1, 2).to(torch.float32) / 255.0  # (N,3,H,W)
            y = F.interpolate(x, size=(H1, W1), mode="bilinear", align_corners=False)
            y8 = (y.clamp(0, 1) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()  # (N,H1,W1,3)

            # 创建黑色画布并将缩放后的图像居中贴到目标画布上
            canvas = torch.zeros((N, Hf, Wf, 3), dtype=torch.uint8, device=views.device)
            top = (Hf - H1) // 2
            left = (Wf - W1) // 2
            canvas[:, top:top+H1, left:left+W1, :] = y8
            return canvas
        else:
            # stretch: 直接拉伸到目标尺寸
            x = views.permute(0, 3, 1, 2).to(torch.float32) / 255.0  # (N,3,H,W)
            y = F.interpolate(x, size=(Hf, Wf), mode="bilinear", align_corners=False)
            y8 = (y.clamp(0, 1) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()  # (N,Hf,Wf,3)
            return y8

    # Numpy路径（兼容旧用法）：views 为 List[np.ndarray HxWx3 BGR, uint8]
    out = []
    for v in views:
        H, W = v.shape[:2]
        if (H == Hf) and (W == Wf):
            out.append(v)
            continue
        if resize_mode == "keep_ar_pad":
            s = min(float(Hf) / max(1.0, float(H)), float(Wf) / max(1.0, float(W)))
            H1 = max(1, int(round(H * s)))
            W1 = max(1, int(round(W * s)))
            v1 = cv2.resize(v, (W1, H1), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((Hf, Wf, 3), dtype=np.uint8)
            top = (Hf - H1) // 2
            left = (Wf - W1) // 2
            canvas[top:top+H1, left:left+W1, :] = v1
            out.append(canvas)
        else:
            out.append(cv2.resize(v, (Wf, Hf), interpolation=cv2.INTER_LINEAR))
    return out



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

def _precompute_warp_static(Ks: torch.Tensor, focus_z: float, offsets, device: str):
    N = len(offsets)
    Ks_b = Ks.expand(N, -1, -1)
    Kt_b = Ks_b
    dT_list = [build_convergent_camera_motion(tx=o, focus_z=focus_z) for o in offsets]
    dT_b = torch.from_numpy(np.stack(dT_list, axis=0)).float().to(device)
    return {"Ks_b": Ks_b, "Kt_b": Kt_b, "dT_b": dT_b, "N": N}

def _estimate_chunk_size(H, W, N, C=3, amp=True):
    # 简单启发式：按显存可用空间估计一次处理多少视角
    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024**3)
    except Exception:
        return N
    bytes_per = 2 if amp else 4
    # 近似：numer(C*H*W) + denom(H*W) + 中间量（约2倍）
    per_view_bytes = (C*H*W + H*W) * bytes_per * 3.0
    if per_view_bytes <= 0:
        return N
    max_views = int(max(1, min(N, (free_mem * 0.5) // per_view_bytes)))
    return max_views if max_views > 0 else 1

def to_tensor_rgb_bgr(bgr, H_warp, W_warp, device, pin=True):
    if (bgr.shape[0] != H_warp) or (bgr.shape[1] != W_warp):
        bgr = cv2.resize(bgr, (W_warp, H_warp), interpolation=cv2.INTER_AREA)
    # 直接通道翻转（BGR->RGB），比 cv2.cvtColor 更轻
    rgb = bgr[..., ::-1].copy()  # .copy() 确保连续内存
    ten = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if pin and ten.device.type == "cpu" and torch.cuda.is_available():
        ten = ten.pin_memory()
        return ten.to(device, non_blocking=True)
    return ten.to(device)


def main():
    ap = argparse.ArgumentParser("端到端 裸眼3D 生成 (图像/目录/视频) + 无损输出 + 可指定合成分辨率（已优化加速）")
    ap.add_argument("--image", help="单张RGB图路径")
    ap.add_argument("--depth", help="对应深度图路径（可彩色或灰度；uint16 metric或0~255归一化均可）")
    ap.add_argument("--pair_dir", help="目录模式：目录内有 frame_xxx.(jpg/png) 和 depth/depth_xxx.png")
    ap.add_argument("--video", help="RGB视频（文件）")
    ap.add_argument("--depth_video", help="深度视频（文件）或深度帧目录")

    ap.add_argument("--img_size", default="0,0", help="warp尺寸 H,W (0,0=保持输入)")
    ap.add_argument("--fuse_size", default="2160,3840", help="合成裸眼3D输出尺寸 H,W (0,0=跟随warp尺寸)")
    ap.add_argument("--fuse_resize_mode", default="stretch", choices=["stretch", "keep_ar_pad"], help="合成前缩放模式：stretch=直接拉伸到目标尺寸；keep_ar_pad=等比缩放后居中黑边填充到目标尺寸"
    )
    ap.add_argument("--manual_K", default=None, help="手动相机内参 'fx,fy,cx,cy'")
    ap.add_argument("--K_units", default="auto", choices=["auto","pixel","normalized"])
    ap.add_argument("--meta", default=None, help="目录/单图模式：clip的 .txt meta（包含K与位姿），用于获得每帧K")
    ap.add_argument("--pose_convention", default="w2c", choices=["w2c","c2w"])
    ap.add_argument("--fps", type=float, default=30.0, help="输出视频帧率（目录/视频模式）")

    ap.add_argument("--depth_mode", default="auto", choices=["auto","metric","normalized"])
    ap.add_argument("--depth_scale", type=float, default=10.0)
    ap.add_argument("--depth_color_decode", default="auto", choices=["auto","pca","luma","rgb24","r","g","b"])
    ap.add_argument("--far_value", default="larger", choices=["larger","smaller"], help="深度值更大是否代表更远")

    ap.add_argument("--rescale_depth", default="linear", choices=["none","linear"], help="warp前线性重标定深度")
    ap.add_argument("--rescale_range", default="2,10", help="重标定目标区间 [dmin,dmax]")
    ap.add_argument("--rescale_percentiles", default="1,99", help="用于估计线性映射的分位点")

    ap.add_argument("--num_per_side", type=int, default=4, help="左右各视角数（包含中心共2N+1）")
    ap.add_argument("--spacing", default="linear", choices=["linear","cosine"])
    ap.add_argument("--tx_max", type=float, default=0.0, help="最大水平平移（>0则使用该值）")
    ap.add_argument("--max_disp_px", type=float, default=48.0, help="若tx_max<=0，根据fx与参考深度换算")
    ap.add_argument("--disp_ref_percentile", type=float, default=0.5, help="参考深度分位点（0~1）")

    ap.add_argument("--band_px", type=int, default=2, help="污染带半径像素")
    ap.add_argument("--band_as_edit", action="store_true", help="将污染带视为编辑区域（强制替换）")
    ap.add_argument("--band_alpha", type=float, default=0.0, help="污染带软融合系数 0~1")

    ap.add_argument("--temperature", type=float, default=30.0)
    ap.add_argument("--occlusion", default="hard", choices=["hard","soft"])
    ap.add_argument("--hard_z_epsilon", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true", help="启用AMP半精度（需CUDA）")

    ap.add_argument("--refiner_type", default="MGMI", choices=["MGMI","InpaintRefiner"], help="补洞模型类型")
    ap.add_argument("--refiner_ckpt", required=True, help="补洞模型权重路径（建议使用EMA权重）")

    ap.add_argument("--slope", type=float, default=0.166666, help="mask斜率，theta=atan(slope)")
    ap.add_argument("--X", type=float, default=4.66666, help="光栅周期参数")
    ap.add_argument("--koff", type=float, default=5.0, help="相位偏移参数")

    ap.add_argument("--focus_depth", type=float, default=0.0, help=">0则直接使用该深度为汇聚深度；视频模式默认弹出GUI")
    ap.add_argument("--auto_focus_face", action="store_true", help="视频模式：启用人脸自动对焦，逐帧检测最近人脸中心深度作为对焦深度；优先级高于 --focus_depth 与 GUI 手动选择")
    ap.add_argument("--out", required=True, help="输出路径：图像模式为图片；目录/视频模式为视频")

    ap.add_argument("--gpu", type=int, default=-1, help="-1自动；>=0 指定GPU编号")
    ap.add_argument("--save_views_dir", default=None, help="若指定，保存所有视角到该目录")
    ap.add_argument("--chunk_views", type=int, default=0, help="每次处理的视角数，0=不分块")

    ap.add_argument("--lossless", action="store_true", help="使用OpenCV无损编码（FFV1/HFYU/LAGS自动尝试）")
    ap.add_argument("--ffmpeg_lossless", action="store_true", help="使用FFmpeg无损管道（优先libx265 lossless，回退libx264 CRF 0），文件更小")
    ap.add_argument("--ffmpeg_close_timeout", type=float, default=300.0, help="FFmpeg 关闭阶段的最长等待秒数（x265 无损 4K 建议 300+）")

    ap.add_argument("--ffmpeg_h264", action="store_true", help="使用 FFmpeg H.264 压缩（libx264），可设置 QP。")
    ap.add_argument("--h264_codec", default="libx264", choices=["libx264", "h264_nvenc", "hevc_nvenc"], help="H.264/H.265 编码器（CPU或NVENC）")
    ap.add_argument("--h264_qp", type=int, default=23, help="H.264 QP（0-51，越小越清晰，默认23）")
    ap.add_argument("--h264_preset", default="medium", help="x264 preset（ultrafast~veryslow，默认medium）")
    ap.add_argument("--h264_pix_fmt", default="yuv420p", choices=["yuv420p","yuv444p"], help="像素格式（默认yuv420p，保真可用yuv444p）")



    # 新增加速开关
    ap.add_argument("--channels_last", action="store_true", help="使用 channels_last 内存格式加速Conv（默认关闭）")
    ap.add_argument("--compile", action="store_true", help="使用 torch.compile 加速Refiner（PyTorch 2.0+，默认关闭）")
    ap.add_argument("--tf32", action="store_true", default=True, help="启用TF32（Ampere+，默认开启）")
    ap.add_argument("--threads", type=int, default=0, help="设置OpenCV/PyTorch线程数（0=不改动）")
    ap.add_argument("--autotune_chunk", action="store_true", help="自动估计 chunk_views（默认关闭）")
    ap.add_argument("--sync_timing", action="store_true", help="严格计时（强制CUDA同步，默认关闭以提升速度）")
    ap.add_argument("--async_io", action="store_true", help="启用异步I/O（预取读取 + 后台写盘编码），减少帧间等待")


    args = ap.parse_args()

    if args.threads and args.threads > 0:
        try:
            torch.set_num_threads(int(args.threads))
        except Exception:
            pass
        try:
            cv2.setNumThreads(int(args.threads))
        except Exception:
            pass

    if torch.cuda.is_available() and args.gpu is not None and args.gpu >= 0:
        device = f"cuda:{int(args.gpu)}"
        torch.cuda.set_device(int(args.gpu))
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = bool(args.amp) and str(device).startswith("cuda")
    torch.set_grad_enabled(False)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    Hs, Ws = [int(x) for x in args.img_size.replace(",", " ").split()]
    resize_to = None if (Hs==0 or Ws==0) else (Hs,Ws)
    Hf, Wf = [int(x) for x in args.fuse_size.replace(",", " ").split()]
    fuse_to = None if (Hf==0 or Wf==0) else (Hf, Wf)

    splat_cfg = {
        "temperature": float(args.temperature),
        "occlusion": args.occlusion,
        "hard_z_epsilon": float(args.hard_z_epsilon)
    }

    refiner_cfg = {
        "model": {
            "refiner": {
                "type": args.refiner_type,
                "in_ch": 5,
                "out_ch": 3,
                "base_ch": 24 if args.refiner_type.lower() == "mgmi" else 48,
                "width_mult": 1.25 if args.refiner_type.lower() == "mgmi" else 1.0,
                "depth": 5 if args.refiner_type.lower() == "inpaintrefiner" else None,
                "aspp_rates": [1, 2, 4, 8],
                "act": "silu",
                "norm": "bn",
                "use_se": True
            }
        }
    }
    if args.refiner_type.lower() == "mgmi":
        refiner_cfg["model"]["refiner"].pop("depth", None)
    else:
        for k in ["width_mult","aspp_rates","act","norm","use_se"]:
            refiner_cfg["model"]["refiner"].pop(k, None)

    refiner = build_refiner(refiner_cfg).to(device).eval()
    ckpt = args.refiner_ckpt
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Refiner权重不存在: {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    state = state.get("model", state)
    clean = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    refiner.load_state_dict(clean, strict=False)

    if args.channels_last:
        refiner = refiner.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        try:
            refiner = torch.compile(refiner)  # PyTorch 2.0+
            print("[Init] 使用 torch.compile 加速 Refiner")
        except Exception as e:
            warnings.warn(f"torch.compile 失败，回退普通模式: {e}")

    print(f"[Init] Refiner OK | 设备: {device} | AMP: {amp_enabled} | channels_last: {args.channels_last}")

    mask_cache = {}
    def get_cached_mask(H, W, N):
        key = (H, W, N, args.slope, args.X, args.koff)
        if key in mask_cache:
            return mask_cache[key]
        theta = math.atan(float(args.slope))
        m = get_mask(H, W, theta, N, X=float(args.X), koff=float(args.koff))
        mask_cache[key] = m
        return m

    def unify_K_for_size(K_in: np.ndarray, src_hw, dst_hw):
        if is_normalized_K(K_in):
            return scale_intrinsics(K_in, dst_hw[0], dst_hw[1])
        else:
            return resize_intrinsics(K_in, src_hw, dst_hw)

    def make_K(H, W, src_hw=None, meta_K=None):
        if args.manual_K is not None:
            K_manual = parse_manual_K(args.manual_K)
            if args.K_units == "normalized" or (args.K_units=="auto" and is_normalized_K(K_manual)):
                Kpx = scale_intrinsics(K_manual, H, W)
            elif src_hw is not None:
                Kpx = resize_intrinsics(K_manual, src_hw, (H, W))
            else:
                Kpx = K_manual.astype(np.float32)
        elif meta_K is not None:
            Kpx = unify_K_for_size(meta_K, src_hw, (H, W))
        else:
            Kpx = default_K(H, W)
        return torch.from_numpy(Kpx).float().unsqueeze(0).to(device)

    def compute_offsets(Ks_np, Dnp):
        fx_px = float(Ks_np[0,0])
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
        return offsets

    def maybe_save_views(views_bgr, offsets, stem, save_dir):
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)

        # Torch路径：views_bgr 是 (N,H,W,3) BGR uint8
        if isinstance(views_bgr, torch.Tensor):
            v_cpu = views_bgr.detach().cpu().numpy()  # (N,H,W,3), BGR, uint8
            for i in range(v_cpu.shape[0]):
                fn = f"{stem}_s{i:02d}_tx{offsets[i]:+.5f}.png"
                cv2.imwrite(os.path.join(save_dir, fn), v_cpu[i])
            return

        # 兼容旧用法：List[np.ndarray] BGR
        for i, (img, tx) in enumerate(zip(views_bgr, offsets)):
            fn = f"{stem}_s{i:02d}_tx{tx:+.5f}.png"
            cv2.imwrite(os.path.join(save_dir, fn), img)

    rmin, rmax = [float(x) for x in args.rescale_range.replace(",", " ").split()]
    plo, phi = [float(x) for x in args.rescale_percentiles.replace(",", " ").split()]
    rescale_on = (args.rescale_depth == "linear")
    rescale_cfg = (rmin, rmax, plo, phi)

    is_image_mode = (args.image is not None and args.depth is not None and args.pair_dir is None and args.video is None and args.depth_video is None)
    is_dir_mode   = (args.pair_dir is not None and args.image is None and args.depth is None and args.video is None and args.depth_video is None)
    is_video_mode = (args.video is not None and args.depth_video is not None and args.image is None and args.depth is None and args.pair_dir is None)
    if not (is_image_mode or is_dir_mode or is_video_mode):
        raise ValueError("需三选一：图像模式(--image --depth) 或 目录模式(--pair_dir) 或 视频模式(--video --depth_video)。")

    # 功能函数：构造输入张量（可Pinned Memory + 非阻塞拷贝）
    def to_tensor_rgb_bgr(bgr, H_warp, W_warp, device, pin=True):
        if (bgr.shape[0] != H_warp) or (bgr.shape[1] != W_warp):
            bgr = cv2.resize(bgr, (W_warp, H_warp), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0) / 255.0
        if pin and ten.device.type == "cpu" and torch.cuda.is_available():
            ten = ten.pin_memory()
            return ten.to(device, non_blocking=True)
        return ten.to(device)

    band_px = int(max(0, args.band_px))
    band_as_edit = bool(args.band_as_edit)
    band_alpha = float(args.band_alpha)

    # 下面分别处理三种模式
    if is_image_mode:
        Is, (H0, W0) = load_rgb(args.image, size=resize_to)
        Ds, _ = load_depth(args.depth, size=resize_to, mode=args.depth_mode, scale=args.depth_scale,
                           color_decode=args.depth_color_decode, far_value=args.far_value)
        if rescale_on:
            Ds = rescale_depth_tensor_linear(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
        H, W = Is.shape[-2:]
        meta_map = read_meta_txt(args.meta, pose_convention=args.pose_convention) if args.meta else {}
        ts = extract_timestamp_from_frame_path(args.image)
        meta_K = meta_map.get(ts, (None, None))[0] if (ts is not None and ts in meta_map) else None
        Ks = make_K(H, W, src_hw=(H0, W0), meta_K=meta_K); Ks_np = Ks.squeeze(0).cpu().numpy()

        Dnp = Ds.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        offsets = compute_offsets(Ks_np, Dnp)
        N = len(offsets)

        if args.focus_depth > 0:
            focus_z = float(args.focus_depth)
        else:
            valid = Dnp[np.isfinite(Dnp) & (Dnp > 1e-6)]
            focus_z = float(np.median(valid) if valid.size > 0 else 1.0)

        # 预计算静态参数
        precomp = _precompute_warp_static(Ks.to(device), focus_z, offsets, device)

        # 自动估计chunk
        chunk_views = args.chunk_views
        if (chunk_views <= 0) and args.autotune_chunk and str(device).startswith("cuda"):
            chunk_views = _estimate_chunk_size(H, W, N, C=3, amp=amp_enabled)

        Is = _maybe_to_channels_last(Is.to(device), args.channels_last)
        Ds = _maybe_to_channels_last(Ds.to(device), args.channels_last)
        views_bgr, t_warp, t_refine = warp_and_refine_batch(
            Is, Ds, Ks.to(device), focus_z, offsets, device, splat_cfg, refiner, amp_enabled,
            chunk_size=chunk_views, precomp=precomp, channels_last=args.channels_last,
            sync_timing=args.sync_timing, band_px=band_px, band_as_edit=band_as_edit, band_alpha=band_alpha
        )

        fuse_hw = (H, W) if fuse_to is None else fuse_to
        views_bgr_f = maybe_resize_views(views_bgr, fuse_hw, resize_mode=args.fuse_resize_mode)
        mask = get_cached_mask(fuse_hw[0], fuse_hw[1], N)
        out_bgr = create_3d_img(mask, views_bgr_f)

        out_dir = os.path.dirname(args.out)
        os.makedirs(out_dir, exist_ok=True) if out_dir else None
        cv2.imwrite(args.out, out_bgr)

        stem = os.path.splitext(os.path.basename(args.image))[0].replace("frame_", "warp_")
        maybe_save_views(views_bgr, offsets, stem, args.save_views_dir)

        print(f"[Done-Image] 视角数={N} | warp={t_warp:.3f}s | refine={t_refine:.3f}s | 输出={args.out}")
        return

    if is_dir_mode:
        clip_dir = args.pair_dir
        depth_dir = os.path.join(clip_dir, "depth")
        if not os.path.isdir(depth_dir):
            raise FileNotFoundError(f"目录缺少 depth 子目录: {depth_dir}")
        frames = [f for f in os.listdir(clip_dir) if f.lower().startswith("frame_") and f.lower().endswith(VALID_IMG_EXTS)]
        frames.sort(key=natural_key)
        pairs = []
        for f in frames:
            ts = re.sub(r"^frame_", "", os.path.splitext(f)[0])
            dep = os.path.join(depth_dir, f"depth_{ts}.png")
            if os.path.isfile(dep):
                pairs.append((os.path.join(clip_dir, f), dep))
        if not pairs:
            raise RuntimeError("未找到任何(frame_xxx, depth/depth_xxx)对")

        ok_rgb = cv2.imread(pairs[0][0], cv2.IMREAD_COLOR)
        if ok_rgb is None: raise RuntimeError("首帧读取失败")
        H0, W0 = ok_rgb.shape[:2]
        H_warp, W_warp = (H0, W0) if resize_to is None else resize_to
        fuse_hw = (H_warp, W_warp) if fuse_to is None else fuse_to

        meta_map = read_meta_txt(args.meta, pose_convention=args.pose_convention) if args.meta else {}

        dep0 = cv2.imread(pairs[0][1], cv2.IMREAD_UNCHANGED)
        Ds0, _ = decode_depth_array(dep0, size=(H_warp, W_warp), mode=args.depth_mode, scale=args.depth_scale,
                                    color_decode=args.depth_color_decode, far_value=args.far_value)
        if rescale_on:
            Ds0 = rescale_depth_tensor_linear(Ds0, rng=(rmin, rmax), percentiles=(plo, phi))

        ts0 = extract_timestamp_from_frame_path(pairs[0][0])
        meta_K0 = meta_map.get(ts0, (None, None))[0] if (ts0 is not None and ts0 in meta_map) else None
        Ks0 = make_K(H_warp, W_warp, src_hw=(H0, W0), meta_K=meta_K0).to(device)
        Ks0_np = Ks0.squeeze(0).detach().cpu().numpy()

        Dnp0 = Ds0.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        offsets = compute_offsets(Ks0_np, Dnp0)
        N = len(offsets)
        mask = get_cached_mask_torch(fuse_hw[0], fuse_hw[1], N, device, args.slope, args.X, args.koff)

        if args.focus_depth > 0:
            focus_z = float(args.focus_depth)
        else:
            valid = Dnp0[np.isfinite(Dnp0) & (Dnp0 > 1e-6)]
            focus_z = float(np.median(valid) if valid.size > 0 else 1.0)

        # 预计算静态参数
        precomp = _precompute_warp_static(Ks0, focus_z, offsets, device)

        # 自动估计chunk
        chunk_views = args.chunk_views
        if (chunk_views <= 0) and args.autotune_chunk and str(device).startswith("cuda"):
            chunk_views = _estimate_chunk_size(H_warp, W_warp, N, C=3, amp=amp_enabled)

        fps_out = float(args.fps) if args.fps > 0 else 30.0
        writer = None
        writer_kind = ""
        if args.ffmpeg_h264:
            writer = FFmpegH264Writer(args.out, fps_out, (fuse_hw[1], fuse_hw[0]),
                                      qp=args.h264_qp, preset=args.h264_preset,
                                      pix_fmt=args.h264_pix_fmt, close_timeout=args.ffmpeg_close_timeout)
            writer_kind = f"FFmpeg H.264 (qp={writer.qp}, preset={writer.preset}, pix_fmt={writer.pix_fmt})"
            print(f"[Writer] 使用 {writer_kind} -> {writer.out_path}")
        elif args.ffmpeg_lossless:
            writer = FFmpegPipeWriter(args.out, fps_out, (fuse_hw[1], fuse_hw[0]), close_timeout=args.ffmpeg_close_timeout)
            writer_kind = "FFmpeg lossless"
            print(f"[Writer] 使用 {writer_kind} -> {writer.out_path}")
        elif args.lossless:
            writer, out_path, codec = choose_lossless_writer(args.out, fps_out, (fuse_hw[1], fuse_hw[0]))
            writer_kind = f"OpenCV {codec}"
            print(f"[Writer] 使用无损编码: {codec} -> {out_path}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = args.out
            writer = cv2.VideoWriter(out_path, fourcc, float(fps_out), (fuse_hw[1], fuse_hw[0]), isColor=True)
            if not writer.isOpened():
                raise RuntimeError(f"无法创建视频输出: {out_path}")
            writer_kind = "OpenCV mp4v"
        writer_async = AsyncWriter(writer, max_queue=32) if args.async_io else None

        t_total0 = time.time()
        num = 0
        time_warp_sum = 0.0
        time_refine_sum = 0.0
        time_fuse_sum = 0.0

        Ks_const = None
        if (args.manual_K is None) and (not meta_map):
            Ks_const = Ks0

        for i, (rgb_p, dep_p) in enumerate(pairs):
            bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
            dep = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED)
            if bgr is None or dep is None:
                print(f"[Skip] 无法读取: {rgb_p} 或 {dep_p}")
                continue
            Is = to_tensor_rgb_bgr(bgr, H_warp, W_warp, device)
            Ds, _ = decode_depth_array(dep, size=(H_warp, W_warp), mode=args.depth_mode, scale=args.depth_scale,
                                       color_decode=args.depth_color_decode, far_value=args.far_value)
            # 替换原先 CPU rescale 逻辑为：CUDA 优先
            if rescale_on:
                if str(device).startswith("cuda"):
                    Ds = Ds.to(device, non_blocking=True)
                    Ds = rescale_depth_tensor_linear_torch(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
                else:
                    Ds = rescale_depth_tensor_linear(Ds, rng=(rmin, rmax), percentiles=(plo, phi))

            if Ks_const is not None:
                Ks_i = Ks_const
            else:
                if meta_map:
                    ts_i = extract_timestamp_from_frame_path(rgb_p)
                    meta_Ki = meta_map.get(ts_i, (None, None))[0] if (ts_i is not None and ts_i in meta_map) else None
                else:
                    meta_Ki = None
                Ks_i = make_K(H_warp, W_warp, src_hw=(H0, W0), meta_K=meta_Ki).to(device)

            Is = _maybe_to_channels_last(Is, args.channels_last)
            Ds = _maybe_to_channels_last(Ds.to(device), args.channels_last)
            views_bgr, t_warp, t_ref = warp_and_refine_batch(
                Is, Ds, Ks_i, focus_z, offsets, device, splat_cfg, refiner, amp_enabled,
                chunk_size=chunk_views, precomp=precomp, channels_last=args.channels_last,
                sync_timing=args.sync_timing, band_px=band_px, band_as_edit=band_as_edit, band_alpha=band_alpha
            )
            t2 = time.time()
            views_bgr_f = maybe_resize_views(views_bgr, fuse_hw, resize_mode=args.fuse_resize_mode)
            out_bgr = create_3d_img(mask, views_bgr_f)
            t_fuse = time.time() - t2

            if writer_async is not None:
                writer_async.write(out_bgr)
            else:
                writer.write(out_bgr)

            num += 1
            time_warp_sum += t_warp
            time_refine_sum += t_ref
            time_fuse_sum += t_fuse

            stem = os.path.splitext(os.path.basename(rgb_p))[0].replace("frame_", "warp_")
            maybe_save_views(views_bgr, offsets, stem, args.save_views_dir)

            if (i+1) % 10 == 0:
                print(f"[{i+1}/{len(pairs)}] warp={t_warp:.3f}s refine={t_ref:.3f}s fuse={t_fuse:.3f}s | {writer_kind}")

        if writer_async is not None:
            writer_async.release()
        else:
            writer.release()

        t_total = time.time() - t_total0
        if num > 0:
            print(f"[Done-Dir] 帧数={num} 视角数={N} | 平均: warp={time_warp_sum/num:.3f}s refine={time_refine_sum/num:.3f}s fuse={time_fuse_sum/num:.3f}s | 总时长={t_total:.3f}s | 输出={out_path}")
        else:
            print("[Done-Dir] 无有效帧输出")
        return

    if is_video_mode:
        rgb_reader = StreamReader(args.video, is_depth=False)
        dep_reader = StreamReader(args.depth_video, is_depth=True)

        ok_rgb, rgb0 = rgb_reader.read()
        ok_dep, dep0 = dep_reader.read()
        if not ok_rgb or not ok_dep:
            rgb_reader.release(); dep_reader.release()
            raise RuntimeError("无法读取视频/深度首帧")

        H0, W0 = rgb0.shape[:2]
        H_warp, W_warp = (H0, W0) if resize_to is None else resize_to
        fuse_hw = (H_warp, W_warp) if fuse_to is None else fuse_to

        Ks0 = make_K(H_warp, W_warp, src_hw=(H0, W0), meta_K=None).to(device)
        Ks_np = Ks0.squeeze(0).detach().cpu().numpy()

        Ds0, _ = decode_depth_array(dep0, size=(H_warp, W_warp), mode=args.depth_mode, scale=args.depth_scale,
                                    color_decode=args.depth_color_decode, far_value=args.far_value)
        if rescale_on:
            Ds0 = rescale_depth_tensor_linear(Ds0, rng=(rmin, rmax), percentiles=(plo, phi))
        Dnp0 = Ds0.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        offsets = compute_offsets(Ks_np, Dnp0)
        N = len(offsets)
        mask = get_cached_mask_torch(fuse_hw[0], fuse_hw[1], N, device, args.slope, args.X, args.koff)
        # 人脸检测器（仅在自动对焦开启时加载）
        face_cascade = None
        if args.auto_focus_face:
            face_cascade = cv2.CascadeClassifier('/media/a1/16THDD/Zhan/depth_warp_vs/haarcascade_files/haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                print("[Warn] 加载人脸分类器失败：haarcascade_files/haarcascade_frontalface_default.xml，已关闭自动对焦")
                args.auto_focus_face = False
        # 初始对焦深度：若启用人脸自动对焦，则不弹出GUI，用默认值以防首帧未检测到人脸
        if args.auto_focus_face:
            if args.focus_depth > 0:
                focus_z = float(args.focus_depth)  # 作为首帧默认值
            else:
                valid = Dnp0[np.isfinite(Dnp0) & (Dnp0 > 1e-6)]
                focus_z = float(np.median(valid) if valid.size > 0 else 1.0)  # 否则用深度中位数
        else:
            if args.focus_depth > 0:
                focus_z = float(args.focus_depth)
            else:
                try:
                    Dnp_gui = Ds0.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                    focus_z = pick_focus_depth_gui(Dnp_gui, "深度图 - 点击选择对焦深度，ESC/Enter确认")
                except Exception:
                    valid = Dnp0[np.isfinite(Dnp0) & (Dnp0 > 1e-6)]
                    focus_z = float(np.median(valid) if valid.size > 0 else 1.0)

        # 预计算静态参数：自动对焦时禁用预计算（每帧对焦深度变化需要重建dT）
        precomp = None if args.auto_focus_face else _precompute_warp_static(Ks0, focus_z, offsets, device)

        # 自动估计chunk
        chunk_views = args.chunk_views
        if (chunk_views <= 0) and args.autotune_chunk and str(device).startswith("cuda"):
            chunk_views = _estimate_chunk_size(H_warp, W_warp, N, C=3, amp=amp_enabled)

        fps_out = float(args.fps) if args.fps > 0 else (rgb_reader.fps or 30.0)
        writer = None
        writer_kind = ""
        if args.ffmpeg_h264:
            writer = FFmpegH264Writer(args.out, fps_out, (fuse_hw[1], fuse_hw[0]),
                                      qp=args.h264_qp, preset=args.h264_preset,
                                      pix_fmt=args.h264_pix_fmt, codec=args.h264_codec,
                                      close_timeout=args.ffmpeg_close_timeout)
            writer_kind = f"FFmpeg {writer.codec} (qp={writer.qp}, preset={writer.preset}, pix_fmt={writer.pix_fmt})"
            print(f"[Writer] 使用 {writer_kind} -> {writer.out_path}")
        elif args.ffmpeg_lossless:
            writer = FFmpegPipeWriter(args.out, fps_out, (fuse_hw[1], fuse_hw[0]), close_timeout=args.ffmpeg_close_timeout)
            writer_kind = "FFmpeg lossless"
            print(f"[Writer] 使用 {writer_kind} -> {writer.out_path}")
        elif args.lossless:
            writer, out_path, codec = choose_lossless_writer(args.out, fps_out, (fuse_hw[1], fuse_hw[0]))
            writer_kind = f"OpenCV {codec}"
            print(f"[Writer] 使用无损编码: {codec} -> {out_path}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = args.out
            writer = cv2.VideoWriter(out_path, fourcc, fps_out, (fuse_hw[1], fuse_hw[0]), isColor=True)
            if not writer.isOpened():
                rgb_reader.release(); dep_reader.release()
                raise RuntimeError(f"无法创建视频输出: {out_path}")
            writer_kind = "OpenCV mp4v"
        writer_async = AsyncWriter(writer, max_queue=32) if args.async_io else None

        # 保存对焦深度的状态：若本帧无人脸，沿用上一帧（或默认）对焦深度
        focus_state = {"z": focus_z}

        def process_one(rgb_bgr, dep_img, frame_idx=None):
            # 先准备深度（CPU），以便在自动对焦时用CPU侧的深度图采样人脸中心点深度
            Ds_cpu, _ = decode_depth_array(dep_img, size=(H_warp, W_warp), mode=args.depth_mode,
                                           scale=args.depth_scale, color_decode=args.depth_color_decode,
                                           far_value=args.far_value)

            # 人脸自动对焦：从所有人脸中选择“最近”的（以人脸中心点的深度值判定）
            local_focus = focus_state["z"]
            if args.auto_focus_face and (face_cascade is not None):
                # 为对齐深度分辨率，将当前RGB帧缩放到与深度一致的 H_warp,W_warp 做检测
                bgr_small = cv2.resize(rgb_bgr, (W_warp, H_warp), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

                # 可按需要调整以下参数以权衡速度与精度
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    minSize=(max(24, W_warp // 20), max(24, H_warp // 20))
                )

                # 为保证与warp使用的深度标定一致，这里在CPU侧也应用相同的线性重标定（若开启）
                if rescale_on:
                    D_focus_cpu = rescale_depth_tensor_linear(Ds_cpu.clone(), rng=(rmin, rmax), percentiles=(plo, phi))
                else:
                    D_focus_cpu = Ds_cpu
                depth_np = D_focus_cpu.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

                best_z = None
                for (x, y, w, h) in faces:
                    cx = x + w // 2
                    cy = y + h // 2
                    if 0 <= cx < W_warp and 0 <= cy < H_warp:
                        z = float(depth_np[cy, cx])
                        if not np.isfinite(z) or z <= 1e-6:
                            continue
                        if best_z is None:
                            best_z = z
                        else:
                            if args.far_value == "smaller":
                                # 若far_value=smaller，表示“更远=更小”，则最近者应取更大的深度
                                if z > best_z: best_z = z
                            else:
                                # 默认far_value=larger：更远=更大，则最近者取更小的深度
                                if z < best_z: best_z = z

                # 若检测到人脸则更新本帧对焦深度；若未检测到，则沿用上一帧（focus_state["z"]）
                if best_z is not None:
                    local_focus = best_z
                    focus_state["z"] = best_z

            # 构建输入张量（RGB）
            Is = to_tensor_rgb_bgr(rgb_bgr, H_warp, W_warp, device)

            # 将深度送往设备并做（GPU/CPU）重标定，供warp与refine使用
            if rescale_on:
                if str(device).startswith("cuda"):
                    Ds = Ds_cpu.to(device, non_blocking=True)
                    Ds = rescale_depth_tensor_linear_torch(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
                else:
                    Ds = rescale_depth_tensor_linear(Ds_cpu, rng=(rmin, rmax), percentiles=(plo, phi))
            else:
                Ds = Ds_cpu.to(device, non_blocking=True)

            Is_ = _maybe_to_channels_last(Is, args.channels_last)
            Ds_ = _maybe_to_channels_last(Ds, args.channels_last)

            # 自动对焦时 precomp 必须为 None（每帧根据 local_focus 重建 dT）
            precomp_d = None if args.auto_focus_face else precomp

            views_bgr, t_warp, t_ref = warp_and_refine_batch(
                Is_, Ds_, Ks0, local_focus, offsets, device, splat_cfg, refiner, amp_enabled,
                chunk_size=chunk_views, precomp=precomp_d, channels_last=args.channels_last,
                sync_timing=args.sync_timing, band_px=band_px, band_as_edit=band_as_edit, band_alpha=band_alpha
            )

            t2 = time.time()
            views_bgr_f = maybe_resize_views(views_bgr, fuse_hw, resize_mode=args.fuse_resize_mode)
            out_bgr = create_3d_img(mask, views_bgr_f)
            t_fuse = time.time() - t2

            if args.save_views_dir is not None:
                stem = f"warp_frame{(0 if frame_idx is None else frame_idx):06d}"
                maybe_save_views(views_bgr, offsets, stem, args.save_views_dir)

            return out_bgr, t_warp, t_ref, t_fuse

        t_total0 = time.time()
        num = 0
        time_warp_sum = 0.0
        time_refine_sum = 0.0
        time_fuse_sum = 0.0

        out_bgr0, tw, tr, tf = process_one(rgb0, dep0, frame_idx=0)
        if isinstance(writer, (FFmpegPipeWriter, FFmpegH264Writer)):
            writer.write(out_bgr0)
            out_path = writer.out_path
        else:
            writer.write(out_bgr0)
        num += 1; time_warp_sum += tw; time_refine_sum += tr; time_fuse_sum += tf
        print(f"[1st] warp={tw:.3f}s refine={tr:.3f}s fuse={tf:.3f}s | {writer_kind}")

        # 预取剩余帧（从 idx=1 开始，因为首帧 idx=0 已处理）
        prefetch_n = 16 if args.async_io else 0
        if args.async_io and prefetch_n > 0:
            iterable = iter_video_pairs_prefetch(rgb_reader, dep_reader, start_idx=1, prefetch=prefetch_n)
            for idx, rgbn, depn in iterable:
                out_bgr, tw, tr, tf = process_one(rgbn, depn, frame_idx=idx)
                if writer_async is not None:
                    writer_async.write(out_bgr)
                else:
                    writer.write(out_bgr)
                num += 1;
                time_warp_sum += tw;
                time_refine_sum += tr;
                time_fuse_sum += tf
                if (num % 10) == 0:
                    print(f"[{num}] warp={tw:.3f}s refine={tr:.3f}s fuse={tf:.3f}s | {writer_kind}")
        else:
            while True:
                ok_rgb, rgbn = rgb_reader.read()
                ok_dep, depn = dep_reader.read()
                if not ok_rgb or not ok_dep: break
                out_bgr, tw, tr, tf = process_one(rgbn, depn, frame_idx=num)
                if writer_async is not None:
                    writer_async.write(out_bgr)
                else:
                    writer.write(out_bgr)
                num += 1;
                time_warp_sum += tw;
                time_refine_sum += tr;
                time_fuse_sum += tf
                if (num % 10) == 0:
                    print(f"[{num}] warp={tw:.3f}s refine={tr:.3f}s fuse={tf:.3f}s | {writer_kind}")

        if writer_async is not None:
            writer_async.release()
        else:
            writer.release()
        rgb_reader.release(); dep_reader.release()
        t_total = time.time() - t_total0
        if num > 0:
            print(f"[Done-Video] 帧数={num} 视角数={N} | 平均: warp={time_warp_sum/num:.3f}s refine={time_refine_sum/num:.3f}s fuse={time_fuse_sum/num:.3f}s | 总时长={t_total:.3f}s | 输出={out_path}")
        else:
            print("[Done-Video] 无有效帧输出")
        return

if __name__ == "__main__":
    main()
# python -m depth_warp_vs.main --video /media/a1/16THDD/Zhan/left_eye.mp4 --depth_video /media/a1/16THDD/Zhan/111_depth.mp4 --refiner_type MGMI --refiner_ckpt ./checkpoints/MGMI/2025100804/refiner_best.pth --out /media/a1/16THDD/Zhan/out_3d.mp4 --num_per_side 1 --max_disp_px 25 --amp --gpu 1 --manual_K 1402.1,1402.1,968.77,506.154 --focus_depth 5.9 --fuse_size 1080,1920 --chunk_views 0 --channels_last --compile --async_io --ffmpeg_h264 --h264_codec=h264_nvenc --band_px 2
