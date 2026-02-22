# depth_warp_vs/realtime.py
import os, cv2, time, argparse, math, sys, warnings, re
import numpy as np
import torch
import torch.nn.functional as F

try:
    from depth_warp_vs.models.splatting.softmax_splat import softmax_splat
    from depth_warp_vs.models.refiner import build_refiner
except Exception:
    from models.splatting.softmax_splat import softmax_splat
    from depth_warp_vs.models.refiner import build_refiner

# ---------------------- 计时工具 ----------------------

class StepTimer:
    def __init__(self):
        self.times = {}
        self._stack = []

    def tic(self, name: str):
        self._stack.append((name, time.time()))

    def toc(self, name: str = None):
        if not self._stack:
            return 0.0
        nm, t0 = self._stack.pop()
        if name is None:
            name = nm
        dt = time.time() - t0
        self.times[name] = self.times.get(name, 0.0) + dt
        return dt

    def add(self, name: str, dt: float):
        self.times[name] = self.times.get(name, 0.0) + float(dt)

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.times.get(name, default))

    def reset(self):
        self.times.clear()
        self._stack.clear()

def _sync_cuda(dev):
    try:
        if torch.cuda.is_available():
            if isinstance(dev, str):
                if dev.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(dev))
            elif isinstance(dev, torch.device):
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            elif isinstance(dev, int):
                torch.cuda.synchronize(dev)
    except Exception:
        pass

# ---------------------- 基础与工具函数 ----------------------

_IDX_BGR_CACHE = {}
def _get_idx_bgr(device):
    key = str(device)
    t = _IDX_BGR_CACHE.get(key, None)
    if t is None:
        t = torch.tensor([2, 1, 0], device=device)
        _IDX_BGR_CACHE[key] = t
    return t

VALID_VID_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv")

def is_video_file(path: str) -> bool:
    return os.path.isfile(path) and (os.path.splitext(path)[1].lower() in VALID_VID_EXTS)

def parse_manual_K(s: str):
    vals = [float(x) for x in s.replace(",", " ").split() if x]
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

def get_mask(rows, cols, theta, N, X=4.66666, koff=5):
    y, x = np.mgrid[0:rows, 0:cols*3].astype(np.float32)
    r = (np.mod((x + koff - 3.0*y*np.tan(np.float32(theta))), np.float32(X))) * (np.float32(N)/np.float32(X))
    pattern = np.floor(r)
    mask = np.zeros((rows, cols, 3), dtype=np.int32)
    mask[:, :, 0] = pattern[:, 0::3].astype(np.int32)
    mask[:, :, 1] = pattern[:, 1::3].astype(np.int32)
    mask[:, :, 2] = pattern[:, 2::3].astype(np.int32)
    return mask

def create_3d_img(mask, views):
    """
    融合函数（GPU优先）：
    - views: torch.Tensor [N,H,W,3] (BGR, uint8，device=CPU/CUDA) 或 List[np.uint8 HxWx3 BGR]
    - mask:  numpy(int32) HxWx3 或 torch(long) HxWx3（建议与views同device、long，避免拷贝）
    返回: numpy.uint8 HxWx3 BGR
    """
    if isinstance(views, torch.Tensor):
        assert views.dim() == 4 and views.size(-1) == 3, "views 需为 (N,H,W,3)"
        N, H, W, _ = views.shape
        device = views.device
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
    # Numpy回退
    views_np = np.stack(views, axis=0)  # (N,H,W,3)
    N, H, W, _ = views_np.shape
    out = np.empty((H, W, 3), dtype=np.uint8)
    m = np.clip(mask, 0, N-1).astype(np.int64)
    for c in range(3):
        out[..., c] = np.take_along_axis(views_np[..., c], m[..., c][None, ...], axis=0)[0]
    return out

def rescale_depth_tensor_linear_torch(Ds: torch.Tensor, rng=(2.0, 10.0), percentiles=(1.0, 99.0)) -> torch.Tensor:
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
                          chunk_size=0, precomp=None, channels_last=False):
    """
    返回:
    - views_stacked: torch.uint8 (N,H,W,3) BGR
    - t_warp_total: 仅包含 softmax_splat 的累计时间（秒，已同步）
    - t_refine_total: 仅包含 refiner 的累计时间（秒，已同步）
    """
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

    chunk = int(N if (chunk_size is None or chunk_size <= 0) else max(1, min(chunk_size, N)))
    chunks = []
    t_warp_total = 0.0
    t_refine_total = 0.0

    use_cuda = isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()
    dev_obj = torch.device(device) if isinstance(device, str) else device

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Is_c = Is.expand(e - s, -1, -1, -1)
        Ds_c = Ds.expand(e - s, -1, -1, -1)
        Ks_c = Ks_b[s:e]
        Kt_c = Kt_b[s:e]
        dT_c = dT_b[s:e]

        if use_cuda:
            with torch.cuda.device(dev_obj):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", enabled=amp_enabled):
                        Iw, V = softmax_splat(
                            Is_c, Ds_c, Ks_c, Kt_c, dT_c,
                            temperature=float(splat_cfg["temperature"]),
                            normalize=True,
                            occlusion=splat_cfg["occlusion"],
                            hard_z_epsilon=float(splat_cfg["hard_z_epsilon"])
                        )
                ev1.record()
                torch.cuda.synchronize(dev_obj)
                t_warp_total += ev0.elapsed_time(ev1) / 1000.0
        else:
            t0 = time.time()
            with torch.inference_mode():
                Iw, V = softmax_splat(
                    Is_c, Ds_c, Ks_c, Kt_c, dT_c,
                    temperature=float(splat_cfg["temperature"]),
                    normalize=True,
                    occlusion=splat_cfg["occlusion"],
                    hard_z_epsilon=float(splat_cfg["hard_z_epsilon"])
                )
            t_warp_total += (time.time() - t0)

        hole = (V <= 1e-8).to(Iw.dtype)
        x = torch.cat([Iw, hole], dim=1)
        x = _maybe_to_channels_last(x, channels_last)

        if use_cuda:
            with torch.cuda.device(dev_obj):
                ev2 = torch.cuda.Event(enable_timing=True)
                ev3 = torch.cuda.Event(enable_timing=True)
                ev2.record()
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", enabled=amp_enabled):
                        Ipred = refiner_model(x)
                ev3.record()
                torch.cuda.synchronize(dev_obj)
                t_refine_total += ev2.elapsed_time(ev3) / 1000.0
        else:
            t1 = time.time()
            with torch.inference_mode():
                Ipred = refiner_model(x)
            t_refine_total += (time.time() - t1)

        if Ipred.dtype != Iw.dtype:
            Ipred = Ipred.to(Iw.dtype)
        hole = hole.to(Iw.dtype)

        Ifinal = torch.lerp(Iw, Ipred, hole).clamp_(0, 1)
        Ifinal_u8 = (Ifinal * 255.0).to(torch.uint8)
        views_rgb = Ifinal_u8.permute(0, 2, 3, 1).contiguous()
        idx_bgr = _get_idx_bgr(views_rgb.device)
        views_bgr = views_rgb.index_select(-1, idx_bgr)
        chunks.append(views_bgr)
        del Iw, V, hole, x, Ipred, Ifinal, Ifinal_u8, views_rgb, views_bgr

    views_stacked = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    return views_stacked, t_warp_total, t_refine_total

def maybe_resize_views(views, fuse_size, resize_mode="stretch"):
    if fuse_size is None:
        return views
    Hf, Wf = fuse_size
    resize_mode = (resize_mode or "stretch").lower()
    if isinstance(views, torch.Tensor):
        assert views.dim() == 4 and views.size(-1) == 3, "views 需为 (N,H,W,3)"
        N, H, W, C = views.shape
        if (H == Hf) and (W == Wf):
            return views
        if resize_mode == "keep_ar_pad":
            s = min(float(Hf) / max(1.0, float(H)), float(Wf) / max(1.0, float(W)))
            H1 = max(1, int(round(H * s))); W1 = max(1, int(round(W * s)))
            x = views.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            y = F.interpolate(x, size=(H1, W1), mode="bilinear", align_corners=False)
            y8 = (y.clamp(0, 1) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            canvas = torch.zeros((N, Hf, Wf, 3), dtype=torch.uint8, device=views.device)
            top = (Hf - H1) // 2; left = (Wf - W1) // 2
            canvas[:, top:top+H1, left:left+W1, :] = y8
            return canvas
        else:
            x = views.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            y = F.interpolate(x, size=(Hf, Wf), mode="bilinear", align_corners=False)
            y8 = (y.clamp(0, 1) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            return y8
    # Numpy 回退
    out = []
    for v in views:
        H, W = v.shape[:2]
        if (H == Hf) and (W == Wf):
            out.append(v); continue
        if resize_mode == "keep_ar_pad":
            s = min(float(Hf) / max(1.0, float(H)), float(Wf) / max(1.0, float(W)))
            H1 = max(1, int(round(H * s))); W1 = max(1, int(round(W * s)))
            v1 = cv2.resize(v, (W1, H1), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((Hf, Wf, 3), dtype=np.uint8)
            top = (Hf - H1) // 2; left = (Wf - W1) // 2
            canvas[top:top+H1, left:left+W1, :] = v1
            out.append(canvas)
        else:
            out.append(cv2.resize(v, (Wf, Hf), interpolation=cv2.INTER_LINEAR))
    return out

def _precompute_warp_static(Ks: torch.Tensor, focus_z: float, offsets, device: str):
    N = len(offsets)
    Ks_b = Ks.expand(N, -1, -1)
    Kt_b = Ks_b
    dT_list = [build_convergent_camera_motion(tx=o, focus_z=focus_z) for o in offsets]
    dT_b = torch.from_numpy(np.stack(dT_list, axis=0)).float().to(device)
    return {"Ks_b": Ks_b, "Kt_b": Kt_b, "dT_b": dT_b, "N": N}

def _estimate_chunk_size(H, W, N, C=3, amp=True):
    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
    except Exception:
        return N
    bytes_per = 2 if amp else 4
    per_view_bytes = (C*H*W + H*W) * bytes_per * 3.0
    if per_view_bytes <= 0:
        return N
    max_views = int(max(1, min(N, (free_mem * 0.5) // per_view_bytes)))
    return max_views if max_views > 0 else 1

def to_tensor_rgb_bgr(bgr, H_warp, W_warp, device, pin=True):
    if (bgr.shape[0] != H_warp) or (bgr.shape[1] != W_warp):
        bgr = cv2.resize(bgr, (W_warp, H_warp), interpolation=cv2.INTER_AREA)
    # 使用通道翻转避免额外拷贝
    rgb = bgr[..., ::-1].copy()
    ten = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if pin and ten.device.type == "cpu" and torch.cuda.is_available():
        ten = ten.pin_memory()
        return ten.to(device, non_blocking=True)
    return ten.to(device)

# ---------------------- 多GPU辅助 ----------------------

def parse_gpu_arg(gpu_arg: str):
    """
    返回设备列表（字符串），如 ['cuda:0','cuda:1'] 或 ['cpu']。
    输入示例：
    - "0" -> ['cuda:0']
    - "0,1,2" -> ['cuda:0','cuda:1','cuda:2']
    - "-1" 或 "auto" -> 全部可见GPU；若无CUDA则['cpu']
    - "" 或 None -> 若有CUDA则['cuda:0']否则['cpu']
    """
    has_cuda = torch.cuda.is_available()
    if not gpu_arg:
        return ['cuda:0'] if has_cuda else ['cpu']
    s = str(gpu_arg).strip().lower()
    if (s == "-1") or (s == "auto"):
        if has_cuda:
            n = torch.cuda.device_count()
            return [f"cuda:{i}" for i in range(n)] if n > 0 else ['cpu']
        else:
            return ['cpu']
    # 逗号/空格/分号分隔
    parts = [p for p in re.split(r"[,\s;]+", s) if p != ""]
    ids = []
    for p in parts:
        try:
            ids.append(int(p))
        except:
            # 允许 "cuda:0" 形式
            if p.startswith("cuda:"):
                try:
                    ids.append(int(p.split(":")[1]))
                except:
                    pass
    if len(ids) == 0:
        return ['cuda:0'] if has_cuda else ['cpu']
    if has_cuda:
        return [f"cuda:{i}" for i in ids]
    else:
        return ['cpu']

def split_offsets_across_devices(offsets, num_devices: int):
    """
    将offsets按设备连续切分，返回 [(start,end), ...] 与各自的子列表。
    """
    N = len(offsets)
    if num_devices <= 1 or N == 0:
        return [(0, N)], [offsets]
    # 近似均分（连续块）
    base = N // num_devices
    rem = N % num_devices
    idx = 0
    ranges = []
    subs = []
    for d in range(num_devices):
        k = base + (1 if d < rem else 0)
        s, e = idx, idx + k
        ranges.append((s, e))
        subs.append(offsets[s:e])
        idx = e
    return ranges, subs

# ---------------------- 实时主流程 ----------------------

def main():
    ap = argparse.ArgumentParser("实时裸眼3D（模拟UDP）：RGB视频+深度视频逐帧处理->窗口显示（左上角FPS，q退出，循环播放）")
    # 输入（模拟实时：来自两个视频）
    ap.add_argument("--video", required=True, help="RGB视频路径（用于模拟实时RGB流）")
    ap.add_argument("--depth_video", required=True, help="深度视频路径（用于模拟实时深度流，彩色或灰度均可）")

    # warp与输出尺寸
    ap.add_argument("--img_size", default="0,0", help="warp尺寸 H,W (0,0=跟随输入视频分辨率)")
    ap.add_argument("--fuse_size", default="1080,1920", help="合成裸眼3D输出尺寸 H,W (0,0=跟随warp尺寸)")
    ap.add_argument("--fuse_resize_mode", default="stretch", choices=["stretch", "keep_ar_pad"], help="合成前缩放模式")

    # 相机内参
    ap.add_argument("--manual_K", default=None, help="手动相机内参 'fx,fy,cx,cy'")
    ap.add_argument("--K_units", default="auto", choices=["auto","pixel","normalized"])

    # 深度读取与缩放
    ap.add_argument("--depth_mode", default="auto", choices=["auto","metric","normalized"])
    ap.add_argument("--depth_scale", type=float, default=10.0)
    ap.add_argument("--depth_color_decode", default="auto", choices=["auto","pca","luma","rgb24","r","g","b"])
    ap.add_argument("--far_value", default="larger", choices=["larger","smaller"], help="深度值更大是否代表更远")
    ap.add_argument("--rescale_depth", default="linear", choices=["none","linear"], help="warp前线性重标定深度（GPU上）")
    ap.add_argument("--rescale_range", default="2,10", help="重标定目标区间 [dmin,dmax]")
    ap.add_argument("--rescale_percentiles", default="1,99", help="用于线性映射的分位点")

    # 视角与视差
    ap.add_argument("--num_per_side", type=int, default=4, help="左右各视角数（含中心，共2N+1）")
    ap.add_argument("--spacing", default="linear", choices=["linear","cosine"])
    ap.add_argument("--tx_max", type=float, default=0.0, help="最大水平平移（>0则直接使用该值）")
    ap.add_argument("--max_disp_px", type=float, default=48.0, help="若tx_max<=0，根据fx与参考深度换算")
    ap.add_argument("--disp_ref_percentile", type=float, default=0.5, help="参考深度分位点（0~1）")

    # splatting与修复
    ap.add_argument("--temperature", type=float, default=30.0)
    ap.add_argument("--occlusion", default="hard", choices=["hard","soft"])
    ap.add_argument("--hard_z_epsilon", type=float, default=1e-3)
    ap.add_argument("--refiner_type", default="MGMI", choices=["MGMI","InpaintRefiner"])
    ap.add_argument("--refiner_ckpt", required=True, help="补洞模型权重路径（建议EMA）")

    # 裸眼3D掩膜参数
    ap.add_argument("--slope", type=float, default=0.166666, help="mask斜率，theta=atan(slope)")
    ap.add_argument("--X", type=float, default=4.66666, help="光栅周期参数")
    ap.add_argument("--koff", type=float, default=5.0, help="相位偏移参数")

    # 对焦
    ap.add_argument("--focus_depth", type=float, default=0.0, help=">0则固定对焦深度；<=0时用首帧深度中位数作为对焦")

    # 加速/设备
    ap.add_argument("--amp", action="store_true", help="启用AMP半精度（需CUDA）")
    ap.add_argument("--channels_last", action="store_true", help="使用channels_last加速Conv")
    ap.add_argument("--compile", action="store_true", help="使用torch.compile加速Refiner（PyTorch 2.0+）")
    ap.add_argument("--tf32", action="store_true", default=True, help="启用TF32（Ampere+）")
    # 改造：--gpu 支持多卡输入
    ap.add_argument("--gpu", type=str, default="-1", help="GPU选择：单卡如'0'，多卡如'0,1,2,3'，-1或'auto'为自动；CPU则传'cpu'")
    ap.add_argument("--multi_gpu_mode", default="views", choices=["views","off"], help="多GPU模式：views=按视角切分；off=禁用多GPU")
    ap.add_argument("--gather_to", type=int, default=None, help="汇总与融合所在GPU编号（默认为列表第一个）")
    ap.add_argument("--chunk_views", type=int, default=0, help="每次处理的视角数（分块），0=不分块")
    ap.add_argument("--autotune_chunk", action="store_true", help="自动估计 chunk_views（按显存）")

    ap.add_argument("--window", default="DWvs Realtime", help="窗口标题")
    ap.add_argument("--fps_avg_win", type=int, default=30, help="FPS滑动平均窗口大小")

    args = ap.parse_args()

    # 解析设备与加速开关（多GPU）
    devices = parse_gpu_arg(args.gpu)
    use_cuda = any([d.startswith("cuda") for d in devices]) and torch.cuda.is_available()
    multi_gpu_on = (args.multi_gpu_mode == "views") and use_cuda and (len(devices) > 1)

    if use_cuda and len(devices) >= 1:
        torch.cuda.set_device(int(devices[0].split(":")[1]))
    device_primary = devices[0] if use_cuda else "cpu"
    amp_enabled = bool(args.amp) and use_cuda
    torch.set_grad_enabled(False)
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # 打开视频（模拟实时）
    if not is_video_file(args.video):
        raise FileNotFoundError(f"RGB视频不存在或格式不支持: {args.video}")
    if not is_video_file(args.depth_video):
        raise FileNotFoundError(f"深度视频不存在或格式不支持: {args.depth_video}")

    cap_rgb = cv2.VideoCapture(args.video)
    cap_dep = cv2.VideoCapture(args.depth_video)
    if (not cap_rgb.isOpened()) or (not cap_dep.isOpened()):
        raise RuntimeError("无法打开视频输入")

    # 读取首帧
    t_init_timer = StepTimer()
    t_init_timer.tic("read_init")
    ok_rgb, rgb0 = cap_rgb.read()
    ok_dep, dep0 = cap_dep.read()
    t_init_timer.toc("read_init")
    if not ok_rgb or not ok_dep:
        cap_rgb.release(); cap_dep.release()
        raise RuntimeError("无法读取首帧")

    H0, W0 = rgb0.shape[:2]
    Hs, Ws = [int(x) for x in args.img_size.replace(",", " ").split()]
    H_warp, W_warp = (H0, W0) if (Hs==0 or Ws==0) else (Hs, Ws)
    Hf, Wf = [int(x) for x in args.fuse_size.replace(",", " ").split()]
    fuse_hw = (H_warp, W_warp) if (Hf==0 or Wf==0) else (Hf, Wf)

    # 内参
    def make_K(H, W, src_hw=None):
        if args.manual_K is not None:
            K_manual = parse_manual_K(args.manual_K)
            if args.K_units == "normalized" or (args.K_units=="auto" and is_normalized_K(K_manual)):
                Kpx = scale_intrinsics(K_manual, H, W)
            elif src_hw is not None:
                Kpx = resize_intrinsics(K_manual, src_hw, (H, W))
            else:
                Kpx = K_manual.astype(np.float32)
        else:
            Kpx = default_K(H, W)
        return torch.from_numpy(Kpx).float().unsqueeze(0)

    Ks0_cpu = make_K(H_warp, W_warp, src_hw=(H0, W0))
    Ks_np = Ks0_cpu.squeeze(0).detach().cpu().numpy()

    # 深度首帧 -> 计算offsets与初始对焦
    splat_cfg = {
        "temperature": float(args.temperature),
        "occlusion": args.occlusion,
        "hard_z_epsilon": float(args.hard_z_epsilon)
    }
    rmin, rmax = [float(x) for x in args.rescale_range.replace(",", " ").split()]
    plo, phi = [float(x) for x in args.rescale_percentiles.replace(",", " ").split()]
    rescale_on = (args.rescale_depth == "linear")

    Ds0_cpu, _ = decode_depth_array(dep0, size=(H_warp, W_warp), mode=args.depth_mode,
                                    scale=args.depth_scale, color_decode=args.depth_color_decode,
                                    far_value=args.far_value)
    if rescale_on and use_cuda:
        # H2D
        t0 = time.time()
        Ds0_in = Ds0_cpu.clone().to(device_primary, non_blocking=True)
        _sync_cuda(device_primary)
        dep_h2d_dt = time.time() - t0
        # GPU重标定计时（CUDA events）
        with torch.cuda.device(torch.device(device_primary)):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            Ds0 = rescale_depth_tensor_linear_torch(Ds0_in, rng=(rmin, rmax), percentiles=(plo, phi))
            e1.record()
            torch.cuda.synchronize(torch.device(device_primary))
            dep_rescale_dt = e0.elapsed_time(e1) / 1000.0
    elif rescale_on and (not use_cuda):
        t0 = time.time()
        Ds0 = rescale_depth_tensor_linear_torch(Ds0_cpu.clone(), rng=(rmin, rmax), percentiles=(plo, phi))
        dep_h2d_dt = 0.0
        dep_rescale_dt = time.time() - t0
    else:
        if use_cuda:
            t0 = time.time()
            Ds0 = Ds0_cpu.clone().to(device_primary, non_blocking=True)
            _sync_cuda(device_primary)
            dep_h2d_dt = time.time() - t0
        else:
            Ds0 = Ds0_cpu.to("cpu")
            dep_h2d_dt = 0.0
        dep_rescale_dt = 0.0

    Dnp0 = Ds0.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    valid0 = Dnp0[np.isfinite(Dnp0) & (Dnp0 > 1e-6)]
    if valid0.size < 16:
        z_ref = float(np.maximum(Dnp0.mean(), 1e-3))
    else:
        q = float(np.clip(args.disp_ref_percentile, 0.0, 1.0))
        z_ref = float(np.quantile(valid0, q))
        z_ref = max(z_ref, 1e-6)

    if args.tx_max > 0.0:
        tx_max = float(args.tx_max)
    else:
        fx_px = float(Ks_np[0,0])
        tx_max = compute_tx_from_disp(fx_px, z_ref, args.max_disp_px)

    offsets = build_offsets(tx_max, args.num_per_side, args.spacing)
    N = len(offsets)

    # 对焦深度：>0使用指定，否则用首帧中位数
    if args.focus_depth > 0:
        focus_z = float(args.focus_depth)
    else:
        focus_z = float(np.median(valid0) if valid0.size > 0 else 1.0)

    # Refiner 配置
    refiner_cfg = {
        "model": {
            "refiner": {
                "type": args.refiner_type,
                "in_ch": 4,
                "out_ch": 3,
                "base_ch": 24 if args.refiner_type.lower()=="mgmi" else 48,
                "width_mult": 1.25 if args.refiner_type.lower()=="mgmi" else 1.0,
                "depth": 5 if args.refiner_type.lower()=="inpaintrefiner" else None,
                "aspp_rates": [1,2,4,8],
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

    ckpt = args.refiner_ckpt
    if not os.path.isfile(ckpt):
        cap_rgb.release(); cap_dep.release()
        raise FileNotFoundError(f"Refiner权重不存在: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    clean = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }

    # 单/多卡模型加载
    models = {}
    Ks_per_dev = {}
    precomp_per_dev = {}
    chunk_views_per_dev = {}
    devices_use = devices if (multi_gpu_on) else [device_primary]

    for idx, dev in enumerate(devices_use):
        refiner = build_refiner(refiner_cfg).to(dev).eval()
        refiner.load_state_dict(clean, strict=False)
        if args.channels_last:
            refiner = refiner.to(memory_format=torch.channels_last)
        if args.compile and hasattr(torch, "compile") and dev.startswith("cuda"):
            try:
                refiner = torch.compile(refiner)
            except Exception as e:
                warnings.warn(f"torch.compile 失败（{dev}），回退普通模式: {e}")
        models[dev] = refiner
        Ks_per_dev[dev] = Ks0_cpu.clone().to(dev)
    # mask在汇总设备上缓存
    gather_dev = (f"cuda:{args.gather_to}" if (args.gather_to is not None and use_cuda) else devices_use[0])
    mask_torch = get_cached_mask_torch(fuse_hw[0], fuse_hw[1], N, gather_dev if use_cuda else "cpu",
                                       args.slope, args.X, args.koff)

    # offsets按设备切分并预计算（静态）
    ranges, offsets_splits = split_offsets_across_devices(offsets, len(devices_use))
    for dev, offs in zip(devices_use, offsets_splits):
        precomp_per_dev[dev] = _precompute_warp_static(Ks_per_dev[dev], focus_z, offs, dev)

    # 自动估计chunk（按设备）
    user_chunk_views = int(args.chunk_views)
    for dev, offs in zip(devices_use, offsets_splits):
        if (user_chunk_views <= 0) and args.autotune_chunk and dev.startswith("cuda"):
            chunk_views_per_dev[dev] = _estimate_chunk_size(H_warp, W_warp, len(offs), C=3, amp=amp_enabled)
        else:
            chunk_views_per_dev[dev] = (user_chunk_views if user_chunk_views > 0 else 0)

    print(f"[Init] 设备={devices_use} | 汇总到={gather_dev} | AMP={amp_enabled} channels_last={args.channels_last} | 视角总数={N} | warp={H_warp}x{W_warp} | fuse={fuse_hw[0]}x{fuse_hw[1]}")
    if (not multi_gpu_on) and use_cuda:
        print("[Init] 单GPU模式（原始行为）")
    elif multi_gpu_on:
        print(f"[Init] 多GPU视角并行：offsets按{len(devices_use)}卡切分为 {[len(x) for x in offsets_splits]}")

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    # FPS统计
    t_prev = time.time()
    dt_hist = []
    fps_win = max(1, int(args.fps_avg_win))

    def read_looping(cap, fallback_frame):
        ok, frm = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frm = cap.read()
            if not ok:
                frm = fallback_frame
        return frm

    # 主循环
    frame_idx = 0
    try:
        while True:
            timer = StepTimer()
            t_frame_start = time.time()

            # 读取帧
            timer.tic("read")
            rgb_bgr = read_looping(cap_rgb, rgb0)
            dep_img = read_looping(cap_dep, dep0)
            timer.toc("read")

            # 预处理：RGB -> CPU tensor（pin）
            timer.tic("rgb2tensor")
            Is_cpu = to_tensor_rgb_bgr(rgb_bgr, H_warp, W_warp, device="cpu", pin=True)  # [1,3,H,W] float
            timer.toc("rgb2tensor")

            # 深度 decode 到CPU
            timer.tic("dep_decode")
            Ds_cpu, _ = decode_depth_array(dep_img, size=(H_warp, W_warp), mode=args.depth_mode,
                                           scale=args.depth_scale, color_decode=args.depth_color_decode,
                                           far_value=args.far_value)
            timer.toc("dep_decode")

            # 深度重标定/拷贝至首设备
            if rescale_on and use_cuda:
                # H2D
                t0 = time.time()
                Ds_in = Ds_cpu.to(device_primary, non_blocking=True)
                _sync_cuda(device_primary)
                timer.add("dep_h2d", time.time() - t0)
                # GPU rescale
                with torch.cuda.device(torch.device(device_primary)):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    Ds_primary = rescale_depth_tensor_linear_torch(Ds_in, rng=(rmin, rmax), percentiles=(plo, phi))
                    e1.record()
                    torch.cuda.synchronize(torch.device(device_primary))
                    timer.add("dep_rescale", e0.elapsed_time(e1) / 1000.0)
            elif rescale_on and (not use_cuda):
                t0 = time.time()
                Ds_primary = rescale_depth_tensor_linear_torch(Ds_cpu.clone(), rng=(rmin, rmax), percentiles=(plo, phi))
                timer.add("dep_rescale", time.time() - t0)
            else:
                if use_cuda:
                    t0 = time.time()
                    Ds_primary = Ds_cpu.to(device_primary, non_blocking=True)
                    _sync_cuda(device_primary)
                    timer.add("dep_h2d", time.time() - t0)
                else:
                    Ds_primary = Ds_cpu.to("cpu")

            # 统计计算时间
            t_warp_total = 0.0
            t_ref_total = 0.0
            t_resize_total = 0.0
            t_bcast_total = 0.0

            views_parts = []

            # 遍历各设备：传输->计算->尺寸调整
            for (dev, offs) in zip(devices_use, offsets_splits):
                if len(offs) == 0:
                    views_parts.append((dev, torch.empty((0, H_warp, W_warp, 3), dtype=torch.uint8)))
                    continue

                # 传输（计时+同步，确保时间真实）
                tb0 = time.time()
                Is_dev = Is_cpu.to(dev, non_blocking=True)
                if use_cuda:
                    _sync_cuda(dev)
                if dev == device_primary:
                    Ds_dev = Ds_primary
                else:
                    Ds_dev = (Ds_primary.to(dev, non_blocking=True) if use_cuda else Ds_primary.clone())
                    if use_cuda:
                        _sync_cuda(dev)
                t_bcast_total += (time.time() - tb0)

                Is_dev_ = _maybe_to_channels_last(Is_dev, args.channels_last)
                Ds_dev_ = _maybe_to_channels_last(Ds_dev, args.channels_last)

                # 计算（分块）——函数内部已使用CUDA Event并同步，保证时间准确
                views_bgr_dev, t_warp, t_ref = warp_and_refine_batch(
                    Is_dev_, Ds_dev_, Ks_per_dev[dev], focus_z, offs, dev, splat_cfg, models[dev], amp_enabled,
                    chunk_size=chunk_views_per_dev[dev], precomp=precomp_per_dev[dev], channels_last=args.channels_last
                )
                t_warp_total += float(t_warp)
                t_ref_total += float(t_ref)

                # 可选在本设备内先resize以减少跨设备拷贝带宽（计时+同步）
                tr0 = time.time()
                views_bgr_dev = maybe_resize_views(views_bgr_dev, fuse_hw, resize_mode=args.fuse_resize_mode)
                if use_cuda:
                    _sync_cuda(dev)
                t_resize_total += (time.time() - tr0)

                views_parts.append((dev, views_bgr_dev))

            # 汇总到gather_dev并拼接为完整视角序（计时+同步）
            tg0 = time.time()
            views_cat_list = []
            for dev, v in views_parts:
                if v.numel() == 0:
                    continue
                v2 = (v if (dev == gather_dev) else v.to(gather_dev, non_blocking=True))
                views_cat_list.append(v2)
            if len(views_cat_list) > 1:
                views_all = torch.cat(views_cat_list, dim=0)
            elif len(views_cat_list) == 1:
                views_all = views_cat_list[0]
            else:
                views_all = torch.empty((0, fuse_hw[0], fuse_hw[1], 3), dtype=torch.uint8, device=gather_dev)
            if use_cuda:
                _sync_cuda(gather_dev)
            t_gather = time.time() - tg0

            # 融合（在gather_dev上）-> numpy（计时）
            tf0 = time.time()
            out_bgr = create_3d_img(mask_torch, views_all)
            t_fuse = time.time() - tf0

            # FPS
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            dt_hist.append(dt)
            if len(dt_hist) > fps_win:
                dt_hist.pop(0)
            fps = (len(dt_hist) / sum(dt_hist)) if sum(dt_hist) > 1e-6 else 0.0

            # 文本叠加（合计/细分）
            ms_read = timer.get("read", 0.0) * 1000.0
            ms_rgb2ten = timer.get("rgb2tensor", 0.0) * 1000.0
            ms_depdec = timer.get("dep_decode", 0.0) * 1000.0
            ms_deph2d = timer.get("dep_h2d", 0.0) * 1000.0
            ms_depres = timer.get("dep_rescale", 0.0) * 1000.0
            ms_bcast = t_bcast_total * 1000.0
            ms_warp = t_warp_total * 1000.0
            ms_ref = t_ref_total * 1000.0
            ms_resize = t_resize_total * 1000.0
            ms_gather = t_gather * 1000.0
            ms_fuse = t_fuse * 1000.0
            ms_total = (time.time() - t_frame_start) * 1000.0

            txt = (
                f"FPS:{fps:5.1f} | "
                f"read:{ms_read:.1f} rgb2ten:{ms_rgb2ten:.1f} dep_dec:{ms_depdec:.1f} "
                f"dep_h2d:{ms_deph2d:.1f} dep_res:{ms_depres:.1f} bcast:{ms_bcast:.1f} "
                f"warp:{ms_warp:.1f} ref:{ms_ref:.1f} resize:{ms_resize:.1f} "
                f"gather:{ms_gather:.1f} fuse:{ms_fuse:.1f} total:{ms_total:.1f} | "
                f"N={N} {out_bgr.shape[1]}x{out_bgr.shape[0]}"
            )
            cv2.putText(out_bgr, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,255,0), 2, cv2.LINE_AA)

            # 显示（计时可选）
            td0 = time.time()
            cv2.imshow(args.window, out_bgr)
            key = cv2.waitKey(1) & 0xFF
            _ = time.time() - td0  # 如需展示display时间可计入
            if key == ord('q'):
                break

            frame_idx += 1

    finally:
        cap_rgb.release(); cap_dep.release()
        cv2.destroyAllWindows()
        try:
            if use_cuda:
                torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    main()
# 使用示例：
# 单GPU（保持兼容）：--gpu 0
# 多GPU并行（按视角切分）：--gpu 0,1,2,3 --multi_gpu_mode views
# 其余参数如原：
# python -m depth_warp_vs.realtime --video "D:/xx/left_eye.mp4" --depth_video "D:/xx/111_depth.mp4" --refiner_type MGMI --refiner_ckpt ./checkpoints/refiner_latest.pth --num_per_side 1 --max_disp_px 25 --amp --gpu 0,1,2,3 --manual_K 1402.1,1402.1,968.77,506.154 --focus_depth 5.9 --img_size 0,0 --fuse_size 1080,1920 --channels_last --compile --autotune_chunk
# D:\naked-eye 3D\video call\DWvs>python -m depth_warp_vs.realtime --video "D:/naked-eye 3D/video call/DWvs/left_eye.mp4" --depth_video "D:/naked-eye 3D/video call/DWvs/111_depth.mp4" --refiner_type MGMI --refiner_ckpt ./checkpoints/refiner_latest.pth --num_per_side 1 --max_disp_px 25 --amp --gpu 0 --manual_K 1402.1,1402.1,968.77,506.154 --focus_depth 5.9 --img_size 0,0 --fuse_size 1080,1920 --channels_last --compile --chunk_views 0