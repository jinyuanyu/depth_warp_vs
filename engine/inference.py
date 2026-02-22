# depth_warp_vs/engine/inference.py
import os
import cv2
import yaml
import torch
import numpy as np
from models.route_a_model import RouteAModel
from data.cameras.camera import Camera
from data.cameras.pose_utils import invert

def load_image(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    H0, W0 = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    img_t = torch.from_numpy(img).float()/255.0
    img_t = img_t.permute(2,0,1).unsqueeze(0)
    return img_t, (H0, W0)

def load_depth(path, size=None, mode="auto", scale=10.0):
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(path)
    H0, W0 = dep.shape[:2]
    if dep.ndim == 3:
        dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
    dep = dep.astype(np.float32)
    if mode == "metric":
        if dep.dtype == np.uint16 or dep.max() > 255.0:
            dep = dep / 1000.0
        dep = np.clip(dep, 1e-3, 1e6)
    else:
        if dep.max() > 1.0:
            dep = dep / 255.0
        dep = np.clip(dep, 1e-4, 1.0) * float(scale)
    if size is not None:
        dep = cv2.resize(dep, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    dep_t = torch.from_numpy(dep).float().unsqueeze(0).unsqueeze(0)
    return dep_t, (H0, W0)

def parse_concat_image(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    H, W, _ = img.shape
    mid = W // 2
    rgb = img[:, :mid, :]
    dep = img[:, mid:, :]
    H0, W0 = rgb.shape[:2]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
    if size is not None:
        rgb = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        dep = cv2.resize(dep, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    rgb_t = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0)/255.0
    dep = dep.astype(np.float32)
    dep = dep / max(1.0, dep.max())
    dep = np.clip(dep, 1e-3, 1.0) * 10.0
    dep_t = torch.from_numpy(dep).float().unsqueeze(0).unsqueeze(0)
    return rgb_t, dep_t, (H0, W0)

def default_K(B, H, W, device):
    return Camera.make_default(B,H,W, device=device).K

def _resize_intrinsics(K: np.ndarray, src_hw, dst_hw):
    H0, W0 = src_hw
    H1, W1 = dst_hw
    sx = W1 / max(1.0, float(W0))
    sy = H1 / max(1.0, float(H0))
    K = K.copy().astype(np.float32)
    K[0,0] *= sx
    K[1,1] *= sy
    K[0,2] *= sx
    K[1,2] *= sy
    return K

def _is_normalized_K(K: np.ndarray, thresh: float = 4.0) -> bool:
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    return (abs(fx) <= thresh) and (abs(fy) <= thresh) and (abs(cx) <= thresh) and (abs(cy) <= thresh)

def _scale_norm_K(K: np.ndarray, H: int, W: int) -> np.ndarray:
    Kp = K.copy().astype(np.float32)
    Kp[0,0] *= float(W)
    Kp[0,2] *= float(W)
    Kp[1,1] *= float(H)
    Kp[1,2] *= float(H)
    Kp[2,2] = 1.0
    return Kp

def _to_numpy_K(obj):
    # obj can be list of floats, np.array, torch tensor, or None
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().astype(np.float32)
    if isinstance(obj, (list, tuple)):
        vals = [float(x) for x in obj]
    elif isinstance(obj, str):
        s = obj.strip()
        if os.path.isfile(s):
            ext = os.path.splitext(s)[1].lower()
            if ext == ".npy":
                return np.load(s).astype(np.float32)
            elif ext in [".txt", ".csv"]:
                vals = []
                with open(s, "r") as f:
                    for line in f:
                        line = line.strip().replace(",", " ")
                        if not line: continue
                        parts = [p for p in line.split() if p]
                        vals.extend([float(p) for p in parts])
            elif ext == ".json":
                import json
                js = json.load(open(s, "r"))
                if isinstance(js, dict):
                    if all(k in js for k in ["fx","fy","cx","cy"]):
                        fx,fy,cx,cy = js["fx"], js["fy"], js["cx"], js["cy"]
                        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
                        return K
                    elif "K" in js:
                        Karr = np.array(js["K"], dtype=np.float32)
                        return Karr.reshape(3,3)
                raise ValueError(f"Unsupported JSON schema for K: {s}")
            else:
                raise ValueError(f"Unsupported K file extension: {ext}")
        else:
            # assume comma or space separated floats
            s = s.replace(",", " ")
            vals = [float(x) for x in s.split()]
    else:
        raise ValueError("Unsupported K type")
    if len(vals) == 4:
        fx,fy,cx,cy = vals
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        return K
    elif len(vals) == 9:
        return np.array(vals, dtype=np.float32).reshape(3,3)
    else:
        raise ValueError(f"Expect 4 (fx,fy,cx,cy) or 9 (3x3) floats for K, got {len(vals)}")

def _parse_T(obj):
    # parse 4x4 SE3 from path or string of 16 floats
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        T = obj.astype(np.float32).reshape(4,4)
        return T
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().astype(np.float32).reshape(4,4)
    if isinstance(obj, (list, tuple)):
        vals = [float(x) for x in obj]
    elif isinstance(obj, str):
        s = obj.strip()
        if os.path.isfile(s):
            ext = os.path.splitext(s)[1].lower()
            if ext == ".npy":
                T = np.load(s).astype(np.float32)
                return T.reshape(4,4)
            elif ext in [".txt", ".csv"]:
                vals = []
                with open(s, "r") as f:
                    for line in f:
                        line = line.strip().replace(",", " ")
                        if not line: continue
                        parts = [p for p in line.split() if p]
                        vals.extend([float(p) for p in parts])
            elif ext == ".json":
                import json
                js = json.load(open(s, "r"))
                if "T" in js:
                    T = np.array(js["T"], dtype=np.float32).reshape(4,4)
                    return T
                raise ValueError(f"Unsupported JSON schema for T: {s}")
            else:
                raise ValueError(f"Unsupported T file extension: {ext}")
        else:
            s = s.replace(",", " ")
            vals = [float(x) for x in s.split()]
    else:
        raise ValueError("Unsupported T type")
    if len(vals) == 16:
        return np.array(vals, dtype=np.float32).reshape(4,4)
    else:
        raise ValueError(f"Expect 16 floats for 4x4 T, got {len(vals)}")

def _load_ckpt(model, ckpt_path, device, strict=False):
    if ckpt_path is None or not ckpt_path or (not os.path.isfile(ckpt_path)):
        return
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("model", ck)
    new_state = {}
    for k,v in state.items():
        new_state[k[7:]] = v if k.startswith("module.") else v
    model.load_state_dict(new_state, strict=strict)

def run_image_pair(cfg_path,
                   image=None,
                   depth=None,
                   concat=None,
                   Ks=None,
                   Kt=None,
                   Ts=None,
                   Tt=None,
                   deltaT=None,
                   out="output.png",
                   ckpt=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["infer"]["img_size"]
    amp = cfg["infer"]["amp"]
    # Load inputs
    if concat is not None:
        Is, Ds, (H0, W0) = parse_concat_image(concat, size=size)
    else:
        Is, (H0, W0) = load_image(image, size=size)
        dmode = cfg.get("data", {}).get("depth_mode", "auto")
        dscale = float(cfg.get("data", {}).get("depth_scale", 10.0))
        Ds, _ = load_depth(depth, size=size, mode=dmode, scale=dscale)
    B, _, H, W = Is.shape

    # Parse intrinsics
    Ks_np = _to_numpy_K(Ks) if Ks is not None else None
    Kt_np = _to_numpy_K(Kt) if Kt is not None else None
    if Ks_np is None and Kt_np is None:
        Ks_t = Kt_t = default_K(B,H,W, device=device)
    else:
        if Ks_np is None and Kt_np is not None:
            Ks_np = Kt_np.copy()
        if Kt_np is None and Ks_np is not None:
            Kt_np = Ks_np.copy()
        # 自动识别是否归一化K：若是，则直接按最终尺寸(H,W)放大；否则按原图(H0,W0) -> (H,W)缩放
        if _is_normalized_K(Ks_np):
            Ks_np_fin = _scale_norm_K(Ks_np, H, W)
        else:
            Ks_np_fin = _resize_intrinsics(Ks_np, (H0, W0), (H, W))
        if _is_normalized_K(Kt_np):
            Kt_np_fin = _scale_norm_K(Kt_np, H, W)
        else:
            Kt_np_fin = _resize_intrinsics(Kt_np, (H0, W0), (H, W))
        Ks_t = torch.from_numpy(Ks_np_fin).float().unsqueeze(0).repeat(B,1,1)
        Kt_t = torch.from_numpy(Kt_np_fin).float().unsqueeze(0).repeat(B,1,1)

    # Parse poses
    dT_t = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    if deltaT is not None:
        # 约定：直接传入的deltaT为 camera motion
        dT_np = _parse_T(deltaT)
        dT_t = torch.from_numpy(dT_np).float().unsqueeze(0).repeat(B, 1, 1)
    elif (Ts is not None) and (Tt is not None):
        # Ts/Tt 为世界->相机，先得到 points transform，再取逆得到camera motion
        Ts_np = _parse_T(Ts)
        Tt_np = _parse_T(Tt)
        Ts_t = torch.from_numpy(Ts_np).float().unsqueeze(0)
        Tt_t = torch.from_numpy(Tt_np).float().unsqueeze(0)
        dT_pts = Tt_t @ invert(Ts_t)  # points transform
        dT_t = invert(dT_pts)  # camera motion

    # Build model & load weights
    model = RouteAModel(cfg)
    model.to(device).eval()
    _ckpt = ckpt if ckpt is not None else (cfg.get("infer", {}).get("ckpt", None) or cfg.get("log", {}).get("resume", ""))
    _load_ckpt(model, _ckpt, device, strict=False)

    Is = Is.to(device); Ds = Ds.to(device)
    Ks_t = Ks_t.to(device); Kt_t = Kt_t.to(device); dT_t = dT_t.to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=amp):
            It, aux = model(Is, Ds, Ks_t, Kt_t, dT_t)
    out_img = (It.clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out, out_bgr)
    return out, aux


if __name__ == "__main__":
    # minimal self-test kept
    import numpy as np
    os.makedirs("assets", exist_ok=True)
    H,W=256,512
    rgb = np.zeros((H,W,3), np.uint8)
    rgb[:,:W//2,:]=[255,0,0]
    rgb[:,W//2:,:]=[0,255,0]
    dep = (np.linspace(10,1,W).reshape(1,W).repeat(H,0)).astype(np.float32)
    dep = (dep - dep.min())/(dep.max()-dep.min())
    dep = (dep*255).astype(np.uint8)
    concat = np.concatenate([rgb, cv2.cvtColor(dep, cv2.COLOR_GRAY2BGR)], axis=1)
    cv2.imwrite("assets/test_concat.png", concat)
    cfg_tmp = { "device":"cpu", "infer":{"img_size":[256,256],"amp":False}, "model": { "residual_flow_net": {"in_ch": 5, "base_ch": 16}, "refiner": {"in_ch": 5, "out_ch": 3, "depth": 3, "base_ch": 16}, "softmax_splat": {"temperature": 10.0, "normalize": True}, "grid": {"align_corners": True} } }
    import yaml
    with open("temp_infer.yaml","w") as f:
        yaml.dump(cfg_tmp,f)
    out, aux = run_image_pair("temp_infer.yaml", concat="assets/test_concat.png", out="assets/out.png")
    print("Inference self-tests saved to assets/out.png")
