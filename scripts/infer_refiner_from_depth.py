# depth_warp_vs/scripts/infer_refiner_from_depth.py
import argparse, os, re, cv2, numpy as np, torch, yaml
from models.refiner.inpaint_refiner import InpaintRefiner
from models.splatting.softmax_splat import softmax_splat

def _extract_ts(path):
    m = re.search(r"frame_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None

def _read_meta(meta_path: str, pose_convention: str = "w2c"):
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    frames={}
    for line in lines[1:]:
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        if len(parts) < 19: continue
        ts = int(parts[0]); fx,fy,cx,cy = map(float, parts[1:5])
        pose_vals = list(map(float, parts[7:7+12]))
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
        P = np.array(pose_vals, np.float32).reshape(3,4)
        T = np.eye(4, dtype=np.float32); T[:3,:4] = P
        if pose_convention.lower()=="c2w":
            T = np.linalg.inv(T).astype(np.float32)
        frames[ts] = (K, T)
    if not frames:
        raise RuntimeError(f"No frames in {meta_path}")
    return frames

def _is_norm_K(K, thr=4.0):
    fx,fy,cx,cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    return (abs(fx)<=thr and abs(fy)<=thr and abs(cx)<=thr and abs(cy)<=thr)

def _scale_K(K, H, W):
    Kp = K.copy().astype(np.float32)
    Kp[0,0] *= float(W); Kp[0,2] *= float(W); Kp[1,1] *= float(H); Kp[1,2] *= float(H); Kp[2,2]=1.0
    return Kp

def _resize_intr(K, src_hw, dst_hw):
    H0,W0 = src_hw; H1,W1 = dst_hw
    sx = W1/max(1.0,W0); sy = H1/max(1.0,H0)
    Kp = K.copy().astype(np.float32)
    Kp[0,0]*=sx; Kp[1,1]*=sy; Kp[0,2]*=sx; Kp[1,2]*=sy; Kp[2,2]=1.0
    return Kp

def _load_rgb(path, size=None):
    im = cv2.imread(path, cv2.IMREAD_COLOR);
    if im is None: raise FileNotFoundError(path)
    H0,W0 = im.shape[:2]; im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if size is not None: im = cv2.resize(im,(size[1], size[0]), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy(im).float().permute(2,0,1).unsqueeze(0)/255.0
    return ten,(H0,W0)

def _load_depth(path, size=None, mode="auto", scale=10.0):
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None: raise FileNotFoundError(path)
    H0,W0 = dep.shape[:2]
    if dep.ndim==3: dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
    dep = dep.astype(np.float32)
    if mode=="metric":
        if dep.max()>255.0: dep/=1000.0
        dep = np.clip(dep,1e-3,1e6)
    else:
        if dep.max()>1.0: dep/=255.0
        dep = np.clip(dep,1e-4,1.0)*float(scale)
    if size is not None:
        dep = cv2.resize(dep,(size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    ten = torch.from_numpy(dep).float().unsqueeze(0).unsqueeze(0)
    return ten, (H0,W0)

def main():
    ap = argparse.ArgumentParser("Infer Refiner from Depth (warp+refine)")
    ap.add_argument("--config", required=True, help="yaml配置（含模型/推理通用参数）")
    ap.add_argument("--image", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pose_convention", default="w2c", choices=["w2c","c2w"])
    ap.add_argument("--target_image", default=None)
    ap.add_argument("--target_timestamp", type=int, default=None)
    ap.add_argument("--img_size", default="0,0", help="H,W (0,0=原尺寸)")
    ap.add_argument("--depth_mode", default="auto", choices=["auto","metric","normalized"])
    ap.add_argument("--depth_scale", type=float, default=10.0)
    ap.add_argument("--tx", type=float, default=0.1)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--tz", type=float, default=0.0)
    ap.add_argument("--rx", type=float, default=0.0)
    ap.add_argument("--ry", type=float, default=0.0)
    ap.add_argument("--rz", type=float, default=0.0)
    ap.add_argument("--hard_z_epsilon", type=float, default=1e-3)
    ap.add_argument("--out", default="refine_out.png")
    ap.add_argument("--ckpt", default="", help="Refiner权重（默认用config里infer.ckpt）")
    args = ap.parse_args()

    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    Hs,Ws = [int(x) for x in args.img_size.replace(","," ").split()]
    resize_to = None if (Hs==0 or Ws==0) else (Hs,Ws)

    Is,(H0,W0) = _load_rgb(args.image, size=resize_to)
    Ds,_ = _load_depth(args.depth, size=resize_to, mode=args.depth_mode, scale=args.depth_scale)

    src_ts = _extract_ts(args.image)
    tgt_ts = _extract_ts(args.target_image) if args.target_image else None
    if args.target_timestamp is not None: tgt_ts = int(args.target_timestamp)
    meta = _read_meta(args.meta, pose_convention=args.pose_convention)
    if src_ts not in meta: raise KeyError(f"ts {src_ts} not in meta")
    Ks_meta, Ts_np = meta[src_ts]
    if tgt_ts is not None:
        if tgt_ts not in meta: raise KeyError(f"tgt ts {tgt_ts} not in meta")
        Kt_meta, Tt_np = meta[tgt_ts]
    else:
        Kt_meta, Tt_np = Ks_meta.copy(), None

    H,W = Is.shape[-2:]
    if _is_norm_K(Ks_meta): Ks_np = _scale_K(Ks_meta, H, W)
    else: Ks_np = _resize_intr(Ks_meta, (H0,W0), (H,W))
    if _is_norm_K(Kt_meta): Kt_np = _scale_K(Kt_meta, H, W)
    else: Kt_np = _resize_intr(Kt_meta, (H0,W0), (H,W))

    Ks = torch.from_numpy(Ks_np).float().unsqueeze(0)
    Kt = torch.from_numpy(Kt_np).float().unsqueeze(0)
    if Tt_np is not None:
        Ts_t = torch.from_numpy(Ts_np).float().unsqueeze(0)
        Tt_t = torch.from_numpy(Tt_np).float().unsqueeze(0)
        # ΔT(camera motion) = invert(Tt @ invert(Ts))
        from data.cameras.pose_utils import invert as se3_invert
        dT_pts = Tt_t @ se3_invert(Ts_t)
        dT = se3_invert(dT_pts)
    else:
        def euler(rx,ry,rz):
            rx=np.deg2rad(rx); ry=np.deg2rad(ry); rz=np.deg2rad(rz)
            cx,sx=np.cos(rx),np.sin(rx); cy,sy=np.cos(ry),np.sin(ry); cz,sz=np.cos(rz),np.sin(rz)
            Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],np.float32)
            Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],np.float32)
            Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],np.float32)
            return (Rz@Ry@Rx).astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = euler(args.rx,args.ry,args.rz)
        T[:3,3] = np.array([args.tx,args.ty,args.tz], np.float32)
        dT = torch.from_numpy(T).float().unsqueeze(0)
    Is,Ds,Ks,Kt,dT = Is.to(device), Ds.to(device), Ks.to(device), Kt.to(device), dT.to(device)

    # 1) warp (硬遮挡)
    Iw, V = softmax_splat(Is, Ds, Ks, Kt, dT, temperature=cfg.get("infer",{}).get("temperature",30.0),
                          normalize=True, occlusion=cfg.get("model",{}).get("softmax_splat",{}).get("occlusion","hard"),
                          hard_z_epsilon=float(cfg.get("model",{}).get("softmax_splat",{}).get("hard_z_epsilon", args.hard_z_epsilon)))
    # 2) 生成mask（1=hole）
    hole_mask = (V <= 1e-8).float()
    x = torch.cat([Iw, hole_mask], dim=1)

    # 3) 加载Refiner并推理
    mcfg = cfg["model"]["refiner"]
    net = InpaintRefiner(in_ch=mcfg.get("in_ch",4), base_ch=mcfg.get("base_ch",48), depth=mcfg.get("depth",5)).to(device).eval()

    ckpt = args.ckpt if args.ckpt else cfg.get("infer",{}).get("ckpt","")
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Refiner ckpt not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    state = state.get("model", state)
    new_state = {}
    for k,v in state.items():
        new_state[k[7:]] = v if k.startswith("module.") else v
    net.load_state_dict(new_state, strict=False)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=bool(cfg.get("infer",{}).get("amp", True)) and (device.startswith("cuda"))):
            It = net(x)
    out = (It.clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"Saved refined image to {args.out}")

if __name__ == "__main__":
    main()
