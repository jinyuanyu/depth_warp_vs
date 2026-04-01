# depth_warp_vs/batch_synth_warp.py
import os, re, argparse, cv2, numpy as np, torch
from tqdm import tqdm

from models.splatting.softmax_splat import softmax_splat

def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H,W = img.shape[:2]
    ten = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0) / 255.0
    return ten, (H,W)

def _read_depth(path, mode="auto", scale=10.0):
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None: raise FileNotFoundError(path)
    if dep.ndim == 3: dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
    dep = dep.astype(np.float32)
    if mode == "metric":
        if dep.dtype == np.uint16 or dep.max() > 255.0:
            dep = dep / 1000.0
        dep = np.clip(dep, 1e-3, 1e6)
    else:
        if dep.max() > 1.0: dep = dep / 255.0
        dep = np.clip(dep, 1e-4, 1.0) * float(scale)
    ten = torch.from_numpy(dep).float().unsqueeze(0).unsqueeze(0)
    return ten

def _is_norm_K(K: np.ndarray, thr=4.0):
    fx,fy,cx,cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    return (abs(fx)<=thr and abs(fy)<=thr and abs(cx)<=thr and abs(cy)<=thr)

def _scale_norm_K(K: np.ndarray, H:int, W:int):
    Kp = K.copy().astype(np.float32)
    Kp[0,0] *= float(W); Kp[0,2] *= float(W); Kp[1,1] *= float(H); Kp[1,2] *= float(H)
    Kp[2,2]=1.0
    return Kp

def _parse_meta(meta_path, pose_convention="w2c"):
    frames={}
    if not os.path.isfile(meta_path): return frames
    with open(meta_path,"r",encoding="utf-8") as f:
        lines=[ln.strip() for ln in f if ln.strip()]
    for line in lines[1:]:
        parts=[p for p in re.split(r"[\s,]+", line) if p]
        if len(parts)<19: continue
        try:
            ts=int(parts[0])
            fx,fy,cx,cy=map(float, parts[1:5])
            pose_vals=list(map(float, parts[7:7+12]))
        except Exception:
            continue
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        P=np.array(pose_vals, dtype=np.float32).reshape(3,4)
        T=np.eye(4, dtype=np.float32); T[:3,:4]=P
        if pose_convention.lower()=="c2w": T=np.linalg.inv(T).astype(np.float32)
        frames[ts]=(K,T)
    return frames

def _timestamp_from_path(p):
    m=re.search(r"frame_(\d+)", os.path.basename(p))
    if not m: return None
    try: return int(m.group(1))
    except Exception: return None

def _deltaT_from_depth_and_K(D: np.ndarray, K_px: np.ndarray, min_px, max_px, depth_percentiles=(0.05,0.95), direction="both"):
    z=D.reshape(-1)
    z=z[np.isfinite(z)&(z>1e-6)]
    if z.size<16:
        z_near, z_far = float(D.min()), float(D.max())
    else:
        z_near=float(np.quantile(z, depth_percentiles[0]))
        z_far=float(np.quantile(z, depth_percentiles[1]))
    z_near=max(z_near,1e-3); z_far=max(z_far, z_near+1e-6)
    fx=float(K_px[0,0])
    tx_raw=(min_px*z_far)/max(1e-6,fx)
    disp_near=fx*tx_raw/z_near
    tx=tx_raw
    if disp_near>max_px:
        tx=(max_px*z_near)/max(1e-6,fx)
    if direction=="left": sgn=-1.0
    elif direction=="right": sgn=1.0
    else: sgn=-1.0 if (np.random.rand()<0.5) else 1.0
    T=np.eye(4, dtype=np.float32); T[0,3]=sgn*tx
    return T

def main():
    ap=argparse.ArgumentParser("Batch synth warp generator (horizontal camera motion)")
    ap.add_argument("--root", required=True, help="MannequinChallenge 根目录")
    ap.add_argument("--split", default="train", choices=["train","validation","test"])
    ap.add_argument("--pose_convention", default="w2c", choices=["w2c","c2w"])
    ap.add_argument("--img_size", default="0,0", help="H,W (0,0=使用原图尺寸)")
    ap.add_argument("--depth_mode", default="auto", choices=["auto","metric","normalized"])
    ap.add_argument("--depth_scale", type=float, default=10.0)
    ap.add_argument("--min_px", type=float, default=4.0)
    ap.add_argument("--max_px", type=float, default=48.0)
    ap.add_argument("--percentiles", default="0.05,0.95", help="z近远分位点")
    ap.add_argument("--direction", default="both", choices=["left","right","both"])
    ap.add_argument("--occlusion", default="hard", choices=["hard","soft"])
    ap.add_argument("--temperature", type=float, default=30.0)
    ap.add_argument("--hard_z_epsilon", type=float, default=1e-3)
    args=ap.parse_args()

    Hs,Ws=[int(x) for x in args.img_size.replace(","," ").split()]
    resize_to=None if (Hs==0 or Ws==0) else (Hs,Ws)
    p0,p1=[float(x) for x in args.percentiles.replace(","," ").split()]
    root=args.root; split=args.split
    split_dir=os.path.join(root, split)
    device="cuda" if torch.cuda.is_available() else "cpu"

    # 遍历所有clip
    txts=[p for p in os.listdir(split_dir) if p.endswith(".txt")]
    for txt in txts:
        clip=os.path.splitext(txt)[0]
        clip_dir=os.path.join(split_dir, clip)
        depth_dir=os.path.join(clip_dir, "depth")
        if not os.path.isdir(clip_dir) or not os.path.isdir(depth_dir): continue

        meta=_parse_meta(os.path.join(split_dir, txt), pose_convention=args.pose_convention)

        out_dir=os.path.join(clip_dir, "synth_warp")
        os.makedirs(out_dir, exist_ok=True)

        frames=sorted([f for f in os.listdir(clip_dir) if f.startswith("frame_") and f.lower().endswith((".jpg",".png",".jpeg"))], key=lambda s:[int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)])
        pbar=tqdm(frames, desc=f"{clip}")
        for f in pbar:
            img_path=os.path.join(clip_dir, f)
            ts=_timestamp_from_path(img_path)
            dep_name=f"depth_{ts}.png" if ts is not None else None
            dep_path=os.path.join(depth_dir, dep_name) if dep_name else None
            if (ts is None) or (not os.path.isfile(dep_path)): continue

            Is,(H0,W0)=_read_rgb(img_path)
            Ds=_read_depth(dep_path, mode=args.depth_mode, scale=args.depth_scale)
            if resize_to is not None:
                Is=torch.nn.functional.interpolate(Is, size=resize_to, mode="bilinear", align_corners=True)
                Ds=torch.nn.functional.interpolate(Ds, size=resize_to, mode="nearest")
                H,W=resize_to
            else:
                H,W=Is.shape[-2:]

            # 内参缩放
            if ts in meta:
                K_norm,_=meta[ts]
                if _is_norm_K(K_norm):
                    Kpx=_scale_norm_K(K_norm, H, W)
                else:
                    # 极少数情况：meta已是像素尺度
                    Kpx=K_norm.astype(np.float32)
            else:
                # 回退一个大致默认K
                fx=fy=0.9*W; cx=W/2.0; cy=H/2.0
                Kpx=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)

            # 依据深度估计水平平移
            Dnp=Ds.squeeze(0).squeeze(0).numpy().astype(np.float32)
            dT_np=_deltaT_from_depth_and_K(Dnp, Kpx, args.min_px, args.max_px, depth_percentiles=(p0,p1), direction=args.direction)

            Is=Is.to(device); Ds=Ds.to(device)
            Ks=torch.from_numpy(Kpx).float().unsqueeze(0).to(device)
            Kt=Ks.clone()
            dT=torch.from_numpy(dT_np).float().unsqueeze(0).to(device)

            with torch.no_grad():
                Iw, V = softmax_splat(Is, Ds, Ks, Kt, dT,
                                      temperature=args.temperature,
                                      normalize=True,
                                      occlusion=args.occlusion,
                                      hard_z_epsilon=args.hard_z_epsilon)
            Iw_np=(Iw.clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
            Vn=V.cpu().squeeze(0).squeeze(0).numpy()
            Vn=Vn - Vn.min()
            if Vn.max()>1e-8: Vn=Vn/Vn.max()
            Vc=(np.stack([Vn,Vn,Vn], axis=-1)*255).astype(np.uint8)

            out_warp=os.path.join(out_dir, f.replace("frame_","warp_"))
            out_vis =os.path.join(out_dir, f.replace("frame_","vis_"))
            cv2.imwrite(out_warp, cv2.cvtColor(Iw_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_vis, Vc)
