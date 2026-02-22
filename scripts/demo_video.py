# depth_warp_vs/scripts/demo_video.py
import argparse, yaml, cv2, torch
import numpy as np
from models.route_a_model import RouteAModel
from data.cameras.camera import Camera

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--depth_video", required=True)
    ap.add_argument("--out", default="video_out.mp4")
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)
    cap = cv2.VideoCapture(args.video)
    capd = cv2.VideoCapture(args.depth_video)
    assert cap.isOpened() and capd.isOpened()
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = cfg["infer"]["img_size"]
    device = cfg.get("device","cuda" if torch.cuda.is_available() else "cpu")
    model = RouteAModel(cfg).to(device).eval()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (size[1], size[0]))
    while True:
        ret, f1 = cap.read()
        ret2, fd = capd.read()
        if not ret or not ret2: break
        rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        dep = fd
        if dep.ndim == 3: dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
        rgb = cv2.resize(rgb, (size[1], size[0]))
        dep = cv2.resize(dep, (size[1], size[0]))
        Is = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0)/255.0
        dep = dep.astype(np.float32)
        dep = dep / max(1.0, dep.max())
        Ds = torch.from_numpy(dep).float().unsqueeze(0).unsqueeze(0) * 10.0
        Is, Ds = Is.to(device), Ds.to(device)
        Ks = Kt = Camera.make_default(1,size[0],size[1], device=device).K
        dT = torch.eye(4, device=device).unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg["infer"]["amp"]):
                It, aux = model(Is, Ds, Ks, Kt, dT)
        out = (It.clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)
    cap.release(); capd.release(); writer.release()
    print(f"Saved video to {args.out}")

if __name__ == "__main__":
    main()
