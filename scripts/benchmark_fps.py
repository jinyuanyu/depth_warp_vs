# depth_warp_vs/scripts/benchmark_fps.py
import argparse, yaml, time, torch
from models.route_a_model import RouteAModel
from data.cameras.camera import Camera

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)
    device = cfg.get("device","cuda" if torch.cuda.is_available() else "cpu")
    model = RouteAModel(cfg).to(device).eval()
    Is = torch.randn(1,3,args.H,args.W, device=device)
    Ds = torch.ones(1,1,args.H,args.W, device=device)
    Ks = Kt = Camera.make_default(1,args.H,args.W, device=device).K
    dT = torch.eye(4, device=device).unsqueeze(0)
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model(Is, Ds, Ks, Kt, dT)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(args.iters):
        with torch.no_grad():
            _ = model(Is, Ds, Ks, Kt, dT)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = (time.time() - t0) / args.iters
    print(f"HxW={args.H}x{args.W}: {1.0/dt:.2f} FPS, {dt*1000:.2f} ms")
if __name__ == "__main__":
    main()
