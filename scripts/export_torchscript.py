# depth_warp_vs/scripts/export_torchscript.py
import argparse, yaml, torch
from models.route_a_model import RouteAModel
from data.cameras.camera import Camera

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="route_a.ts")
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    args = ap.parse_args()
    with open(args.config,"r") as f: cfg = yaml.safe_load(f)
    model = RouteAModel(cfg).eval()
    Is = torch.randn(1,3,args.H,args.W)
    Ds = torch.ones(1,1,args.H,args.W)
    Ks = Kt = Camera.make_default(1,args.H,args.W).K
    dT = torch.eye(4).unsqueeze(0)
    ts = torch.jit.trace(model, (Is, Ds, Ks, Kt, dT))
    ts.save(args.out)
    print(f"Exported {args.out}")

if __name__ == "__main__":
    main()
