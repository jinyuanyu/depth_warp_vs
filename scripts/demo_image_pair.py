# depth_warp_vs/scripts/demo_image_pair.py
import argparse
from engine.inference import run_image_pair

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", default=None, help="RGB image path")
    ap.add_argument("--depth", default=None, help="Depth map path (uint16 metric or normalized)")
    ap.add_argument("--concat", default=None, help="path to concatenated RGB|DEPTH image")
    ap.add_argument("--Ks", default=None, help="src intrinsics: path(.npy/.txt/.json) or 'fx,fy,cx,cy' or 9 floats")
    ap.add_argument("--Kt", default=None, help="tgt intrinsics: path(.npy/.txt/.json) or 'fx,fy,cx,cy' or 9 floats")
    ap.add_argument("--Ts", default=None, help="source pose 4x4 (path or 16 floats), world->cam")
    ap.add_argument("--Tt", default=None, help="target pose 4x4 (path or 16 floats), world->cam")
    ap.add_argument("--deltaT", default=None, help="relative pose 4x4 (path or 16 floats), cam_s->cam_t")
    ap.add_argument("--out", default="output.png")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    out, aux = run_image_pair(
        args.config,
        image=args.image,
        depth=args.depth,
        concat=args.concat,
        Ks=args.Ks,
        Kt=args.Kt,
        Ts=args.Ts,
        Tt=args.Tt,
        deltaT=args.deltaT,
        out=args.out,
        ckpt=args.ckpt
    )
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
# usage(单图 + 深度 + 内参（fx,fy,cx,cy）+ 相对位姿): python depth_warp_vs/scripts/demo_image_pair.py --config depth_warp_vs/configs/infer/realtime_512.yaml --image path/to/rgb.png --depth path/to/depth.png --Ks "1200,1200,640,360" --deltaT path/to/deltaT.txt --out out.png
# usage(有源/目标的世界到相机位姿 Ts/Tt): python depth_warp_vs/scripts/demo_image_pair.py --config depth_warp_vs/configs/infer/realtime_512.yaml --image src.png --depth src_depth.png --Ks path/to/Ks.npy --Kt path/to/Kt.npy --Ts path/to/Ts.txt --Tt path/to/Tt.txt --out out.png