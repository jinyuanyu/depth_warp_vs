# depth_warp_vs/scripts/train_refiner.py
import argparse
from depth_warp_vs.engine.trainer_refiner import train_refiner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train_refiner(args.config)

if __name__ == "__main__":
    main()
# (DWvs) D:\naked-eye 3D\video call>python -m depth_warp_vs.scripts.train_refiner --config depth_warp_vs/configs/refiner_train.yaml