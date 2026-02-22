# depth_warp_vs/scripts/train.py
import argparse
from engine.trainer import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)

if __name__ == "__main__":
    main()
