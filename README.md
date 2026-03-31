# DepthWarpVS

Depth-guided warp-and-refine pipeline for monocular novel-view synthesis and naked-eye 3D rendering.

This repository focuses on a practical route:

- estimate or provide a monocular depth map
- forward warp the source view to multiple target views with visibility-aware splatting
- detect hole / polluted regions caused by occlusion and depth boundary errors
- refine only these regions with a lightweight mask-guided inpainting network
- fuse the generated views into a lenticular-style naked-eye 3D output

The codebase currently contains both:

- a full research workspace with training, data preparation, experiments, and utilities
- a `reduced_core/` minimal pipeline for novel-view generation and deployment-oriented inference

## Overview

![Pipeline Overview](assets/readme/pipeline_overview.png)

The method is built around a simple but efficient design:

1. single-view RGB + depth as input
2. depth-guided forward splatting to target camera poses
3. visibility accumulation to localize holes
4. optional `pollute` band and `valid` mask to explicitly describe unreliable boundaries
5. MGMI refiner conditioned on warped RGB, masks, and optional human priors
6. final multi-view fusion for naked-eye 3D output

## Results

### Qualitative comparison

![Qualitative Comparison](assets/readme/qualitative_comparison.png)

### Real-time display demo

![Realtime Demo](assets/readme/realtime_demo.png)

## Highlights

- Depth-guided novel-view synthesis with forward splatting
- Hard/soft occlusion handling in `softmax_splat`
- Mask-aware local refinement instead of full-image regeneration
- Explicit `hole`, `valid`, and `pollute` channels for training/inference consistency
- Optional human priors: person segmentation, parsing, keypoints
- Optional boundary sharpening before warp to suppress foreground/background tearing
- Single image, folder, and video inference modes
- Research utilities for training, ablation, and left/right-view evaluation

## What Is In This Repo

### Core inference path

- [`reduced_core/depth_warp_vs/main.py`](reduced_core/depth_warp_vs/main.py)
- [`reduced_core/depth_warp_vs/models/splatting/softmax_splat.py`](reduced_core/depth_warp_vs/models/splatting/softmax_splat.py)
- [`reduced_core/depth_warp_vs/models/refiner/MGMI.py`](reduced_core/depth_warp_vs/models/refiner/MGMI.py)

### Refiner training path

- [`data/mannequin_refine_dataset.py`](data/mannequin_refine_dataset.py)
- [`engine/trainer_refiner.py`](engine/trainer_refiner.py)
- [`configs/mgmi_refiner_train.yaml`](configs/mgmi_refiner_train.yaml)
- [`configs/mgmi_refiner_train_prior.yaml`](configs/mgmi_refiner_train_prior.yaml)

### Data preparation and evaluation

- [`scripts/prepare_simwarp_new.py`](scripts/prepare_simwarp_new.py)
- [`scripts/prepare_priors_dataset.py`](scripts/prepare_priors_dataset.py)
- [`reduced_core/compare/eval_left_right_vda_prior.py`](reduced_core/compare/eval_left_right_vda_prior.py)

### Additional notes

- [`reduced_core/README_REDUCED.md`](reduced_core/README_REDUCED.md): minimal package notes
- [`Agent_depth_warp_vs-CN.md`](Agent_depth_warp_vs-CN.md): Chinese technical summary of the current library

## Repository Layout

```text
depth_warp_vs/
├── README.md
├── assets/readme/                  # figures used by this README
├── configs/                        # training / inference configs
├── data/                           # datasets, camera tools, refine dataset
├── engine/                         # training and evaluation loops
├── models/                         # splatting, refiner, losses, geometry
├── reduced_core/                   # minimal warp + refiner + fusion path
├── runtime/                        # service / runtime-related code
├── scripts/                        # data prep, training, export, demos
├── third_party/Video-Depth-Anything
└── tests/
```

## Installation

The repository root itself is the Python package `depth_warp_vs`, so the safest way is to run commands from its parent directory.

### 1. Install dependencies

From the parent directory of this repo:

```bash
cd /Users/yjy/Desktop/3DView/Zhan
pip install -r depth_warp_vs/requirements.txt
```

If you only want the minimal inference pipeline, the dependency list in [`reduced_core/depth_warp_vs/requirements.txt`](reduced_core/depth_warp_vs/requirements.txt) is enough for most cases.

### 2. Optional components

- [Video-Depth-Anything](third_party/Video-Depth-Anything) for monocular video depth estimation
- `mediapipe` for lightweight person segmentation / keypoints
- `sam2` plus local checkpoints for stronger segmentation priors
- `torch-scatter` for splatting acceleration fallback paths

## Quick Start

### Single-image novel-view generation

Recommended path: run the minimal pipeline in `reduced_core/`.

```bash
cd /Users/yjy/Desktop/3DView/Zhan
PYTHONPATH=/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/reduced_core \
python -m depth_warp_vs.main \
  --image /path/to/source.png \
  --depth /path/to/depth.png \
  --refiner_ckpt /path/to/refiner_ema_best.pth \
  --out /path/to/output.png
```

Useful options:

- `--num_per_side`: number of synthesized views on each side
- `--tx_max` or `--max_disp_px`: target view strength
- `--manual_K fx,fy,cx,cy`: explicit intrinsics
- `--warp_edge_sharpen`: cut ambiguous boundary pixels before warp
- `--refiner_use_pollute`: enable `pollute` band channel
- `--prior_person_seg`: add person segmentation prior
- `--save_views_dir`: save all generated views before fusion

### Video inference

```bash
cd /Users/yjy/Desktop/3DView/Zhan
PYTHONPATH=/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/reduced_core \
python -m depth_warp_vs.main \
  --video /path/to/rgb.mp4 \
  --depth_video /path/to/depth.mp4 \
  --refiner_ckpt /path/to/refiner_ema_best.pth \
  --out /path/to/out.mp4 \
  --num_per_side 1 \
  --max_disp_px 25 \
  --manual_K 1402.1,1402.1,968.77,506.154 \
  --focus_depth 5.9 \
  --ffmpeg_h264
```

### Folder-mode inference

The folder should contain:

- `frame_xxx.png` or `frame_xxx.jpg`
- `depth/depth_xxx.png`

Then run:

```bash
cd /Users/yjy/Desktop/3DView/Zhan
PYTHONPATH=/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/reduced_core \
python -m depth_warp_vs.main \
  --pair_dir /path/to/pair_dir \
  --refiner_ckpt /path/to/refiner_ema_best.pth \
  --out /path/to/out.mp4
```

## Training

The current refiner training pipeline is based on offline-generated warp samples.

### 1. Prepare simulated warp / hole / pollute data

```bash
cd /Users/yjy/Desktop/3DView/Zhan
python depth_warp_vs/scripts/prepare_simwarp_new.py \
  --root /path/to/MannequinChallenge \
  --splits train,validation,test
```

This stage prepares clip folders that contain:

- `sim_warp/`
- `hole_mask/`
- `pollute_mask/`
- `edit_mask/`

### 2. Optional: prepare human priors

```bash
cd /Users/yjy/Desktop/3DView/Zhan
python depth_warp_vs/scripts/prepare_priors_dataset.py \
  --root /path/to/MannequinChallenge \
  --splits train,validation,test \
  --backend sam2 \
  --sam2_ckpt /path/to/sam2_checkpoint.pt
```

This generates optional:

- `prior_person_seg/`
- `prior_parsing/`
- `prior_keypoints/`

### 3. Train the 6-channel baseline refiner

```bash
cd /Users/yjy/Desktop/3DView/Zhan
PYTHONPATH=/Users/yjy/Desktop/3DView/Zhan \
python -m depth_warp_vs.scripts.train_refiner \
  --config /Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/configs/mgmi_refiner_train.yaml
```

Baseline input contract:

`[Iw(3), hole(1), valid(1), pollute(1)]`

### 4. Train the 7-channel prior model

```bash
cd /Users/yjy/Desktop/3DView/Zhan
PYTHONPATH=/Users/yjy/Desktop/3DView/Zhan \
python -m depth_warp_vs.scripts.train_refiner \
  --config /Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/configs/mgmi_refiner_train_prior.yaml
```

Current prior training config uses:

`[Iw(3), hole(1), valid(1), pollute(1), seg(1)]`

## Evaluation

The repo includes an external validation script that uses left/right views as pseudo-ground-truth pairs and can optionally estimate depth with Video-Depth-Anything.

```bash
cd /Users/yjy/Desktop/3DView/Zhan
python depth_warp_vs/reduced_core/compare/eval_left_right_vda_prior.py \
  --refiner_ckpt /path/to/baseline_ema_best.pth \
  --refiner_ckpt_prior /path/to/prior_ema_best.pth \
  --out_dir /path/to/eval_out
```

Typical outputs:

- `metrics_summary.json`
- `comparison_panel.png`
- `baseline_best_view.png`
- `prior_fair_view.png`

## Current Design Choices

The latest code path in this repo is organized around the following idea:

- keep geometry explicit in the warp stage
- represent uncertainty explicitly with masks
- let the refiner repair only the risky regions

In practice, this means the final novel view is computed as:

`Ifinal = lerp(Iw, Ipred, mk_union)`

where:

- `Iw` is the warped image
- `mk_union` is usually `hole ∪ pollute`
- `Ipred` is the refiner output

This design improves locality and efficiency, but it also means the method is still strongly limited by depth quality and visibility reasoning.

## Known Limitations

This repository is research code and still exposes several limitations that matter in real scenes:

- Strong dependence on sharp, well-aligned depth boundaries
- Face/body interior may remain too similar to the source view when most pixels are still marked visible
- Large self-occlusion reveals regions that are fundamentally missing in single-view input
- Training/inference mismatch can significantly degrade quality
- Adding priors without matching the training distribution can cause severe OOD failures

The repo already contains experiments on:

- `valid` mask input
- `pollute` band supervision
- warp-edge sharpening
- segmentation / parsing / keypoint priors
- left/right-view external evaluation

## Recommended Reading Order

If you want to understand the code quickly, read in this order:

1. [`reduced_core/depth_warp_vs/main.py`](reduced_core/depth_warp_vs/main.py)
2. [`reduced_core/depth_warp_vs/models/splatting/softmax_splat.py`](reduced_core/depth_warp_vs/models/splatting/softmax_splat.py)
3. [`reduced_core/depth_warp_vs/models/refiner/MGMI.py`](reduced_core/depth_warp_vs/models/refiner/MGMI.py)
4. [`data/mannequin_refine_dataset.py`](data/mannequin_refine_dataset.py)
5. [`engine/trainer_refiner.py`](engine/trainer_refiner.py)
6. [`configs/mgmi_refiner_train.yaml`](configs/mgmi_refiner_train.yaml)
7. [`configs/mgmi_refiner_train_prior.yaml`](configs/mgmi_refiner_train_prior.yaml)

## Citation

If you use this repository in academic work, please cite the corresponding paper once the bibliographic entry is finalized.

For now, please also cite the third-party depth estimator if you use it:

- [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)

## Acknowledgements

- Softmax splatting / depth-guided warping literature
- Video-Depth-Anything for monocular depth estimation experiments
- MediaPipe and SAM2 for optional human priors

## Contact

Issues and pull requests are welcome for:

- bug fixes
- cleaner training / evaluation protocols
- improved prior integration
- better README and examples
