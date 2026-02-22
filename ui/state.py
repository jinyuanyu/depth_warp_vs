# depth_warp_vs/ui/state.py
import os
from dataclasses import dataclass

@dataclass
class RunParams:
    mode: str  # "image" | "dir" | "video"
    image_path: str = ""
    depth_path: str = ""
    pair_dir: str = ""
    video_path: str = ""
    depth_video_path: str = ""

    img_size: str = "0,0"
    fuse_size: str = "2160,3840"
    manual_K: str = ""
    K_units: str = "auto"
    meta_path: str = ""
    pose_convention: str = "w2c"
    fps: float = 30.0

    depth_mode: str = "auto"
    depth_scale: float = 10.0
    depth_color_decode: str = "auto"
    far_value: str = "larger"

    rescale_depth: str = "linear"
    rescale_range: str = "2,10"
    rescale_percentiles: str = "1,99"

    num_per_side: int = 4
    spacing: str = "linear"
    tx_max: float = 0.0
    max_disp_px: float = 48.0
    disp_ref_percentile: float = 0.5

    temperature: float = 30.0
    occlusion: str = "hard"
    hard_z_epsilon: float = 1e-3
    amp: bool = False

    refiner_type: str = "MGMI"
    refiner_ckpt: str = ""

    slope: float = 0.166666
    X: float = 4.66666
    koff: float = 5.0

    focus_depth: float = 0.0
    out_path: str = ""

    gpu: int = -1
    save_views_dir: str = ""
    chunk_views: int = 0

    writer_kind: str = "OpenCV mp4v"  # "OpenCV mp4v", "OpenCV lossless", "FFmpeg lossless", "FFmpeg H.264"
    ffmpeg_close_timeout: float = 300.0
    h264_qp: int = 23
    h264_preset: str = "medium"
    h264_pix_fmt: str = "yuv420p"

def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
