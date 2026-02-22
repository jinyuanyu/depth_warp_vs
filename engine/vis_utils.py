# depth_warp_vs/engine/vis_utils.py
import os
import json
import shutil
import time
from typing import Dict, List, Tuple
import numpy as np

def _now(fmt="%Y%m%d%H%M%S"):
    return time.strftime(fmt, time.localtime())

def make_tmp_run_dir(base_dir: str, model_name: str) -> str:
    # 临时运行目录：./checkpoints/<ModelName>/tmp_<timestamp>
    base = os.path.join(base_dir, str(model_name))
    os.makedirs(base, exist_ok=True)
    tmp_dir = os.path.join(base, f"tmp_{_now()}")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

def finalize_run_dir(tmp_dir: str, base_dir: str, model_name: str) -> str:
    # 训练完成后将 tmp_dir 重命名为完成时间（年月日小时）
    final_name = time.strftime("%Y%m%d%H", time.localtime())
    final_dir = os.path.join(base_dir, str(model_name), final_name)
    if os.path.abspath(tmp_dir) == os.path.abspath(final_dir):
        return final_dir
    # 若目标已存在，则加后缀避免覆盖
    suf = 0
    cand = final_dir
    while os.path.exists(cand):
        suf += 1
        cand = final_dir + f"_{suf}"
    final_dir = cand
    try:
        os.rename(tmp_dir, final_dir)
    except Exception:
        # 跨盘或权限限制时，采用复制+删除兜底
        shutil.copytree(tmp_dir, final_dir, dirs_exist_ok=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return final_dir

def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(history: Dict[str, List[float]], out_path: str):
    # history 包含 step、loss、lr、psnr、ssim 等
    keys = list(history.keys())
    n = len(history.get("step", []))
    lines = []
    lines.append(",".join(keys))
    for i in range(n):
        vals = []
        for k in keys:
            arr = history.get(k, [])
            vals.append(str(arr[i] if i < len(arr) else ""))
        lines.append(",".join(vals))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def plot_curves(history: Dict[str, List[float]], out_dir: str):
    # 生成 loss_curve.png / lr_curve.png / metrics_curve.png
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = history.get("step", [])
    os.makedirs(out_dir, exist_ok=True)

    # Loss
    if steps and history.get("loss", []):
        plt.figure()
        plt.plot(steps, history["loss"], label="loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"))
        plt.close()

    # LR
    if steps and history.get("lr", []):
        plt.figure()
        plt.plot(steps, history["lr"], label="lr")
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title("Learning Rate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "lr_curve.png"))
        plt.close()

    # Metrics
    has_psnr = "psnr" in history and len(history["psnr"]) == len(steps)
    has_ssim = "ssim" in history and len(history["ssim"]) == len(steps)
    if steps and (has_psnr or has_ssim):
        plt.figure()
        if has_psnr:
            plt.plot(steps, history["psnr"], label="PSNR")
        if has_ssim:
            plt.plot(steps, history["ssim"], label="SSIM")
        plt.xlabel("step")
        plt.ylabel("metric")
        plt.title("Training Metrics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "metrics_curve.png"))
        plt.close()

def _to_uint8(img01: np.ndarray) -> np.ndarray:
    x = np.clip(img01, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return x

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    将输入统一为 HxWx3：
    - HxW           -> 重复到3通道
    - HxWx1         -> 重复到3通道
    - HxWx3         -> 原样返回
    - HxWxC (C>3)   -> 取前3通道
    """
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3:
        if img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        if img.shape[2] >= 3:
            return img[..., :3]
    # 兜底：尝试展平成HxW再扩3通道
    img2d = img.reshape(img.shape[0], img.shape[1])
    return np.repeat(img2d[..., None], 3, axis=2)

def save_eval_samples(samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], out_path: str, cols: int = 4, max_rows: int = 4):
    """
    samples: list of (Iw, Mask, PredFinal, GT)，每个元素为 HxWx3 或 HxWx1 或 HxW 的 numpy [0,1].
    保存为网格图片，行数不超过 max_rows。
    """
    import cv2
    if not samples:
        return

    rows = min(max_rows, len(samples))
    # 用第一条样本的 Iw 形状为参考
    Iw0, Mk0, Pf0, Gt0 = samples[0]
    Iw0 = _ensure_rgb(Iw0)
    Mk0 = _ensure_rgb(Mk0)
    Pf0 = _ensure_rgb(Pf0)
    Gt0 = _ensure_rgb(Gt0)
    H, W = Iw0.shape[:2]

    canvas_rows = []
    for i in range(rows):
        Iw, Mk, Pf, Gt = samples[i]
        Iw = _ensure_rgb(Iw)
        Mk = _ensure_rgb(Mk)
        Pf = _ensure_rgb(Pf)
        Gt = _ensure_rgb(Gt)

        # 统一尺寸到参考尺寸（第一条样本）
        if Iw.shape[:2] != (H, W):
            Iw = cv2.resize(_to_uint8(Iw), (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)
            Iw = Iw.astype(np.float32) / 255.0
        if Mk.shape[:2] != (H, W):
            Mk = cv2.resize(_to_uint8(Mk), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            Mk = Mk.astype(np.float32) / 255.0
        if Pf.shape[:2] != (H, W):
            Pf = cv2.resize(_to_uint8(Pf), (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)
            Pf = Pf.astype(np.float32) / 255.0
        if Gt.shape[:2] != (H, W):
            Gt = cv2.resize(_to_uint8(Gt), (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)
            Gt = Gt.astype(np.float32) / 255.0

        Iw8 = _to_uint8(_ensure_rgb(Iw))
        Mk8 = _to_uint8(_ensure_rgb(Mk))
        Pf8 = _to_uint8(_ensure_rgb(Pf))
        Gt8 = _to_uint8(_ensure_rgb(Gt))

        row_img = np.concatenate([Iw8, Mk8, Pf8, Gt8], axis=1)
        canvas_rows.append(row_img)

    out = np.concatenate(canvas_rows, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
