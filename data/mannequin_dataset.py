# depth_warp_vs/data/mannequin_dataset.py
import os, cv2, re
import numpy as np
import torch
from torch.utils.data import Dataset

def _invert_se3_np(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    Rt = R.T
    tinv = -Rt @ t
    Tout = np.eye(4, dtype=np.float32)
    Tout[:3, :3] = Rt
    Tout[:3, 3:4] = tinv
    return Tout

def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def _read_rgb(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    H0, W0 = img.shape[:2]
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(img).float().permute(2,0,1) / 255.0
    return ten, (H0, W0)

def _depth_from_image(dep_img, mode="auto", scale=10.0):
    if dep_img.ndim == 3:
        dep_img = cv2.cvtColor(dep_img, cv2.COLOR_BGR2GRAY)
    dep = dep_img.astype(np.float32)
    if mode == "metric":
        if dep.dtype == np.uint16 or dep.max() > 255.0:
            dep = dep / 1000.0
        dep = np.clip(dep, 1e-3, 1e6)
    elif mode == "normalized":
        if dep.max() > 1.0:
            dep = dep / 255.0
        dep = np.clip(dep, 1e-4, 1.0) * float(scale)
    else:  # auto
        if dep.dtype == np.uint16 or dep.max() > 255.0:
            dep = dep / 1000.0
            dep = np.clip(dep, 1e-3, 1e6)
        else:
            if dep.max() > 1.0:
                dep = dep / 255.0
            dep = np.clip(dep, 1e-4, 1.0) * float(scale)
    return dep

def _read_depth(path, size=None, mode="auto", scale=10.0):
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(path)
    H0, W0 = dep.shape[:2]
    if size is not None:
        dep = cv2.resize(dep, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    depf = _depth_from_image(dep, mode=mode, scale=scale)
    ten = torch.from_numpy(depf).float().unsqueeze(0)
    return ten, (H0, W0)

def _resize_intrinsics(K, src_hw, dst_hw):
    H0, W0 = src_hw; H1, W1 = dst_hw
    sx = W1 / max(1.0, float(W0)); sy = H1 / max(1.0, float(H0))
    K = K.copy()
    K[0,0] *= sx; K[1,1] *= sy; K[0,2] *= sx; K[1,2] *= sy
    return K

def _default_K(H, W, fx=0, fy=0, cx=0, cy=0, explicit=None):
    if explicit is not None and len(explicit)==9:
        K = np.array(explicit, dtype=np.float32).reshape(3,3)
        return K
    if fx<=0 or fy<=0:
        fx = fy = 0.9 * W
    if cx<=0 or cy<=0:
        cx = W/2.0; cy = H/2.0
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    return K

def _extract_id(basename, prefix="frame_", depth_prefix="depth_"):
    name = os.path.splitext(basename)[0]
    if name.startswith(prefix):
        return name[len(prefix):]
    if name.startswith(depth_prefix):
        return name[len(depth_prefix):]
    return None

def _pair_indices(n, window, neighbors):
    # 中间帧作为src，生成(src, tgt_idx)的对
    if window % 2 == 0: window += 1
    half = window // 2
    pairs = []
    for i in range(n):
        # 以每个位置为中心形成窗口，若窗口超界则跳过
        if i - half < 0 or i + half >= n:
            continue
        for off in neighbors:
            j = i + off
            if j < 0 or j >= n or j == i: continue
            pairs.append((i, j))
    return pairs

class MannequinChallengeDataset(Dataset):
    def __init__(self, cfg, split="train"):
        data_cfg = cfg["data"]
        self.root = data_cfg.get("root", "./MannequinChallenge")
        self.split = data_cfg.get("split", split)
        self.img_size = tuple(data_cfg.get("img_size", [384, 512]))
        self.resize_keep_aspect = bool(data_cfg.get("resize_keep_aspect", False))
        self.depth_mode = data_cfg.get("depth_mode", "auto")
        self.depth_scale = float(data_cfg.get("depth_scale", 10.0))
        self.neighbors = data_cfg.get("neighbors", [-2, -1, 1, 2])
        self.window = int(data_cfg.get("window", 5))
        self.max_pairs_per_clip = int(data_cfg.get("max_pairs_per_clip", 0))
        self.use_pose_pnp = bool(data_cfg.get("use_pose_pnp", True))
        self.orb_nfeatures = int(data_cfg.get("orb_nfeatures", 2000))
        self.min_depth = float(data_cfg.get("min_depth", 1e-3))
        self.explicit_K = data_cfg.get("explicit_K", [])
        self.fx = float(data_cfg.get("fx", 0))
        self.fy = float(data_cfg.get("fy", 0))
        self.cx = float(data_cfg.get("cx", 0))
        self.cy = float(data_cfg.get("cy", 0))

        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        # 遍历clip
        self.samples = []  # list of dict {src, tgt, src_depth, Ks, Kt, dT}
        clips = sorted([p for p in os.listdir(split_dir) if p.endswith(".txt")], key=_natural_key)
        for txt in clips:
            stem = os.path.splitext(txt)[0]
            clip_dir = os.path.join(split_dir, stem)
            if not os.path.isdir(clip_dir):
                continue
            depth_dir = os.path.join(clip_dir, "depth")
            if not os.path.isdir(depth_dir):
                continue
            # 读取帧列表
            frames = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith((".jpg",".png",".jpeg")) and f.startswith("frame_")], key=_natural_key)
            ids = []
            for f in frames:
                idv = _extract_id(f, prefix="frame_")
                if idv is not None:
                    ids.append(idv)
            # 过滤出有对应深度的帧
            pairs = []
            valid_frames = []
            for f, idv in zip(frames, ids):
                dep_name = f"depth_{idv}.png"
                if os.path.isfile(os.path.join(depth_dir, dep_name)):
                    valid_frames.append((f, dep_name))
            if len(valid_frames) < 3:
                continue
            n = len(valid_frames)
            idx_pairs = _pair_indices(n, self.window, self.neighbors)
            if self.max_pairs_per_clip > 0:
                idx_pairs = idx_pairs[:self.max_pairs_per_clip]
            for si, ti in idx_pairs:
                sf, sd = valid_frames[si]
                tf, td = valid_frames[ti]
                self.samples.append({
                    "src_rgb": os.path.join(clip_dir, sf),
                    "src_dep": os.path.join(depth_dir, sd),
                    "tgt_rgb": os.path.join(clip_dir, tf),
                })
        if len(self.samples) == 0:
            raise RuntimeError(f"No training pairs found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def _estimate_deltaT(self, Is_np, It_np, Ds_np, K_np):
        if not self.use_pose_pnp:
            T_pts = np.eye(4, dtype=np.float32)
            # 约定输出 camera motion（cam_s -> cam_t）
            return _invert_se3_np(T_pts)
        try:
            from .pose_estimation import estimate_pose_pnp
            T_pts = estimate_pose_pnp(Is_np, It_np, Ds_np, K_np, nfeatures=self.orb_nfeatures, min_depth=self.min_depth)
            # 将points transform 转为 camera motion
            T_cam = _invert_se3_np(T_pts)
            return T_cam
        except Exception:
            T_pts = np.eye(4, dtype=np.float32)
            return _invert_se3_np(T_pts)

    def __getitem__(self, idx):
        r = self.samples[idx]
        # 读取图像/深度（先原始，再resize并调整内参）
        Is_t, (Hs0, Ws0) = _read_rgb(r["src_rgb"], size=None)
        It_t, (Ht0, Wt0) = _read_rgb(r["tgt_rgb"], size=None)
        Ds_t, _ = _read_depth(r["src_dep"], size=None, mode=self.depth_mode, scale=self.depth_scale)

        H, W = self.img_size
        if Is_t.shape[-2:] != (H,W):
            Is_t = torch.nn.functional.interpolate(Is_t.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=True).squeeze(0)
            It_t = torch.nn.functional.interpolate(It_t.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=True).squeeze(0)
            Ds_t = torch.nn.functional.interpolate(Ds_t.unsqueeze(0), size=(H,W), mode="nearest").squeeze(0)

        # 构造内参K，并按resize比例调整
        K0 = _default_K(Hs0, Ws0, self.fx, self.fy, self.cx, self.cy, explicit=self.explicit_K if self.explicit_K else None)
        K = _resize_intrinsics(K0, (Hs0, Ws0), (H, W)).astype(np.float32)

        # 估计ΔT（源→目标的相机坐标变换）
        Is_np = (Is_t.permute(1,2,0).numpy()*255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
        It_np = (It_t.permute(1,2,0).numpy()*255).astype(np.uint8)[:, :, ::-1]
        Ds_np = Ds_t.squeeze(0).numpy().astype(np.float32)
        dT = self._estimate_deltaT(Is_np, It_np, Ds_np, K)

        # 打包
        Ks_t = torch.from_numpy(K).float()
        Kt_t = torch.from_numpy(K.copy()).float()
        dT_t = torch.from_numpy(dT).float()
        return Is_t, Ds_t, Ks_t, Kt_t, dT_t, It_t

if __name__ == "__main__":
    # 简单自测：构造一个最小数据集目录或跳过
    # 这里只测试基本返回类型与尺寸
    cfg = {"data": {"root":"D:\\naked-eye 3D\\video call\\depth_warp_vs\\data\\datasets\\MannequinChallenge", "split":"train", "img_size":[128,128], "neighbors":[-1,1], "window":3, "use_pose_pnp": False}}
    try:
        ds = MannequinChallengeDataset(cfg, split="train")
        a = ds[0]
        assert len(a)==6
        print("MannequinChallengeDataset basic test passed")
    except Exception as e:
        print("Dataset self-test skipped or failed:", e)
