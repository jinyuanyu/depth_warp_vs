# depth_warp_vs/data/mannequin_refine_dataset.py
import os, re, cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def _read_rgb(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy(img).float().permute(2,0,1) / 255.0
    return ten

def _read_mask_raw(path, size=None):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if size is not None:
        m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return m  # uint8 [0..255]

def _binarize01(m):
    return (m > 127).astype(np.uint8)

def _apply_morph(mask01, op, ksize, iters):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if op == "dilate":
        mm = cv2.dilate(mask01, kernel, iterations=iters)
    elif op == "erode":
        mm = cv2.erode(mask01, kernel, iterations=iters)
    elif op == "open":
        mm = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, kernel, iterations=iters)
    elif op == "close":
        mm = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, kernel, iterations=iters)
    else:
        mm = mask01
    return (mm > 0).astype(np.uint8)

def _extract_id_from_frame(name):
    base = os.path.splitext(name)[0]
    m = re.match(r"frame_(\d+)", base)
    return m.group(1) if m else None

class MannequinRefineDataset(Dataset):
    """
    读取模拟生成的 warp / hole_mask / pollute_mask / edit_mask 以及原始 GT frame。
    返回：
      x: 5通道张量 [Iw(3), Mk_hole(1), Mk_pollute(1)]，范围[0,1]
      gt: 3通道张量，范围[0,1]
      mk_union: 1通道张量（空洞 ∪ 污染带），范围[0,1]
    """
    def __init__(self, cfg, split="train"):
        data_cfg = cfg.get("data", {})
        self.root = data_cfg.get("root", "./MannequinChallenge")
        self.split = split
        # 与新代码保持一致的默认分辨率，如果cfg中指定则优先
        self.img_size = tuple(data_cfg.get("img_size", [360, 640]))
        self.max_per_clip = int(data_cfg.get("max_per_clip", 0))

        # 掩码增广配置（应用于 edit_mask）
        aug_cfg = data_cfg.get("mask_aug", {})
        # 训练默认有增广，其他split默认不增广（兼容你原有行为）
        default_p = 0.6 if split == "train" else 0.0
        self.mask_aug_p = float(aug_cfg.get("p", default_p))
        self.mask_aug_ops = list(aug_cfg.get("ops", ["dilate","erode","open","close","none"]))
        k_rng = aug_cfg.get("k", [3,7])
        it_rng = aug_cfg.get("iters", [1,2])
        self.k_min, self.k_max = int(min(k_rng)), int(max(k_rng))
        self.it_min, self.it_max = int(min(it_rng)), int(max(it_rng))

        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        # 样本项：(iw, hole, pollute, edit, gt)
        self.samples = []
        # 遍历clip子目录，兼容新代码的数据组织
        clips = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        clips.sort(key=_natural_key)
        for clip in clips:
            clip_dir = os.path.join(split_dir, clip)
            sim_dir = os.path.join(clip_dir, "sim_warp")
            hole_dir = os.path.join(clip_dir, "hole_mask")
            pollute_dir = os.path.join(clip_dir, "pollute_mask")
            edit_dir = os.path.join(clip_dir, "edit_mask")
            if not (os.path.isdir(sim_dir) and os.path.isdir(hole_dir) and os.path.isdir(pollute_dir) and os.path.isdir(edit_dir)):
                continue

            # 以 sim_warp 中的warp文件为基准
            warp_files = [f for f in os.listdir(sim_dir)
                          if f.lower().startswith("warp_") and f.lower().endswith((".png",".jpg",".jpeg"))]
            warp_files.sort(key=_natural_key)

            cnt = 0
            for wf in warp_files:
                stem = os.path.splitext(wf)[0]  # warp_XXXX
                ts = stem.replace("warp_", "")
                iw_path = os.path.join(sim_dir, wf)
                hole_path = os.path.join(hole_dir, f"hole_{ts}.png")
                poll_path = os.path.join(pollute_dir, f"pollute_{ts}.png")
                edit_path = os.path.join(edit_dir, f"edit_{ts}.png")
                gt_png = os.path.join(clip_dir, f"frame_{ts}.png")
                gt_jpg = os.path.join(clip_dir, f"frame_{ts}.jpg")
                gt_path = gt_png if os.path.isfile(gt_png) else gt_jpg

                if gt_path and all(os.path.isfile(p) for p in [iw_path, hole_path, poll_path, edit_path, gt_path]):
                    self.samples.append({
                        "iw": iw_path,
                        "hole": hole_path,
                        "pollute": poll_path,
                        "edit": edit_path,
                        "gt": gt_path
                    })
                    cnt += 1
                    if self.max_per_clip > 0 and cnt >= self.max_per_clip:
                        break

        if len(self.samples) == 0:
            raise RuntimeError(f"No refine samples in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]
        H, W = self.img_size

        # 读取warp与gt
        Iw = _read_rgb(r["iw"], size=(H, W))
        Gt = _read_rgb(r["gt"], size=(H, W))

        # 读取并二值化各掩码
        mh_raw = _read_mask_raw(r["hole"], size=(H, W))
        mp_raw = _read_mask_raw(r["pollute"], size=(H, W))
        me_raw = _read_mask_raw(r["edit"], size=(H, W))

        mh01 = _binarize01(mh_raw)  # 1=hole
        mp01 = _binarize01(mp_raw)  # 1=pollute band
        me01 = _binarize01(me_raw)  # 1=union标注（初始）

        # 轻度开闭平滑编辑掩码，避免噪点
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        me01 = cv2.morphologyEx(me01.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1)
        me01 = cv2.morphologyEx(me01.astype(np.uint8), cv2.MORPH_OPEN,  k, iterations=1)

        # 可选增强/扰动编辑掩码，提高泛化（仅在训练/或配置p>0时）
        if np.random.rand() < self.mask_aug_p:
            op = np.random.choice(self.mask_aug_ops)
            ksz = int(np.random.randint(self.k_min, self.k_max + 1))
            iters = int(np.random.randint(self.it_min, self.it_max + 1))
            me01 = _apply_morph(me01, op, ksz, iters)  # 保持0/1

        # 将numpy掩码转为torch张量
        mh_t = torch.from_numpy(mh01.astype(np.float32)).unsqueeze(0)
        mp_t = torch.from_numpy(mp01.astype(np.float32)).unsqueeze(0)
        me_t = torch.from_numpy(me01.astype(np.float32)).unsqueeze(0)

        # 保持 hole/pollute 与 union 一致：
        # 防止出现 union=1 但两者都=0 的情况：将 union - (mh|mp) 的部分分配到污染带
        union_cur = ((mh_t + mp_t) > 0).float()
        need_add = ((me_t > 0) & (union_cur == 0)).float()
        mp_t = torch.clamp(mp_t + need_add, 0.0, 1.0)
        # 更新 union
        mk_union = ((mh_t + mp_t) > 0).float()

        # 5通道输入
        x = torch.cat([Iw, mh_t, mp_t], dim=0)  # [3+1+1, H, W]

        return x, Gt, mk_union
