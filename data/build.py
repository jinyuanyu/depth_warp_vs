# depth_warp_vs/data/build.py
import torch
from torch.utils.data import DataLoader

def build_dataset(cfg, split="train"):
    name = cfg["data"].get("train_dataset", "mannequin").lower()
    if name in ["mannequin", "mannequinchallenge", "mc"]:
        from data.mannequin_dataset import MannequinChallengeDataset
        return MannequinChallengeDataset(cfg, split=split)
    elif name in ["mannequin_refine", "mc_refine", "refine"]:
        from depth_warp_vs.data.mannequin_refine_dataset import MannequinRefineDataset
        return MannequinRefineDataset(cfg, split=split)
    elif name in ["dummy", "video_selfsup"]:
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, length=1000, H=256, W=256):
                self.length = length
                self.H, self.W = H,W
            def __len__(self): return self.length
            def __getitem__(self, idx):
                Is = torch.rand(3, self.H, self.W)
                Ds = torch.rand(1, self.H, self.W)*10.0 + 1e-3
                K = torch.tensor([[500.0,0,self.W/2],[0,500.0,self.H/2],[0,0,1.0]], dtype=torch.float32)
                dT = torch.eye(4)
                It_gt = Is.clone()
                return Is, Ds, K, K, dT, It_gt
        H, W = cfg["data"]["img_size"]
        return DummyDataset(H=H, W=W)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def build_dataloader(cfg, split="train"):
    ds = build_dataset(cfg, split=split)
    # refine-only 数据集返回 (x, gt, mask)，其余数据集返回六元组；
    bs = cfg["data"].get("batch_size", 4) if split=="train" else 1
    num_workers = cfg["data"].get("workers", 4)
    shuffle = split=="train"
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle, pin_memory=True)
    return dl
