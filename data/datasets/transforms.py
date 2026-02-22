# depth_warp_vs/data/datasets/transforms.py
import torch
import torchvision.transforms.functional as TF
import random

class ToTensor:
    def __call__(self, sample):
        # sample: dict with 'Is','It_gt' as HxWx3 uint8, 'Ds' HxW float32, Ks,Kt 3x3 float, dT 4x4 float
        Is = TF.to_tensor(sample["Is"])
        It_gt = TF.to_tensor(sample["It_gt"]) if "It_gt" in sample else None
        Ds = torch.from_numpy(sample["Ds"]).unsqueeze(0).float()
        Ks = torch.from_numpy(sample["Ks"]).float()
        Kt = torch.from_numpy(sample["Kt"]).float()
        dT = torch.from_numpy(sample["dT"]).float()
        out = dict(Is=Is, Ds=Ds, Ks=Ks, Kt=Kt, dT=dT)
        if It_gt is not None: out["It_gt"] = It_gt
        if "mask" in sample: out["mask"] = torch.from_numpy(sample["mask"]).unsqueeze(0).float()
        return out

class Normalize:
    def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        sample["Is"] = TF.normalize(sample["Is"], self.mean, self.std)
        if "It_gt" in sample:
            sample["It_gt"] = TF.normalize(sample["It_gt"], self.mean, self.std)
        return sample

class RandomColorJitter:
    def __init__(self, p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        import torchvision.transforms as T
        self.p = p
        self.jit = T.ColorJitter(brightness, contrast, saturation, hue)
    def __call__(self, sample):
        if random.random() < self.p:
            sample["Is"] = self.jit(sample["Is"])
            if "It_gt" in sample:
                sample["It_gt"] = self.jit(sample["It_gt"])
        return sample
