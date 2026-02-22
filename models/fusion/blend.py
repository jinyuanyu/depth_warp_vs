# depth_warp_vs/models/fusion/blend.py
import torch

def fuse(I0: torch.Tensor, I1: torch.Tensor, V: torch.Tensor, O: torch.Tensor, mode="weighted"):
    eps = 1e-6
    if mode == "weighted":
        V = V.clamp(0, 1)
        O = O.clamp(0, 1)
        w0 = V
        w1 = (1.0 - O) * (1.0 - V) # 关键改动：互补权重，避免双满平均
        num = I0 * w0 + I1 * w1
        den = w0 + w1 + eps
        return num / den
    elif mode == "maxV":
        mask = (V > 0.5).float()
        return I0 * mask + I1 * (1 - mask)
    else:
        return 0.5 * I0 + 0.5 * I1

if __name__ == "__main__":
    I0 = torch.ones(1,3,4,4)
    I1 = torch.zeros(1,3,4,4)
    V = torch.ones(1,1,4,4)
    O = torch.zeros(1,1,4,4)
    If = fuse(I0,I1,V,O, mode="weighted")
    assert torch.allclose(If, torch.ones_like(If))
    print("blend self-tests passed")
