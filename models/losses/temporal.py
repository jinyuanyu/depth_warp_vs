# depth_warp_vs/models/losses/temporal.py
import torch

def temporal_consistency(It_t: torch.Tensor, It_t1: torch.Tensor, mask_dyn=None):
    diff = (It_t - It_t1).abs()
    if mask_dyn is not None:
        diff = diff * mask_dyn
        denom = mask_dyn.sum() * It_t.shape[1]
        denom = torch.clamp(denom, min=1.0)
        return diff.sum() / denom
    return diff.mean()

if __name__ == "__main__":
    a = torch.zeros(1,3,8,8)
    b = torch.zeros(1,3,8,8)
    assert temporal_consistency(a,b) == 0
    print("temporal self-tests passed")
