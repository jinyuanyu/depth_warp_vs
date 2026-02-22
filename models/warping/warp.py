# depth_warp_vs/models/warping/warp.py
import torch
import torch.nn.functional as F

def safe_grid_sample(img: torch.Tensor, grid: torch.Tensor, align_corners=True, mode='bilinear', padding_mode='zeros'):
    return F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

if __name__ == "__main__":
    B,C,H,W=1,3,8,8
    img = torch.rand(B,C,H,W)
    yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    out = safe_grid_sample(img, grid)
    assert torch.allclose(out, img, atol=1e-4)
    print("warp self-tests passed")
