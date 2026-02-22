# depth_warp_vs/tests/test_warp.py
import torch
from models.warping.warp import safe_grid_sample
from models.utils.misc import make_coords_grid

def test_identity_warp():
    B,C,H,W=1,3,16,16
    img = torch.rand(B,C,H,W)
    grid = make_coords_grid(B,H,W)
    out = safe_grid_sample(img, grid)
    assert torch.allclose(out, img, atol=1e-4)
