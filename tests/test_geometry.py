# depth_warp_vs/tests/test_geometry.py
import torch
from data.cameras.camera import Camera
from models.geometry.grid_builder import build_geo_grid

def test_roundtrip():
    B,H,W=1,32,48
    Ks = Kt = Camera.make_default(B,H,W).K
    Ds = torch.ones(B,1,H,W)
    dT = torch.eye(4).unsqueeze(0).repeat(B,1,1)
    grid, z = build_geo_grid(Ds, Ks, Kt, dT, H, W)
    assert grid.shape == (B,H,W,2)
