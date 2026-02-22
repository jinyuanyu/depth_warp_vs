# depth_warp_vs/models/geometry/grid_builder.py
import torch
from depth_warp_vs.data.cameras.camera import Camera
from depth_warp_vs.data.cameras.pose_utils import invert
from .projection import transform_points

def build_geo_grid(Ds: torch.Tensor, Ks: torch.Tensor, Kt: torch.Tensor, dT: torch.Tensor,
                   H: int=None, W: int=None, align_corners=True):
    # 约定：dT 为相机运动（source cam -> target cam），用于点坐标需取逆变换
    # Ds: Bx1xHxW, Ks,Kt: Bx3x3, dT: Bx4x4, return grid BxHxWx2 normalized [-1,1]
    B, _, Hs, Ws = Ds.shape
    if H is None: H = Hs
    if W is None: W = Ws
    cam_s = Camera(Ks)
    Xs = cam_s.backproject(Ds)             # Bx3x(HW)
    dT_pts = invert(dT)                    # 将camera motion转为points transform
    Xt = transform_points(Xs, dT_pts)      # Bx3x(HW)
    cam_t = Camera(Kt)
    uv, z = cam_t.project(Xt, Kt)          # Bx2x(HW)
    u = uv[:,0].view(B,Hs,Ws)
    v = uv[:,1].view(B,Hs,Ws)
    # normalize to [-1,1]
    if align_corners:
        x_norm = 2.0 * u / (W-1) - 1.0
        y_norm = 2.0 * v / (H-1) - 1.0
    else:
        x_norm = (u + 0.5) / W * 2 - 1
        y_norm = (v + 0.5) / H * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1)  # BxHxWx2
    return grid, z.view(B,1,Hs,Ws)

if __name__ == "__main__":
    B,H,W=1,8,10
    Ks = Kt = Camera.make_default(B,H,W).K
    Ds = torch.ones(B,1,H,W)
    dT = torch.eye(4).unsqueeze(0).repeat(B,1,1)
    grid, z = build_geo_grid(Ds, Ks, Kt, dT, H, W)
    assert grid.shape == (B,H,W,2)
    print("grid_builder self-tests passed")
