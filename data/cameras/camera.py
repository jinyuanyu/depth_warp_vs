# depth_warp_vs/data/cameras/camera.py
import torch
from .pose_utils import (compose, invert, identity)

class Camera:
    def __init__(self, K: torch.Tensor, T: torch.Tensor=None):
        # K: Bx3x3, T: Bx4x4 (world->cam)
        self.K = K
        if T is None:
            self.T = identity(K.shape[0], device=K.device, dtype=K.dtype)
        else:
            self.T = T

    @staticmethod
    def make_default(B: int, H: int, W: int, device=None, dtype=torch.float32, fov_deg: float=60.0):
        fx = fy = 0.5 * W / torch.tan(torch.tensor(fov_deg/2/180*3.1415926, device=device, dtype=dtype))
        cx = W/2
        cy = H/2
        K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)
        return Camera(K)

    def backproject(self, depth: torch.Tensor):
        # depth: Bx1xHxW, returns X: Bx3x(HW)
        B, _, H, W = depth.shape
        device = depth.device
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        ones = torch.ones_like(xs)
        pix = torch.stack([xs, ys, ones], dim=0).float()  # 3xHxW
        pix = pix.unsqueeze(0).repeat(B,1,1,1)  # Bx3xHxW
        Kinv = torch.inverse(self.K)
        cam = Kinv @ pix.view(B,3,-1)  # Bx3x(HW)
        z = depth.view(B,1,-1)
        X = cam * z
        return X

    def project(self, X: torch.Tensor, K: torch.Tensor=None):
        # X: Bx3xN in camera coordinates, return uv: Bx2xN, z: Bx1xN
        if K is None:
            K = self.K
        B = X.shape[0]
        z = X[:, 2:3, :]
        x = X[:, 0:1, :] / (z + 1e-8)
        y = X[:, 1:2, :] / (z + 1e-8)
        pts = torch.cat([x, y, torch.ones_like(x)], dim=1)  # Bx3xN
        uvw = K @ pts
        u = uvw[:, 0:1, :]
        v = uvw[:, 1:2, :]
        w = uvw[:, 2:3, :]
        u = u / (w + 1e-8)
        v = v / (w + 1e-8)
        return torch.cat([u, v], dim=1), z

    @staticmethod
    def to_deltaT(Ts: torch.Tensor, Tt: torch.Tensor):
        # world->cam transforms
        # X_t = ΔT @ X_s, with ΔT = Tt @ Ts^{-1}
        return Tt @ invert(Ts)

if __name__ == "__main__":
    B,H,W=1,4,5
    cam = Camera.make_default(B,H,W)
    depth = torch.ones(B,1,H,W)
    X = cam.backproject(depth)
    uv, z = cam.project(X)
    u = uv[:,0].view(B,H,W)
    v = uv[:,1].view(B,H,W)
    assert torch.allclose(u, torch.arange(W).float().view(1, 1, W).expand(B, H, W), atol=1e-3)
    assert torch.allclose(v, torch.arange(H).float().view(1, H, 1).expand(B, H, W), atol=1e-3)
    print("camera self-tests passed")
