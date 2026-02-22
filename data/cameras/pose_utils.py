# depth_warp_vs/data/cameras/pose_utils.py
import torch

def invert(T: torch.Tensor) -> torch.Tensor:
    # T: (..., 4,4)
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    tinv = -Rt @ t
    Tout = torch.eye(4, device=T.device, dtype=T.dtype).expand_as(T).clone()
    Tout[..., :3, :3] = Rt
    Tout[..., :3, 3:4] = tinv
    return Tout

def compose(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    return T1 @ T2

def se3_from_rt(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    B = R.shape[0]
    T = torch.eye(4, device=R.device, dtype=R.dtype).repeat(B,1,1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t.view(B,3)
    return T

def identity(B: int, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)

if __name__ == "__main__":
    B=2
    R = torch.eye(3).unsqueeze(0).repeat(B,1,1)
    t = torch.tensor([[1.,0,0],[0,1.,0]])
    T = se3_from_rt(R,t)
    I = identity(B)
    Tinvt = invert(T)
    assert torch.allclose(compose(T, Tinvt), I, atol=1e-5)
    print("pose_utils self-tests passed")
