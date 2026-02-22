# depth_warp_vs/models/geometry/projection.py
import torch

def transform_points(X: torch.Tensor, T: torch.Tensor):
    # X: Bx3xN in source cam; T: Bx4x4 cam_s -> cam_t
    B, _, N = X.shape
    hom = torch.cat([X, torch.ones(B,1,N, device=X.device, dtype=X.dtype)], dim=1) # Bx4xN
    Xt = T @ hom  # Bx4xN
    Xt = Xt[:, :3, :]
    return Xt

if __name__ == "__main__":
    B=1; N=10
    X = torch.rand(B,3,N)
    T = torch.eye(4).unsqueeze(0).repeat(B,1,1)
    Xt = transform_points(X,T)
    assert torch.allclose(Xt,X)
    print("projection self-tests passed")
