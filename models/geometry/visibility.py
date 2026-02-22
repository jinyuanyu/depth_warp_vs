# depth_warp_vs/models/geometry/visibility.py
import torch

def depth_confidence(depth: torch.Tensor, min_z: float=0.1, max_z: float=100.0):
    # depth: Bx1xHxW
    valid = (depth > min_z) & (depth < max_z)
    conf = torch.zeros_like(depth)
    conf[valid] = 1.0
    return conf

def soft_visibility_from_z(z_t: torch.Tensor, temperature: float=30.0):
    # z_t: Bx1xHxW depth in target; nearer has higher weight
    w = torch.exp(-torch.clamp(z_t, min=1e-3) * temperature)
    return w

if __name__ == "__main__":
    z = torch.tensor([[[[1.,2.],[3.,4.]]]])
    w = soft_visibility_from_z(z, temperature=1.0)
    assert torch.all(w[:, :, 0, 0] > w[:, :, 1, 1])
    print("visibility self-tests passed")
