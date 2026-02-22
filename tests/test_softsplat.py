# depth_warp_vs/tests/test_softsplat.py
import torch
from models.splatting.softmax_splat import softmax_splat
from data.cameras.camera import Camera

def test_softsplat_grad():
    B,C,H,W=1,3,32,32
    Is = torch.rand(B,C,H,W, requires_grad=True)
    Ds = torch.ones(B,1,H,W)
    Ks = Kt = Camera.make_default(B,H,W).K
    dT = torch.eye(4).unsqueeze(0).repeat(B,1,1)
    I0, V = softmax_splat(Is, Ds, Ks, Kt, dT, 10.0, True)
    loss = (I0.mean() + V.mean())
    loss.backward()
    assert Is.grad is not None
