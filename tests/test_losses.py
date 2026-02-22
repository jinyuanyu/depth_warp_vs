# depth_warp_vs/tests/test_losses.py
import torch
from models.losses.photometric import recon_loss
from models.losses.regularizers import tv

def test_losses_basic():
    a = torch.ones(1,3,8,8)
    b = torch.ones(1,3,8,8)*0.9
    m = torch.ones(1,1,8,8)
    l = recon_loss(a,b,m)
    assert l > 0
    x = torch.rand(1,2,8,8)
    assert tv(x) > 0
