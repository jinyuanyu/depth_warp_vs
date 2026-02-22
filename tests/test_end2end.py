# depth_warp_vs/tests/test_end2end.py
import torch, yaml
from models.route_a_model import RouteAModel
from data.cameras.camera import Camera

def test_end2end_identity():
    cfg = {
        "model": {
            "residual_flow_net":{"in_ch":5,"base_ch":8},
            "refiner":{"in_ch":5,"out_ch":3,"depth":2,"base_ch":8},
            "softmax_splat":{"temperature":10.0,"normalize":True},
            "grid":{"align_corners":True}
        }
    }
    B,C,H,W=1,3,64,64
    Is = torch.rand(B,C,H,W)
    Ds = torch.ones(B,1,H,W)
    Ks = Kt = Camera.make_default(B,H,W).K
    dT = torch.eye(4).unsqueeze(0).repeat(B,1,1)
    net = RouteAModel(cfg)
    It, aux = net(Is, Ds, Ks, Kt, dT)
    assert It.shape == (B,3,H,W)
