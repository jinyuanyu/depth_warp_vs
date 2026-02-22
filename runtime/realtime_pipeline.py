# depth_warp_vs/runtime/realtime_pipeline.py
import torch
from models.route_a_model import RouteAModel
from data.cameras.camera import Camera

class RealtimePipeline:
    def __init__(self, cfg):
        self.model = RouteAModel(cfg).eval().to(cfg.get("device","cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = cfg
        self.device = cfg.get("device","cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def render(self, Is, Ds, Ks=None, Kt=None, dT=None):
        B,_,H,W = Is.shape
        device = Is.device
        if Ks is None:
            Ks_t = Camera.make_default(B,H,W, device=device).K
        else:
            if Ks.ndim == 2:
                Ks = Ks.unsqueeze(0).repeat(B,1,1)
            Ks_t = Ks.to(device)
        if Kt is None:
            Kt_t = Ks_t
        else:
            if Kt.ndim == 2:
                Kt = Kt.unsqueeze(0).repeat(B,1,1)
            Kt_t = Kt.to(device)
        if dT is None:
            dT_t = torch.eye(4, device=device).unsqueeze(0).repeat(B,1,1)
        else:
            if dT.ndim == 2:
                dT = dT.unsqueeze(0).repeat(B,1,1)
            dT_t = dT.to(device)
        It, aux = self.model(Is, Ds, Ks_t, Kt_t, dT_t)
        return It, aux

if __name__ == "__main__":
    cfg = {
        "device":"cpu",
        "model": {
            "residual_flow_net":{"in_ch":5,"base_ch":16},
            "refiner":{"in_ch":5,"out_ch":3,"depth":3,"base_ch":16},
            "softmax_splat":{"temperature":10.0,"normalize":True},
            "grid":{"align_corners":True}
        }
    }
    pipe = RealtimePipeline(cfg)
    Is = torch.rand(1,3,64,64)
    Ds = torch.ones(1,1,64,64)
    It, aux = pipe.render(Is, Ds)
    print("realtime_pipeline self-tests passed")
