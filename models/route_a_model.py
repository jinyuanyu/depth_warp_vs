# depth_warp_vs/models/route_a_model.py
import torch
from torch import nn
from depth_warp_vs.models.splatting.softmax_splat import softmax_splat
from depth_warp_vs.models.geometry.grid_builder import build_geo_grid
from depth_warp_vs.models.warping.residual_flow_net import ResidualFlowNet
from depth_warp_vs.models.refiner.unet_light import UNetLight

class RouteAModel(nn.Module):
    """
    训练/推理路径：
      1) 前向soft splat：I_warp(带空洞)、可见性V
      2) 几何grid（可用于分析/正则）
      3) 残差小流/遮挡预测 O（仅用于细化）
      4) Refiner接收 [I_warp, V, O]，输出填补图像 It
    """
    def __init__(self, cfg):
        super().__init__()
        in_ch_res = cfg["model"]["residual_flow_net"]["in_ch"]    # 期望: Is + Ds + V
        base_ch_res = cfg["model"]["residual_flow_net"]["base_ch"]
        in_ch_ref = cfg["model"]["refiner"]["in_ch"]              # 期望: I_warp + V + O
        out_ch_ref = cfg["model"]["refiner"]["out_ch"]
        depth_ref = cfg["model"]["refiner"]["depth"]
        base_ref = cfg["model"]["refiner"]["base_ch"]
        smc = cfg["model"]["softmax_splat"]
        self.temperature = smc.get("temperature", 30.0)
        self.normalize = smc.get("normalize", True)
        self.occlusion = smc.get("occlusion", "hard")  # 新增：默认硬遮挡
        self.hard_z_epsilon = float(smc.get("hard_z_epsilon", 1e-3))
        self.align_corners = cfg["model"]["grid"]["align_corners"]
        self.flow_net = ResidualFlowNet(in_ch=in_ch_res, base_ch=base_ch_res)
        self.refiner = UNetLight(in_ch=in_ch_ref, out_ch=out_ch_ref, base_ch=base_ref, depth=depth_ref)

    def forward(self, Is, Ds, Ks, Kt, dT):
        # 前向splat（硬遮挡默认）
        I_warp, V = softmax_splat(
            Is, Ds, Ks, Kt, dT,
            temperature=self.temperature,
            normalize=self.normalize,
            occlusion=self.occlusion,
            hard_z_epsilon=self.hard_z_epsilon
        )

        # 几何grid（用于约束/调试）
        grid_geo, zt = build_geo_grid(Ds, Ks, Kt, dT, Is.shape[-2], Is.shape[-1], align_corners=self.align_corners)

        # 小流与遮挡（基于源图、深度与可见性估计微调与遮挡）
        inp = torch.cat([Is, Ds, V], dim=1)
        dgrid, O = self.flow_net(inp)

        # Refiner输入：I_warp(空洞)、V、O
        ref_in = torch.cat([I_warp, V, O], dim=1)
        It = self.refiner(ref_in)

        aux = {
            "I_warp": I_warp, "V": V, "O": O,
            "grid_geo": grid_geo, "dgrid": dgrid, "z_t": zt
        }
        return It, aux

if __name__ == "__main__":
    B,C,H,W=1,3,64,64
    Is = torch.rand(B,C,H,W)
    Ds = torch.ones(B,1,H,W)
    from data.cameras.camera import Camera
    Ks = Kt = Camera.make_default(B,H,W).K
    dT = torch.eye(4).unsqueeze(0).repeat(B,1,1)
    cfg = {
        "model": {
            "residual_flow_net": {"in_ch": 5, "base_ch": 16},
            "refiner": {"in_ch": 5, "out_ch": 3, "depth": 3, "base_ch": 16},
            "softmax_splat": {"temperature": 10.0, "normalize": True, "occlusion": "hard", "hard_z_epsilon": 1e-3},
            "grid": {"align_corners": True}
        }
    }
    net = RouteAModel(cfg)
    It, aux = net(Is, Ds, Ks, Kt, dT)
    assert It.shape == (B,3,H,W)
    print("RouteAModel self-tests passed")
