# depth_warp_vs/models/refiner/__init__.py
from typing import Dict
from torch import nn
from .inpaint_refiner import InpaintRefiner
from .MGMI import MGMI

def build_refiner(cfg: Dict) -> nn.Module:
    """
    cfg 结构示例：
    model:
      refiner:
        type: "MGMI" | "InpaintRefiner"
        in_ch: 4
        out_ch: 3
        base_ch: 48
        depth: 5          # InpaintRefiner使用
        width_mult: 1.0   # MGMI使用
        act: "silu"
        norm: "bn"
        aspp_rates: [1,2,4,8]
        use_se: true
    """
    rcfg = cfg.get("model", {}).get("refiner", {})
    rtype = str(rcfg.get("type", "InpaintRefiner")).lower()
    if rtype == "mgmi":
        return MGMI(
            in_ch=rcfg.get("in_ch", 4),
            out_ch=rcfg.get("out_ch", 3),
            base_ch=rcfg.get("base_ch", 24),
            width_mult=float(rcfg.get("width_mult", 1.0)),
            depth_enc=tuple(rcfg.get("depth_enc", (2,2,3))),
            expand=int(rcfg.get("expand", 4)),
            act=rcfg.get("act", "silu"),
            norm=rcfg.get("norm", "bn"),
            aspp_rates=tuple(rcfg.get("aspp_rates", (1,2,4,8))),
            use_se=bool(rcfg.get("use_se", True))
        )
    else:
        return InpaintRefiner(
            in_ch=rcfg.get("in_ch", 4),
            base_ch=rcfg.get("base_ch", 48),
            depth=rcfg.get("depth", 5),
            out_ch=rcfg.get("out_ch", 3)
        )
