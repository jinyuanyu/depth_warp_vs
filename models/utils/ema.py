# depth_warp_vs/models/utils/ema.py
import torch
from copy import deepcopy

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model):
        # 正确的EMA更新：ema = decay*ema + (1-decay)*model
        ema_sd = self.ema.state_dict()
        msd = model.state_dict()
        d = self.decay
        for k in ema_sd.keys():
            if k in msd:
                ema_v = ema_sd[k]
                mdl_v = msd[k].detach()
                # 类型/设备对齐
                if ema_v.dtype != mdl_v.dtype:
                    mdl_v = mdl_v.to(dtype=ema_v.dtype)
                if ema_v.device != mdl_v.device:
                    mdl_v = mdl_v.to(device=ema_v.device)
                ema_sd[k].copy_(ema_v * d + mdl_v * (1.0 - d))
