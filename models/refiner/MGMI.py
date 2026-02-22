# depth_warp_vs/models/refiner/MGMI.py
import torch
from torch import nn
import torch.nn.functional as F

def get_act(name="silu"):
    if name.lower() in ["silu", "swish"]:
        return nn.SiLU(inplace=True)
    elif name.lower() in ["gelu"]:
        return nn.GELU()
    else:
        return nn.ReLU(inplace=True)

class SeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act="silu", norm="bn", dilation=1):
        super().__init__()
        padding = dilation
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, padding, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        Norm = nn.BatchNorm2d if norm == "bn" else nn.GroupNorm
        self.n1 = Norm(in_ch, affine=True) if norm == "bn" else Norm(8, in_ch)
        self.n2 = Norm(out_ch, affine=True) if norm == "bn" else Norm(8, out_ch)
        self.act = get_act(act)
    def forward(self, x):
        x = self.dw(x); x = self.n1(x); x = self.act(x)
        x = self.pw(x); x = self.n2(x); x = self.act(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand=4, act="silu", norm="bn"):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        mid = int(in_ch * expand)
        Norm = nn.BatchNorm2d if norm == "bn" else nn.GroupNorm
        self.pw1 = nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False)
        self.n1  = (Norm(mid, affine=True) if norm=="bn" else Norm(8, mid))
        self.act = get_act(act)
        self.dw  = nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False)
        self.n2  = (Norm(mid, affine=True) if norm=="bn" else Norm(8, mid))
        self.pw2 = nn.Conv2d(mid, out_ch, 1, 1, 0, bias=False)
        self.n3  = (Norm(out_ch, affine=True) if norm=="bn" else Norm(8, out_ch))
    def forward(self, x):
        y = self.pw1(x); y = self.n1(y); y = self.act(y)
        y = self.dw(y);  y = self.n2(y); y = self.act(y)
        y = self.pw2(y); y = self.n3(y)
        if self.use_res:
            y = x + y
        return self.act(y)

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(8, ch // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, hid, 1, 1, 0)
        self.fc2 = nn.Conv2d(hid, ch, 1, 1, 0)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        s = self.avg(x)
        s = self.fc1(s); s = self.act(s)
        s = self.fc2(s); s = self.gate(s)
        return x * s

class DWASPP(nn.Module):
    def __init__(self, in_ch, rates=(1,2,4,8), act="silu", norm="bn", use_se=True):
        super().__init__()
        brs = []
        out_each = max(8, in_ch // len(rates))
        for r in rates:
            brs.append(SeparableConv(in_ch, out_each, stride=1, act=act, norm=norm, dilation=r))
        self.branches = nn.ModuleList(brs)
        self.fuse = nn.Conv2d(out_each*len(rates), in_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(in_ch) if norm=="bn" else nn.GroupNorm(8, in_ch)
        self.act = get_act(act)
        self.se = SEBlock(in_ch) if use_se else nn.Identity()
        self.gc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False)
        )
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        y = self.fuse(y); y = self.bn(y); y = self.act(y)
        y = self.se(y)
        g = self.gc(x)
        y = y + g
        return y

class MaskGate(nn.Module):
    def __init__(self, in_ch, mask_ch=1, act="silu", norm="bn"):
        super().__init__()
        self.mask_ch = int(mask_ch)
        self.gproj = SeparableConv(in_ch + self.mask_ch, in_ch, stride=1, act=act, norm=norm)
        self.sigmoid = nn.Sigmoid()
    def forward(self, feat, mask):
        # mask 支持多通道，若输入通道不足则做截取/补齐
        if mask is None:
            g = feat
        else:
            if mask.shape[-2:] != feat.shape[-2:]:
                mask = F.interpolate(mask, size=feat.shape[-2:], mode='bilinear', align_corners=True)
            m = mask
            if m.shape[1] > self.mask_ch:
                m = m[:, :self.mask_ch]
            elif m.shape[1] < self.mask_ch:
                # 不足则用零填充
                pad = torch.zeros(feat.shape[0], self.mask_ch - m.shape[1], feat.shape[2], feat.shape[3], device=feat.device, dtype=feat.dtype)
                m = torch.cat([m, pad], dim=1)
            g = torch.cat([feat, m], dim=1)
        g = self.gproj(g)
        g = self.sigmoid(g)
        return feat * g

class MGMI(nn.Module):
    """
    Mask-Guided Mobile Inpainter
    输入: x = [Iw(3), Mk(...)] in [0,1]，Mk可为多通道（如 hole & pollute）
    输出: Ipred(3) in [0,1]
    推理融合: Ifinal = Mk_union*Ipred + (1 - Mk_union)*Iw
    """
    def __init__(self,
                 in_ch=4,
                 out_ch=3,
                 base_ch=24,
                 width_mult=1.0,
                 depth_enc=(2,2,3),
                 expand=4,
                 act="silu",
                 norm="bn",
                 aspp_rates=(1,2,4,8),
                 use_se=True):
        super().__init__()
        wm = float(width_mult)
        bc = lambda c: max(8, int(round(c * wm)))
        self.in_ch = int(in_ch)
        self.mask_ch = max(1, self.in_ch - 3)

        # Stem
        self.stem = SeparableConv(in_ch, bc(base_ch), stride=2, act=act, norm=norm)
        self.gate_stem = MaskGate(bc(base_ch), mask_ch=self.mask_ch, act=act, norm=norm)

        # Encoder
        self.e1 = nn.Sequential(*[
            InvertedResidual(bc(base_ch if i==0 else base_ch*4/3), bc(int(base_ch*4/3)), stride=(2 if i==0 else 1), expand=expand, act=act, norm=norm)
            for i in range(depth_enc[0])
        ])
        ch_e1 = bc(int(base_ch*4/3))
        self.gate_e1 = MaskGate(ch_e1, mask_ch=self.mask_ch, act=act, norm=norm)

        self.e2 = nn.Sequential(*[
            InvertedResidual(ch_e1 if i==0 else bc(int(base_ch*2)), bc(int(base_ch*2)), stride=(2 if i==0 else 1), expand=expand, act=act, norm=norm)
            for i in range(depth_enc[1])
        ])
        ch_e2 = bc(int(base_ch*2))
        self.gate_e2 = MaskGate(ch_e2, mask_ch=self.mask_ch, act=act, norm=norm)

        ch_e3 = bc(int(base_ch*4))
        blocks_e3 = []
        in_c = ch_e2
        for i in range(depth_enc[2]):
            blocks_e3.append(InvertedResidual(in_c, ch_e3, stride=1, expand=expand, act=act, norm=norm))
            in_c = ch_e3
        self.e3 = nn.Sequential(*blocks_e3)

        self.aspp = DWASPP(ch_e3, rates=aspp_rates, act=act, norm=norm, use_se=use_se)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1_pre = SeparableConv(ch_e3, bc(int(base_ch*8/3)), stride=1, act=act, norm=norm)
        ch_d1i = bc(int(base_ch*8/3)) + ch_e2
        self.gate_d1 = MaskGate(ch_d1i, mask_ch=self.mask_ch, act=act, norm=norm)
        self.d1_fuse = SeparableConv(ch_d1i, bc(int(base_ch*2)), stride=1, act=act, norm=norm)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2_pre = SeparableConv(bc(int(base_ch*2)), bc(int(base_ch*4/3)), stride=1, act=act, norm=norm)
        ch_d2i = bc(int(base_ch*4/3)) + ch_e1
        self.gate_d2 = MaskGate(ch_d2i, mask_ch=self.mask_ch, act=act, norm=norm)
        self.d2_fuse = SeparableConv(ch_d2i, bc(base_ch), stride=1, act=act, norm=norm)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3_pre = SeparableConv(bc(base_ch), bc(int(base_ch*4/5)), stride=1, act=act, norm=norm)
        ch_d3i = bc(int(base_ch*4/5)) + bc(base_ch)
        self.gate_d3 = MaskGate(ch_d3i, mask_ch=self.mask_ch, act=act, norm=norm)
        self.d3_fuse = SeparableConv(ch_d3i, bc(int(base_ch*2/3)), stride=1, act=act, norm=norm)

        self.head1 = SeparableConv(bc(int(base_ch*2/3)), bc(int(base_ch*1/2)), stride=1, act=act, norm=norm)
        self.head2 = nn.Conv2d(bc(int(base_ch*1/2)), out_ch, 1, 1, 0)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # x: [Iw(3), Mk(...)]
        Iw = x[:, :3]
        mk = x[:, 3:] if x.shape[1] > 3 else None

        stem = self.stem(x)
        stem = self.gate_stem(stem, mk)

        e1 = self.e1(stem)
        e1 = self.gate_e1(e1, mk)

        e2 = self.e2(e1)
        e2 = self.gate_e2(e2, mk)

        e3 = self.e3(e2)
        b  = self.aspp(e3)

        y = self.up1(b)
        y = self.d1_pre(y)
        if y.shape[-2:] != e2.shape[-2:]:
            e2 = F.interpolate(e2, size=y.shape[-2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, e2], dim=1)
        y = self.gate_d1(y, mk)
        y = self.d1_fuse(y)

        y = self.up2(y)
        y = self.d2_pre(y)
        if y.shape[-2:] != e1.shape[-2:]:
            e1 = F.interpolate(e1, size=y.shape[-2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, e1], dim=1)
        y = self.gate_d2(y, mk)
        y = self.d2_fuse(y)

        y = self.up3(y)
        y = self.d3_pre(y)
        if y.shape[-2:] != stem.shape[-2:]:
            stem = F.interpolate(stem, size=y.shape[-2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, stem], dim=1)
        y = self.gate_d3(y, mk)
        y = self.d3_fuse(y)

        y = self.head1(y)
        y = self.head2(y)
        y = self.out_act(y)
        return y

if __name__ == "__main__":
    net = MGMI(in_ch=5, out_ch=3, base_ch=24, width_mult=1.0)
    x = torch.rand(1,5,256,256)
    y = net(x)
    print(y.shape)
    assert y.shape == (1,3,256,256)
    print("MGMI 5-ch test passed.")
