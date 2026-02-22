# depth_warp_vs/models/refiner/inpaint_refiner.py
import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        p = dilation
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, padding=p, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, padding=1)
        self.gn2 = nn.GroupNorm(8, ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(x + y)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(out_ch, dilation=1),
            ResBlock(out_ch, dilation=2),
        )
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.conv = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(out_ch, dilation=1),
        )
    def forward(self, x): return self.conv(self.up(x))

class InpaintRefiner(nn.Module):
    def __init__(self, in_ch=4, base_ch=48, depth=5, out_ch=3):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2**i) for i in range(depth)]
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 3, 1, 1),
            nn.GroupNorm(8, chs[0]),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(chs[0], dilation=1),
        )
        self.downs = nn.ModuleList()
        for i in range(depth-1):
            self.downs.append(Down(chs[i], chs[i+1]))
        self.bott = nn.Sequential(
            ResBlock(chs[-1], dilation=1),
            ResBlock(chs[-1], dilation=2),
            ResBlock(chs[-1], dilation=4),
        )
        self.ups = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.ups.append(Up(chs[i+1], chs[i]))
            self.fuse.append(nn.Sequential(
                nn.Conv2d(chs[i]*2, chs[i], 3, 1, 1),
                nn.GroupNorm(8, chs[i]),
                nn.LeakyReLU(0.1, inplace=True),
                ResBlock(chs[i], dilation=1),
            ))
        self.head = nn.Sequential(
            nn.Conv2d(chs[0], out_ch, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x4):
        # x4: [Iw(3), Mk(1)]
        skips = []
        x = self.enc0(x4); skips.append(x)
        for d in self.downs:
            x = d(x); skips.append(x)
        x = self.bott(x)
        for i in range(self.depth-1):
            x = self.ups[i](x)
            skip = skips[-(i+2)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = self.fuse[i](torch.cat([skip, x], dim=1))
        out = self.head(x)
        return out
