# depth_warp_vs/models/refiner/unet_light.py
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNetLight(nn.Module):
    def __init__(self, in_ch, out_ch=3, base_ch=32, depth=4):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            self.downs.append(DoubleConv(ch, base_ch*(2**i)))
            self.pools.append(nn.MaxPool2d(2))
            ch = base_ch*(2**i)
        self.bott = DoubleConv(ch, ch*2)
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        in_up = ch*2
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(in_up, base_ch*(2**i), 2, 2))
            self.convs.append(DoubleConv(base_ch*(2**i)*2, base_ch*(2**i)))
            in_up = base_ch*(2**i)
        self.head = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        skips = []
        out = x
        for i in range(self.depth):
            out = self.downs[i](out)
            skips.append(out)
            out = self.pools[i](out)
        out = self.bott(out)
        for i in range(self.depth):
            out = self.ups[i](out)
            skip = skips[-(i+1)]
            if out.shape[-2:] != skip.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            out = self.convs[i](torch.cat([skip, out], dim=1))
        out = self.head(out)
        out = torch.sigmoid(out)  # final RGB in [0,1]
        return out

if __name__ == "__main__":
    net = UNetLight(in_ch=5, out_ch=3, base_ch=16, depth=3)
    x = torch.rand(1,5,64,64)
    y = net(x)
    assert y.shape == (1,3,64,64)
    print("UNetLight self-tests passed")
