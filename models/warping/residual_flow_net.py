# depth_warp_vs/models/warping/residual_flow_net.py
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.up(x))

class ResidualFlowNet(nn.Module):
    def __init__(self, in_ch=5, base_ch=32, max_offset_norm=0.05):
        super().__init__()
        self.max_offset_norm = max_offset_norm
        ch = base_ch
        self.enc1 = ConvBlock(in_ch, ch)
        self.enc2 = ConvBlock(ch, ch*2, s=2)
        self.enc3 = ConvBlock(ch*2, ch*4, s=2)
        self.bott = ConvBlock(ch*4, ch*4)
        self.up2 = UpBlock(ch*4, ch*2)
        self.dec2 = ConvBlock(ch*4, ch*2)
        self.up1 = UpBlock(ch*2, ch)
        self.dec1 = ConvBlock(ch*2, ch)
        self.head = nn.Conv2d(ch, 3, 3, 1, 1)  # 2 for dgrid, 1 for occlusion
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bott(e3)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.head(d1)
        dgrid = torch.tanh(out[:, :2]) * self.max_offset_norm
        O = self.sigmoid(out[:, 2:3])
        return dgrid, O

if __name__ == "__main__":
    net = ResidualFlowNet(in_ch=5, base_ch=16)
    x = torch.rand(2,5,64,64)
    dgrid, O = net(x)
    assert dgrid.shape == (2,2,64,64)
    assert O.shape == (2,1,64,64)
    print("ResidualFlowNet self-tests passed")
