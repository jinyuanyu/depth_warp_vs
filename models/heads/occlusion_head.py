# depth_warp_vs/models/heads/occlusion_head.py
import torch
from torch import nn

class OcclusionHead(nn.Module):
    def __init__(self, in_ch, base_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.net(x))

if __name__ == "__main__":
    net = OcclusionHead(6)
    x = torch.rand(1,6,32,32)
    y = net(x)
    assert y.shape == (1,1,32,32)
    print("occlusion_head self-tests passed")
