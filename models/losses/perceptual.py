# depth_warp_vs/models/losses/perceptual.py
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptual(nn.Module):
    def __init__(self, layers=(3,8,15), weights='DEFAULT'):
        super().__init__()
        if weights == 'DEFAULT':
            net = vgg16(weights=VGG16_Weights.DEFAULT).features
        else:
            net = vgg16(weights=None).features
        self.slices = nn.ModuleList()
        last = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[net[i] for i in range(last, l)]))
            last = l
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y, mask=None):
        def _f(feat_x, feat_y):
            if mask is not None:
                m = torch.nn.functional.interpolate(mask, size=feat_x.shape[-2:], mode='bilinear', align_corners=True)
                l = (feat_x - feat_y).abs() * m
                denom = torch.clamp(m.sum() * feat_x.shape[1], min=1.0)
                return l.sum() / denom
            else:
                return (feat_x - feat_y).abs().mean()

        loss = 0.0
        xv = x
        yv = y
        for s in self.slices:
            xv = s(xv)
            yv = s(yv)
            loss = loss + _f(xv, yv)
        return loss

if __name__ == "__main__":
    net = VGGPerceptual()
    x = torch.rand(1,3,64,64)
    y = torch.rand(1,3,64,64)
    m = torch.ones(1,1,64,64)
    l = net(x,y,m)
    assert l > 0
    print("perceptual self-tests passed")
