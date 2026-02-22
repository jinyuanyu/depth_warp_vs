# depth_warp_vs/models/losses/regularizers.py
import torch
import torch.nn.functional as F

def tv(x: torch.Tensor):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (dx.abs().mean() + dy.abs().mean())

def image_gradients(img: torch.Tensor):
    dx = img[..., :, 1:] - img[..., :, :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    # pad back to original size
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    return dx, dy

def edge_aware(flow: torch.Tensor, image: torch.Tensor):
    fx, fy = image_gradients(image)
    gx, gy = image_gradients(flow)
    w = torch.exp(-10.0 * (fx.abs().mean(1, keepdim=True) + fy.abs().mean(1, keepdim=True)))
    return (w * (gx.abs().mean(1, keepdim=True) + gy.abs().mean(1, keepdim=True))).mean()

if __name__ == "__main__":
    x = torch.rand(1,2,16,16)
    img = torch.rand(1,3,16,16)
    l_tv = tv(x)
    l_e = edge_aware(x, img)
    assert l_tv > 0 and l_e > 0
    print("regularizers self-tests passed")
