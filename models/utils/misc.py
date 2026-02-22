# depth_warp_vs/models/utils/misc.py
import torch

def make_coords_grid(B, H, W, device=None):
    ys, xs = torch.meshgrid(torch.linspace(-1,1,H, device=device), torch.linspace(-1,1,W, device=device), indexing='ij')
    grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).repeat(B,1,1,1)
    return grid

def pad_to_divisor(x: torch.Tensor, div=32):
    H, W = x.shape[-2:]
    Hn = (H + div - 1) // div * div
    Wn = (W + div - 1) // div * div
    pad_h = Hn - H
    pad_w = Wn - W
    xpad = torch.nn.functional.pad(x, (0,pad_w,0,pad_h))
    return xpad, (pad_h, pad_w)

def unpad(x: torch.Tensor, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h>0:
        x = x[..., :-pad_h, :]
    if pad_w>0:
        x = x[..., :, :-pad_w]
    return x

if __name__ == "__main__":
    x = torch.rand(1,3,33,65)
    y, pad = pad_to_divisor(x, 16)
    z = unpad(y, pad)
    assert z.shape == x.shape
    print("misc self-tests passed")
