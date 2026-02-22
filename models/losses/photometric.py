# depth_warp_vs/models/losses/photometric.py
import torch
import torch.nn.functional as F

def l1_loss(pred, tgt, mask=None, reduce=True):
    l = torch.abs(pred - tgt)
    if mask is not None:
        l = l * mask
        denom = mask.sum() * pred.shape[1]
        denom = torch.clamp(denom, min=1.0)
        l = l.sum() / denom
        return l
    return l.mean() if reduce else l

def ssim(pred, tgt, C1=0.01**2, C2=0.03**2):
    # channel-wise SSIM, average over channels
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(tgt, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred*pred, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(tgt*tgt, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(pred*tgt, 3, 1, 1) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return torch.clamp((1 - ssim_map)/2, 0, 1)

def recon_loss(pred, tgt, mask=None, lam_l1=1.0, lam_ssim=0.2):
    l1 = l1_loss(pred, tgt, mask)
    s = ssim(pred, tgt)
    if mask is not None:
        s = (s * mask).sum() / torch.clamp(mask.sum() * pred.shape[1], min=1.0)
    else:
        s = s.mean()
    return lam_l1 * l1 + lam_ssim * s

if __name__ == "__main__":
    a = torch.ones(1,3,8,8)
    b = torch.ones(1,3,8,8)*0.9
    m = torch.ones(1,1,8,8)
    l = recon_loss(a, b, mask=m)
    assert l > 0
    print("photometric self-tests passed")
