# depth_warp_vs/engine/metrics.py
import torch
import math

def psnr(pred, tgt):
    # pred/tgt: torch in [0,1]
    mse = torch.mean((pred - tgt)**2).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)

def ssim_simple(pred, tgt):
    # 保留原简易版本（与亮度均值差相关，仅用于占位）
    mu_x = pred.mean().item()
    mu_y = tgt.mean().item()
    return 1.0 - abs(mu_x - mu_y)

def ssim_index(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    """
    使用已有 photometric.ssim（返回的是“dissimilarity map”=(1-SSIM)/2），
    将其还原为 SSIM 指数（约在[0,1]，数值越大越好）。
    """
    from depth_warp_vs.models.losses.photometric import ssim as ssim_dissim_map
    with torch.no_grad():
        dmap = ssim_dissim_map(pred, tgt)  # [B, C, H, W] dissimilarity in [0,1]
        d = dmap.mean().item()
        # SSIM ≈ 1 - 2*d（理论上SSIM∈[-1,1]，此处近似到[0,1]区间）
        ssim_val = max(0.0, min(1.0, 1.0 - 2.0 * d))
        return ssim_val

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None):
    """
    返回常用评估指标：
      - l1: 全图或按mask加权的L1
      - psnr: 全图PSNR
      - ssim: 近似SSIM指数（[0,1]）
    pred/gt in [0,1], shape Bx3xHxW, mask Bx1xHxW (1=valid)
    """
    with torch.no_grad():
        if mask is not None:
            denom = torch.clamp(mask.sum() * pred.shape[1], min=1.0)
            l1 = torch.sum(torch.abs(pred - gt) * mask) / denom
        else:
            l1 = torch.mean(torch.abs(pred - gt))
        metrics = {
            "l1": float(l1.item()),
            "psnr": float(psnr(pred, gt)),
            "ssim": float(ssim_index(pred, gt)),
        }
        return metrics

if __name__ == "__main__":
    a = torch.zeros(1,3,8,8)
    b = torch.zeros(1,3,8,8)
    assert psnr(a,b) > 90
    s = ssim_index(a,b)
    assert s >= 0.99
    print("metrics self-tests passed")
