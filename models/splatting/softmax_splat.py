# depth_warp_vs/models/splatting/softmax_splat.py
import torch
from depth_warp_vs.data.cameras.camera import Camera
from depth_warp_vs.data.cameras.pose_utils import invert
from depth_warp_vs.models.geometry.projection import transform_points

def _gather_neighbors(u, v, H, W):
    # u, v: BxN
    u0 = torch.floor(u); v0 = torch.floor(v)
    u1 = u0 + 1; v1 = v0 + 1
    fu = u - u0; fv = v - v0
    w00 = (1 - fu) * (1 - fv)
    w01 = (1 - fu) * fv
    w10 = fu * (1 - fv)
    w11 = fu * fv
    def valid_mask(uu, vv):
        return (uu >= 0) & (uu <= (W - 1)) & (vv >= 0) & (vv <= (H - 1))
    m00 = valid_mask(u0, v0)
    m01 = valid_mask(u0, v1)
    m10 = valid_mask(u1, v0)
    m11 = valid_mask(u1, v1)
    u0c = u0.clamp(0, W - 1).long()
    v0c = v0.clamp(0, H - 1).long()
    u1c = u1.clamp(0, W - 1).long()
    v1c = v1.clamp(0, H - 1).long()
    return (u0c, v0c, u1c, v1c), (w00, w01, w10, w11), (m00, m01, m10, m11)

def softmax_splat(Is: torch.Tensor, Ds: torch.Tensor, Ks: torch.Tensor, Kt: torch.Tensor, dT: torch.Tensor,
                  temperature: float=30.0, normalize: bool=True,
                  occlusion: str="hard", hard_z_epsilon: float=1e-3):
    """
    Is: BxCxHxW
    Ds: Bx1xHxW
    Ks/Kt: Bx3x3
    dT: Bx4x4 (camera motion)
    返回:
      I0: BxCxHxW
      V:  Bx1xHxW
    """
    assert occlusion in ("hard", "soft")
    B, C, H, W = Is.shape
    device = Is.device
    dtype_I = Is.dtype
    N = H * W

    cam_s = Camera(Ks)
    Xs = cam_s.backproject(Ds)                 # Bx3x(HW)
    dT_pts = invert(dT)                        # camera motion -> points transform
    Xt = transform_points(Xs, dT_pts)          # Bx3x(HW)
    cam_t = Camera(Kt)
    uv, z = cam_t.project(Xt, Kt)              # uv: Bx2x(HW), z: Bx1x(HW)
    z = z.view(B, 1, N)
    z_clamped = torch.clamp(z, min=1e-6)
    dtype_z = z_clamped.dtype

    u = uv[:, 0]  # BxN
    v = uv[:, 1]  # BxN

    (u0c, v0c, u1c, v1c), (w00, w01, w10, w11), (m00, m01, m10, m11) = _gather_neighbors(u, v, H, W)

    # 预先构造所有4邻域的像素索引、权重与mask
    idx00 = (v0c * W + u0c).view(B, N)
    idx01 = (v1c * W + u0c).view(B, N)
    idx10 = (v0c * W + u1c).view(B, N)
    idx11 = (v1c * W + u1c).view(B, N)
    idx_all = torch.cat([idx00, idx01, idx10, idx11], dim=1)        # Bx(4N) long

    m_all = torch.cat([m00.view(B, N), m01.view(B, N),
                       m10.view(B, N), m11.view(B, N)], dim=1)      # Bx(4N) bool

    w_lin_all = torch.cat([w00.view(B, N), w01.view(B, N),
                           w10.view(B, N), w11.view(B, N)], dim=1)  # Bx(4N) float

    # 将batch维度扁平化到索引空间，避免for-batch循环
    # 目标平面大小: B * (H*W)
    base_offsets = (torch.arange(B, device=device, dtype=idx_all.dtype) * N).unsqueeze(1)  # Bx1
    idx_flat = (idx_all + base_offsets).reshape(-1)  # (B*4N) long

    # 初始化输出累加器（扁平化版本）
    numer_flat = torch.zeros(C, B * N, device=device, dtype=dtype_I)
    denom_flat = torch.zeros(1, B * N, device=device, dtype=dtype_I)

    # 预备源图数据(重复4次对应4邻域)
    Is_flat = Is.view(B, C, N)                  # BxCxN
    Is_rep = Is_flat.repeat(1, 1, 4)            # BxCx(4N)
    Is_rep_CK = Is_rep.permute(1, 0, 2).reshape(C, -1)  # C x (B*4N)

    # 预备mask/权重
    m_all_flat = m_all.reshape(-1)  # (B*4N)
    w_lin_all_I_flat = w_lin_all.reshape(-1).to(dtype_I)  # (B*4N) -> 图像dtype

    if occlusion == "soft":
        # 软遮挡：exp(-z * T) * 双线性权重
        w_depth = torch.exp(-z_clamped * float(temperature))              # Bx1xN
        w_depth_all = w_depth.repeat(1, 1, 4).reshape(B, 1, 4 * N)        # Bx1x(4N)
        w_all = (w_lin_all.view(B, 1, 4 * N) * w_depth_all)               # Bx1x(4N)
        w_all = (w_all * m_all.view(B, 1, 4 * N)).to(dtype_I)             # Bx1x(4N) -> 图像dtype
        w_all_flat = w_all.reshape(1, -1)                                 # 1 x (B*4N)

        # 一次性向量化累加
        numer_flat.index_add_(1, idx_flat, Is_rep_CK * w_all_flat)
        denom_flat.index_add_(1, idx_flat, w_all_flat)

    else:
        # 硬遮挡：z-buffer 最小z (批量扁平化 scatter-amin)
        z_all = z_clamped.view(B, N).repeat(1, 4)        # Bx(4N), dtype_z
        z_all_flat = z_all.reshape(-1)                   # (B*4N)
        inf_flat = torch.full_like(z_all_flat, float("inf"))

        # 仅对有效位置参与z比较
        zmask_flat = torch.where(m_all_flat, z_all_flat, inf_flat)

        zbuf_flat = torch.full((B * N,), float("inf"), device=device, dtype=dtype_z)
        has_scatter_reduce = hasattr(torch.Tensor, "scatter_reduce_")
        fallback_to_soft = False

        if has_scatter_reduce:
            # amin归约，include_self=True 表示与初始化inf共同取min
            zbuf_flat.scatter_reduce_(0, idx_flat, zmask_flat, reduce="amin", include_self=True)
        else:
            try:
                import torch_scatter
                # torch_scatter.scatter 支持 reduce="min"
                zbuf_flat = torch_scatter.scatter(zmask_flat, idx_flat, dim=0, reduce="min", out=zbuf_flat)
            except Exception:
                # 回退soft
                fallback_to_soft = True

        if fallback_to_soft:
            # 回退到软遮挡路径（向量化）
            w_depth = torch.exp(-z_clamped * float(temperature))              # Bx1xN
            w_depth_all = w_depth.repeat(1, 1, 4).reshape(B, 1, 4 * N)        # Bx1x(4N)
            w_all = (w_lin_all.view(B, 1, 4 * N) * w_depth_all)               # Bx1x(4N)
            w_all = (w_all * m_all.view(B, 1, 4 * N)).to(dtype_I)             # -> 图像dtype
            w_all_flat = w_all.reshape(1, -1)                                 # 1 x (B*4N)

            numer_flat.index_add_(1, idx_flat, Is_rep_CK * w_all_flat)
            denom_flat.index_add_(1, idx_flat, w_all_flat)
        else:
            # 硬遮挡前景筛选 + 向量化累加
            zthr_flat = zbuf_flat.index_select(0, idx_flat) + float(hard_z_epsilon)  # (B*4N)
            front_mask_flat = m_all_flat & (z_all_flat <= zthr_flat)                  # (B*4N)
            wv_flat = torch.where(front_mask_flat, w_lin_all_I_flat, torch.zeros_like(w_lin_all_I_flat))
            wv_flat_1 = wv_flat.unsqueeze(0)  # 1 x (B*4N)

            numer_flat.index_add_(1, idx_flat, Is_rep_CK * wv_flat_1)
            denom_flat.index_add_(1, idx_flat, wv_flat_1)

    # 还原为(B, C/HW)形状
    numer = numer_flat.view(C, B, N).permute(1, 0, 2).view(B, C, H, W)
    denom = denom_flat.view(1, B, N).permute(1, 0, 2).view(B, 1, H, W)

    I0 = numer / (denom + 1e-8) if normalize else numer
    V = denom
    return I0, V

if __name__ == "__main__":
    B,C,H,W=1,3,16,16
    Is = torch.rand(B,C,H,W, requires_grad=True)
    Ds = torch.ones(B,1,H,W)
    Ks = Kt = Camera.make_default(B,H,W).K
    dT = torch.eye(4).unsqueeze(0).repeat(B,1,1)
    I0, V = softmax_splat(Is, Ds, Ks, Kt, dT, temperature=10.0, normalize=True, occlusion="hard")
    loss = (I0.mean() + V.mean())
    loss.backward()
    print("softmax_splat optimized dtype-safe passed")
