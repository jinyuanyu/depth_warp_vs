import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 从主代码导入必要的函数和类
from main import (
    load_rgb, load_depth, build_convergent_camera_motion,
    softmax_splat, default_K, resize_intrinsics
)

def compute_softmax_splat_mask(Is, Ds, Ks, Kt, dT, temperature=0.1, occlusion=True):
    """计算softmax splatting掩码（原方法）"""
    with torch.inference_mode():
        _, V = softmax_splat(
            Is, Ds, Ks, Kt, dT,
            temperature=temperature,
            normalize=True,
            occlusion=occlusion,
            hard_z_epsilon=1e-4
        )
    # 当可见性权重低于阈值时视为空洞
    return (V <= 1e-8).float()

def compute_forward_warp_occupancy_mask(Is, Ds, Ks, Kt, dT):
    """计算前向warp占用掩码"""
    B, C, H, W = Is.shape
    N = dT.shape[0]
    
    # 创建一个计数器，记录每个像素被多少个源视图覆盖
    occupancy = torch.zeros(B, 1, H, W, device=Is.device)
    
    with torch.inference_mode():
        for i in range(N):
            # 对每个源视图单独进行warp
            _, V_i = softmax_splat(
                Is[i:i+1], Ds[i:i+1], Ks[i:i+1], Kt[i:i+1], dT[i:i+1],
                temperature=0.1,
                normalize=True,
                occlusion=True,
                hard_z_epsilon=1e-4
            )
            # 只要有覆盖就计数
            occupancy += (V_i > 1e-8).float()
    
    # 没有被任何源视图覆盖的像素视为空洞
    return (occupancy == 0).float()

def compute_ground_truth_hole_mask(Is, Ds, Ks, Kt, dT, depth_threshold=0.01):
    """通过反投影+深度测试计算真实空洞掩码"""
    B, C, H, W = Is.shape
    N = dT.shape[0]
    device = Is.device
    
    # 生成目标图像的像素坐标网格
    y, x = torch.meshgrid(torch.arange(H, device=device), 
                          torch.arange(W, device=device), 
                          indexing='ij')
    x = x.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    y = y.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 齐次坐标
    ones = torch.ones_like(x)
    pixels = torch.cat([x, y, ones], dim=1)  # [1, 3, H, W]
    
    # 目标相机内参的逆
    Kt_inv = torch.inverse(Kt)
    
    # 初始化为全部是空洞
    ground_truth_mask = torch.ones(B, 1, H, W, device=device)
    
    with torch.inference_mode():
        for i in range(N):
            # 源相机到目标相机的变换
            dT_i = dT[i:i+1]  # [1, 4, 4]
            R = dT_i[:, :3, :3]  # 旋转
            t = dT_i[:, :3, 3:4]  # 平移
            
            # 目标相机内参逆矩阵
            Kt_inv_i = Kt_inv[i:i+1]  # [1, 3, 3]
            
            # 将目标像素反投影到目标相机坐标系的3D点
            pixels_normalized = torch.matmul(Kt_inv_i, pixels.reshape(B, 3, -1))  # [B, 3, H*W]
            
            # 假设目标视图的深度为平均深度
            target_depth = Ds.mean()
            Xt = pixels_normalized * target_depth  # [B, 3, H*W]
            
            # 转换到世界坐标系
            Xt_hom = torch.cat([Xt, torch.ones(B, 1, H*W, device=device)], dim=1)  # [B, 4, H*W]
            Xw = torch.matmul(dT_i, Xt_hom)  # [B, 4, H*W]
            Xw = Xw[:, :3, :]  # [B, 3, H*W]
            
            # 转换到源相机坐标系
            Xs = torch.matmul(R.transpose(1, 2), Xw - t)  # [B, 3, H*W]
            
            # 投影到源图像平面
            Ks_i = Ks[i:i+1]  # [1, 3, 3]
            Xs_proj = torch.matmul(Ks_i, Xs)  # [B, 3, H*W]
            zs = Xs_proj[:, 2:3, :]  # 深度
            xs = Xs_proj[:, 0:1, :] / (zs + 1e-8)  # x坐标
            ys = Xs_proj[:, 1:2, :] / (zs + 1e-8)  # y坐标
            
            # 检查是否在源图像范围内
            valid_x = (xs >= 0) & (xs < W)
            valid_y = (ys >= 0) & (ys < H)
            valid = valid_x & valid_y & (zs > 1e-4)
            
            # 双线性采样源深度图
            xs_norm = (xs / W) * 2 - 1
            ys_norm = (ys / H) * 2 - 1
            grid = torch.cat([xs_norm.transpose(1, 2), ys_norm.transpose(1, 2)], dim=2)  # [B, H*W, 2]
            grid = grid.reshape(B, H, W, 2)  # [B, H, W, 2]
            
            Ds_i = Ds[i:i+1]  # [1, 1, H, W]
            sampled_depth = torch.nn.functional.grid_sample(
                Ds_i, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )  # [B, 1, H, W]
            
            # 深度一致性检查
            zs_reshaped = zs.reshape(B, 1, H, W)
            depth_diff = torch.abs(zs_reshaped - sampled_depth) / (zs_reshaped + 1e-8)
            depth_consistent = (depth_diff < depth_threshold)
            
            # 更新掩码：深度一致且在范围内的像素不是空洞
            valid_reshaped = valid.reshape(B, 1, H, W)
            ground_truth_mask = ground_truth_mask * (1 - (valid_reshaped & depth_consistent).float())
    
    return ground_truth_mask

def visualize_mask_comparison(rgb, masks, mask_names, save_path=None):
    """可视化比较不同的掩码"""
    # 将张量转换为numpy数组用于可视化
    if isinstance(rgb, torch.Tensor):
        rgb_np = (rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    else:
        rgb_np = rgb
    
    mask_np_list = []
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask
        mask_np_list.append(mask_np)
    
    # 创建可视化
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, len(masks) + 1, figure=fig)
    
    # 显示原始图像
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_np)
    ax.set_title('原始图像')
    ax.axis('off')
    
    # 显示掩码
    for i, (mask_np, name) in enumerate(zip(mask_np_list, mask_names)):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(mask_np, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    
    # 显示掩码叠加在原图上的效果
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(rgb_np)
    ax.set_title('原始图像')
    ax.axis('off')
    
    for i, (mask_np, name) in enumerate(zip(mask_np_list, mask_names)):
        ax = fig.add_subplot(gs[1, i+1])
        masked_rgb = rgb_np.copy()
        masked_rgb[mask_np > 0.5] = [255, 0, 0]  # 红色标记空洞
        ax.imshow(masked_rgb)
        ax.set_title(f'{name} (红色为空洞)')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()

def test_mask_comparison(rgb_path, depth_path, save_dir='mask_comparison_results'):
    """测试三种空洞掩码的差异"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载图像和深度图
    print(f"加载图像: {rgb_path}")
    print(f"加载深度图: {depth_path}")
    rgb, (H0, W0) = load_rgb(rgb_path)
    depth, _ = load_depth(depth_path, size=(H0, W0))
    
    # 调整大小并移至设备
    H, W = 512, 512  # 为了加速测试，使用较小的尺寸
    rgb = torch.nn.functional.interpolate(rgb, (H, W), mode='bilinear', align_corners=False)
    depth = torch.nn.functional.interpolate(depth, (H, W), mode='nearest', align_corners=False)
    
    rgb = rgb.to(device)
    depth = depth.to(device)
    
    # 设置相机参数
    K = default_K(H, W)
    K = torch.from_numpy(K).float().to(device).unsqueeze(0)
    
    # 创建几个虚拟视图
    num_views = 5
    offsets = [-0.1, -0.05, 0, 0.05, 0.1]  # 水平偏移量
    dT_list = [build_convergent_camera_motion(tx=o, focus_z=5.0) for o in offsets]
    dT = torch.from_numpy(np.stack(dT_list, axis=0)).float().to(device)
    
    # 扩展输入以匹配视图数量
    Is = rgb.expand(num_views, -1, -1, -1)
    Ds = depth.expand(num_views, -1, -1, -1)
    Ks = K.expand(num_views, -1, -1)
    Kt = Ks.clone()
    
    # 计算三种掩码
    print("计算softmax splat mask...")
    softmax_mask = compute_softmax_splat_mask(Is, Ds, Ks, Kt, dT)
    
    print("计算forward warp occupancy mask...")
    forward_mask = compute_forward_warp_occupancy_mask(Is, Ds, Ks, Kt, dT)
    
    print("计算真实空洞掩码...")
    gt_mask = compute_ground_truth_hole_mask(Is, Ds, Ks, Kt, dT)
    
    # 可视化比较
    mask_names = ['softmax splat mask', 'forward warp mask', '真实空洞掩码']
    visualize_mask_comparison(
        rgb, 
        [softmax_mask[0], forward_mask[0], gt_mask[0]], 
        mask_names,
        os.path.join(save_dir, 'mask_comparison.png')
    )
    
    # 计算掩码之间的差异并可视化
    diff1 = torch.abs(softmax_mask - forward_mask)
    diff2 = torch.abs(softmax_mask - gt_mask)
    diff3 = torch.abs(forward_mask - gt_mask)
    
    visualize_mask_comparison(
        rgb, 
        [diff1[0], diff2[0], diff3[0]], 
        ['softmax vs forward', 'softmax vs 真实', 'forward vs 真实'],
        os.path.join(save_dir, 'mask_differences.png')
    )
    
    print("测试完成！")

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='比较三种空洞掩码的差异')
    parser.add_argument('--rgb', required=True, help='RGB图像路径')
    parser.add_argument('--depth', required=True, help='深度图路径')
    parser.add_argument('--output', default='mask_comparison_results', help='结果保存目录')
    
    args = parser.parse_args()
    
    test_mask_comparison(args.rgb, args.depth, args.output)