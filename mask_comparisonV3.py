import cv2
import torch
import numpy as np
import argparse
import os
import sys
import time
import re
import math

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.splatting.softmax_splat import softmax_splat
except ImportError:
    try:
        from depth_warp_vs.models.splatting.softmax_splat import softmax_splat
    except ImportError as e:
        print(f"无法导入 softmax_splat 模块: {e}")
        print("请确保正确安装项目依赖或设置PYTHONPATH")
        sys.exit(1)

def parse_manual_K(s: str):
    """解析手动相机内参 'fx,fy,cx,cy'"""
    vals = [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    if len(vals) != 4:
        raise ValueError("manual_K 必须为 'fx,fy,cx,cy'")
    K = np.array([[vals[0], 0.0, vals[2]],
                  [0.0, vals[1], vals[3]],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def is_normalized_K(K: np.ndarray, thresh: float = 4.0) -> bool:
    """判断是否是归一化的内参"""
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    if any([np.isnan(fx), np.isnan(fy), np.isnan(cx), np.isnan(cy)]):
        return False
    return (abs(fx) <= thresh) and (abs(fy) <= thresh) and (abs(cx) <= thresh) and (abs(cy) <= thresh)

def scale_intrinsics(K: np.ndarray, H: int, W: int):
    """缩放内参到像素单位"""
    Kp = K.copy().astype(np.float32)
    Kp[0,0] *= float(W); Kp[0,2] *= float(W)
    Kp[1,1] *= float(H); Kp[1,2] *= float(H)
    Kp[2,2] = 1.0
    return Kp

def resize_intrinsics(K: np.ndarray, src_hw, dst_hw):
    """调整内参大小"""
    H0, W0 = src_hw; H1, W1 = dst_hw
    sx = float(W1) / max(1.0, float(W0))
    sy = float(H1) / max(1.0, float(H0))
    Kp = K.copy().astype(np.float32)
    Kp[0,0] *= sx; Kp[1,1] *= sy; Kp[0,2] *= sx; Kp[1,2] *= sy; Kp[2,2] = 1.0
    return Kp

def default_K(H: int, W: int):
    """默认相机内参"""
    fx = fy = 0.9 * W
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)

def load_rgb(path, size=None):
    """加载RGB图像"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取RGB图像: {path}")
    
    H0, W0 = img.shape[:2]
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return ten, (H0, W0)

def color_depth_to_scalar_bgr(dep_bgr: np.ndarray, method: str = "auto") -> np.ndarray:
    """彩色深度图转换为标量深度图"""
    dep_bgr = dep_bgr.astype(np.float32)
    h, w, _ = dep_bgr.shape
    if method in ["r", "g", "b", "luma", "rgb24"]:
        pass
    elif method == "auto":
        diff_rg = np.mean(np.abs(dep_bgr[..., 2] - dep_bgr[..., 1]))
        diff_gb = np.mean(np.abs(dep_bgr[..., 1] - dep_bgr[..., 0]))
        diff_rb = np.mean(np.abs(dep_bgr[..., 2] - dep_bgr[..., 0]))
        method = "luma" if (diff_rg + diff_gb + diff_rb)/3.0 < 2.0 else "pca"
    if method == "r":
        v = dep_bgr[..., 2] / 255.0
    elif method == "g":
        v = dep_bgr[..., 1] / 255.0
    elif method == "b":
        v = dep_bgr[..., 0] / 255.0
    elif method == "luma":
        v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    elif method == "rgb24":
        R = dep_bgr[..., 2]; G = dep_bgr[..., 1]; B = dep_bgr[..., 0]
        v = (R*65536.0 + G*256.0 + B) / (255.0*65536.0 + 255.0*256.0 + 255.0)
    else:
        step = max(1, int(np.ceil(np.sqrt((h*w)/10000.0))))
        sample = dep_bgr[::step, ::step, :].reshape(-1, 3)
        mean = sample.mean(axis=0, keepdims=True)
        X = sample - mean
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, np.argmax(eigvals)]
        proj = (dep_bgr.reshape(-1,3) - mean) @ pc1
        proj = proj.reshape(h, w).astype(np.float32)
        pmin, pmax = float(np.min(proj)), float(np.max(proj))
        if pmax - pmin < 1e-6:
            v = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY)/255.0
        else:
            v = (proj - pmin) / (pmax - pmin)
    return np.clip(v, 0.0, 1.0).astype(np.float32)

def decode_depth_array(dep, size=None, mode="auto", scale=10.0, color_decode="auto", far_value="larger"):
    """解码深度数组"""
    if dep is None:
        raise ValueError("Empty depth array")
    
    H0, W0 = dep.shape[:2]
    
    if dep.ndim == 3:
        v01 = color_depth_to_scalar_bgr(dep, method=color_decode)
        if size is not None:
            v01 = cv2.resize(v01, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        if mode == "metric":
            depth = np.clip(v01*float(scale), 1e-6, 1e6)
        elif mode == "normalized":
            depth = np.clip(v01, 1e-6, 1.0)*float(scale)
        else:
            depth = np.clip(v01, 1e-6, 1.0)*float(scale)
    else:
        dep = dep.astype(np.float32)
        if size is not None:
            dep = cv2.resize(dep, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        if mode == "metric":
            if dep.max() > 255.0:
                dep = dep / 1000.0
            depth = np.clip(dep, 1e-3, 1e6)
        elif mode == "normalized":
            if dep.max() > 1.0:
                dep = dep / 255.0
            depth = np.clip(dep, 1e-4, 1.0) * float(scale)
        else:
            if dep.max() > 255.0:
                dep = dep / 1000.0
                depth = np.clip(dep, 1e-3, 1e6)
            else:
                if dep.max() > 1.0:
                    dep = dep / 255.0
                depth = np.clip(dep, 1e-4, 1.0) * float(scale)
    
    if far_value == "smaller":
        dmin, dmax = float(depth.min()), float(depth.max())
        depth = (dmin + dmax) - depth
        depth = np.clip(depth, 1e-6, 1e9)
    
    ten = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
    return ten, (H0, W0)

def load_depth(path, size=None, mode="auto", scale=10.0, color_decode="auto", far_value="larger"):
    """加载深度图像"""
    dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(f"无法读取深度图像: {path}")
    return decode_depth_array(dep, size=size, mode=mode, scale=scale, color_decode=color_decode, far_value=far_value)

def build_lookat_rotation_from_campos(C: np.ndarray, target: np.ndarray, up_hint: np.ndarray = None) -> np.ndarray:
    """构建朝向目标的旋转矩阵"""
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    fwd = (target.reshape(3,) - C.reshape(3,)).astype(np.float32)
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        fwd = fwd / n
    
    if abs(np.dot(fwd, up_hint)/(np.linalg.norm(up_hint)+1e-8)) > 0.999:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(fwd[2]) < 0.999 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    x_axis = np.cross(up_hint, fwd)
    nx = np.linalg.norm(x_axis)
    x_axis = x_axis/nx if nx>1e-8 else np.array([1.0,0.0,0.0], np.float32)
    
    y_axis = np.cross(fwd, x_axis)
    ny = np.linalg.norm(y_axis)
    y_axis = y_axis/ny if ny>1e-8 else np.array([0.0,1.0,0.0], np.float32)
    
    R = np.stack([x_axis, y_axis, fwd], axis=1).astype(np.float32)
    return R

def build_convergent_camera_motion(tx: float, focus_z: float) -> np.ndarray:
    """构建汇聚式相机运动"""
    C = np.array([tx, 0.0, 0.0], dtype=np.float32)
    P_focus = np.array([0.0, 0.0, max(1e-6, float(focus_z))], dtype=np.float32)
    R_cam = build_lookat_rotation_from_campos(C, P_focus, up_hint=np.array([0.0,1.0,0.0], dtype=np.float32))
    dT = np.eye(4, dtype=np.float32)
    dT[:3, :3] = R_cam
    dT[:3, 3] = C
    return dT

def compute_tx_from_disp(fx_px: float, depth_ref: float, disp_px: float) -> float:
    """从视差计算平移量"""
    return float(disp_px) * float(depth_ref) / max(1e-6, float(fx_px))

def build_offsets(tx_max: float, num_per_side: int, spacing: str):
    """构建偏移列表"""
    N = int(max(0, num_per_side))
    if N == 0:
        return [0.0]
    
    k = np.arange(-N, N+1, dtype=np.float32)
    if spacing == "linear":
        s = k / max(1, N)
    else:
        s = np.sin(0.5 * np.pi * (k/max(1,N)))
    
    return (s * float(tx_max)).tolist()

def compute_offsets_from_args(Ks_np, Dnp, args):
    """根据参数计算偏移"""
    fx_px = float(Ks_np[0,0])
    
    # 计算参考深度
    valid = Dnp[np.isfinite(Dnp) & (Dnp > 1e-6)]
    if valid.size < 16:
        z_ref = float(np.maximum(Dnp.mean(), 1e-3))
    else:
        q = float(np.clip(args.disp_ref_percentile, 0.0, 1.0))
        z_ref = float(np.quantile(valid, q))
        z_ref = max(z_ref, 1e-6)
    
    # 计算最大平移量
    if args.tx_max > 0.0:
        tx_max = float(args.tx_max)
    else:
        tx_max = compute_tx_from_disp(fx_px, z_ref, args.max_disp_px)
    
    # 构建偏移列表
    offsets = build_offsets(tx_max, args.num_per_side, args.spacing)
    return offsets, tx_max, z_ref

def make_K(H, W, src_hw=None, manual_K_str=None):
    """根据参数创建相机内参"""
    if manual_K_str is not None:
        K_manual = parse_manual_K(manual_K_str)
        # 检查是否是归一化内参
        fx, fy, cx, cy = float(K_manual[0,0]), float(K_manual[1,1]), float(K_manual[0,2]), float(K_manual[1,2])
        is_normalized = (abs(fx) <= 4.0) and (abs(fy) <= 4.0) and (abs(cx) <= 4.0) and (abs(cy) <= 4.0)
        
        if is_normalized:
            Kpx = scale_intrinsics(K_manual, H, W)
        elif src_hw is not None:
            Kpx = resize_intrinsics(K_manual, src_hw, (H, W))
        else:
            Kpx = K_manual.astype(np.float32)
    else:
        Kpx = default_K(H, W)
    
    return torch.from_numpy(Kpx).float().unsqueeze(0)

def generate_hole_mask(rgb_path, depth_path, output_path, args, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    生成空洞图像
    
    参数:
        rgb_path: RGB图像路径
        depth_path: 深度图像路径
        output_path: 空洞图像输出路径或目录
        args: 命令行参数
        device: 计算设备
    """
    # 1. 加载图像
    print(f"加载RGB图像: {rgb_path}")
    Is, (H0, W0) = load_rgb(rgb_path, size=args.img_size)
    print(f"RGB原始尺寸: {H0}x{W0}, 处理后尺寸: {Is.shape[-2]}x{Is.shape[-1]}")
    
    print(f"加载深度图像: {depth_path}")
    Ds, _ = load_depth(depth_path, size=args.img_size, mode=args.depth_mode, 
                       scale=args.depth_scale, color_decode=args.depth_color_decode,
                       far_value=args.far_value)
    print(f"深度尺寸: {Ds.shape}")
    
    # 2. 深度重标定
    if args.rescale_depth == "linear":
        print("执行深度重标定...")
        Dnp = Ds.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        v = Dnp[np.isfinite(Dnp) & (Dnp > 1e-8)]
        if v.size >= 16:
            plo, phi = float(args.rescale_percentiles[0]), float(args.rescale_percentiles[1])
            z0 = float(np.percentile(v, plo))
            z1 = float(np.percentile(v, phi))
            if np.isfinite(z0) and np.isfinite(z1) and z1 > z0:
                a, b = float(args.rescale_range[0]), float(args.rescale_range[1])
                d2 = a + (Dnp - z0) * (b - a) / (z1 - z0)
                d2 = np.clip(d2, min(a,b), max(a,b)).astype(np.float32)
                Ds = torch.from_numpy(d2).view(1,1,*d2.shape).to(Ds.dtype)
    
    # 3. 准备数据并移动到设备
    Is = Is.to(device)
    Ds = Ds.to(device)
    
    # 4. 准备相机参数
    H, W = Is.shape[-2:]
    
    Ks = make_K(H, W, src_hw=(H0, W0), manual_K_str=args.manual_K)
    Ks_np = Ks.squeeze(0).numpy()
    
    Dnp = Ds.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    
    # 5. 计算偏移
    offsets, tx_max, z_ref = compute_offsets_from_args(Ks_np, Dnp, args)
    
    # 使用指定的对焦深度
    focus_z = float(args.focus_depth) if args.focus_depth > 0 else 5.0
    
    print(f"相机参数:")
    print(f"  内参 K: fx={Ks_np[0,0]:.2f}, fy={Ks_np[1,1]:.2f}, cx={Ks_np[0,2]:.2f}, cy={Ks_np[1,2]:.2f}")
    print(f"  参考深度: {z_ref:.4f}")
    print(f"  最大平移: {tx_max:.4f}")
    print(f"  对焦深度: {focus_z:.4f}")
    print(f"  偏移列表: {offsets}")
    
    # 6. 确定使用哪个偏移进行空洞检测
    # 默认使用第一个非零偏移（或指定偏移索引）
    if args.offset_index is not None and 0 <= args.offset_index < len(offsets):
        offset_idx = args.offset_index
    else:
        # 找到第一个非零偏移
        offset_idx = 0
        for i, o in enumerate(offsets):
            if abs(o) > 1e-6:
                offset_idx = i
                break
    
    tx = offsets[offset_idx]
    print(f"使用偏移索引 {offset_idx}: tx = {tx}")
    
    # 7. 准备相机运动
    dT_np = build_convergent_camera_motion(tx, focus_z)
    dT = torch.from_numpy(dT_np).float().unsqueeze(0).to(device)
    
    Ks = Ks.to(device)
    Kt = Ks  # 假设目标相机内参与源相同
    
    # 8. 执行softmax_splat获取可见性权重
    print("执行softmax_splat...")
    start_time = time.time()
    
    # 使用 channels_last 格式
    if args.channels_last and Is.dim() == 4:
        Is_ = Is.contiguous(memory_format=torch.channels_last)
        Ds_ = Ds.contiguous(memory_format=torch.channels_last)
    else:
        Is_ = Is
        Ds_ = Ds
    
    with torch.no_grad():
        Iw, V = softmax_splat(
            Is_, Ds_, Ks, Kt, dT,
            temperature=args.temperature,
            normalize=True,
            occlusion=args.occlusion,
            hard_z_epsilon=args.hard_z_epsilon
        )
    
    splat_time = time.time() - start_time
    print(f"softmax_splat 执行时间: {splat_time:.3f}秒")
    
    # 9. 生成空洞掩码（V <= 1e-8 的区域）
    print("生成空洞掩码...")
    hole_mask = (V <= 1e-8).float()
    
    # 10. 转换为可视化的空洞图像
    hole_mask_cpu = hole_mask.squeeze().cpu().numpy()
    
    # 将空洞掩码转换为二值图像（0-255）
    hole_visual = (hole_mask_cpu * 255).astype(np.uint8)
    
    # 可选：应用形态学操作使空洞更清晰
    kernel = np.ones((3, 3), np.uint8)
    hole_visual = cv2.morphologyEx(hole_visual, cv2.MORPH_CLOSE, kernel)
    
    # 11. 确定输出文件路径
    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        # 从RGB路径提取文件名
        rgb_filename = os.path.basename(rgb_path)
        name, ext = os.path.splitext(rgb_filename)
        output_filename = f"{name}_hole_mask_tx{tx:.4f}_focus{focus_z:.1f}.png"
        output_filepath = os.path.join(output_path, output_filename)
    else:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        output_filepath = output_path
    
    # 12. 保存空洞图像
    print(f"保存空洞图像到: {output_filepath}")
    cv2.imwrite(output_filepath, hole_visual)
    
    # 13. 统计信息
    total_pixels = hole_mask_cpu.size
    hole_pixels = np.sum(hole_mask_cpu)
    hole_ratio = hole_pixels / total_pixels * 100
    
    print(f"空洞统计:")
    print(f"  总像素数: {total_pixels}")
    print(f"  空洞像素数: {int(hole_pixels)}")
    print(f"  空洞比例: {hole_ratio:.2f}%")
    
    # 14. 生成带空洞标注的RGB图像
    rgb_with_holes = cv2.imread(rgb_path)
    if rgb_with_holes is not None:
        # 将空洞区域标记为红色
        hole_mask_bool = hole_mask_cpu > 0.5
        
        # 如果原始图像尺寸与空洞掩码尺寸不一致，调整空洞掩码
        if rgb_with_holes.shape[:2] != hole_mask_cpu.shape[:2]:
            # 调整空洞掩码大小
            hole_mask_resized = cv2.resize(hole_mask_cpu.astype(np.float32), 
                                          (rgb_with_holes.shape[1], rgb_with_holes.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
            hole_mask_bool = hole_mask_resized > 0.5
        
        # 创建标记图像
        marked_rgb = rgb_with_holes.copy()
        marked_rgb[hole_mask_bool] = [0, 0, 255]  # BGR中的红色
        
        # 创建标记图像的输出路径
        if os.path.isdir(output_path):
            marked_filename = f"{name}_hole_marked_tx{tx:.4f}_focus{focus_z:.1f}.png"
            output_path_marked = os.path.join(output_path, marked_filename)
        else:
            output_path_marked = output_filepath.replace(".png", "_marked.png").replace(".jpg", "_marked.jpg")
        
        cv2.imwrite(output_path_marked, marked_rgb)
        print(f"带空洞标记的RGB图像已保存到: {output_path_marked}")
    
    # 15. 保存V值（可见性权重）用于分析
    V_cpu = V.squeeze().cpu().numpy()
    V_normalized = ((V_cpu - V_cpu.min()) / (V_cpu.max() - V_cpu.min() + 1e-8) * 255).astype(np.uint8)
    
    if os.path.isdir(output_path):
        V_filename = f"{name}_visibility_tx{tx:.4f}_focus{focus_z:.1f}.png"
        output_path_V = os.path.join(output_path, V_filename)
    else:
        output_path_V = output_filepath.replace(".png", "_visibility.png").replace(".jpg", "_visibility.jpg")
    
    cv2.imwrite(output_path_V, V_normalized)
    print(f"可见性权重图已保存到: {output_path_V}")
    
    # 16. 保存扭曲后的图像
    Iw_cpu = Iw.squeeze().permute(1, 2, 0).cpu().numpy()
    Iw_uint8 = (Iw_cpu * 255).clip(0, 255).astype(np.uint8)
    Iw_bgr = cv2.cvtColor(Iw_uint8, cv2.COLOR_RGB2BGR)
    
    if os.path.isdir(output_path):
        Iw_filename = f"{name}_warped_tx{tx:.4f}_focus{focus_z:.1f}.png"
        output_path_Iw = os.path.join(output_path, Iw_filename)
    else:
        output_path_Iw = output_filepath.replace(".png", "_warped.png").replace(".jpg", "_warped.jpg")
    
    cv2.imwrite(output_path_Iw, Iw_bgr)
    print(f"扭曲后图像已保存到: {output_path_Iw}")
    
    # 17. 保存详细参数信息
    params_file = os.path.join(os.path.dirname(output_filepath), f"{name}_params.txt")
    with open(params_file, 'w') as f:
        f.write("空洞生成参数:\n")
        f.write(f"  RGB图像: {rgb_path}\n")
        f.write(f"  深度图像: {depth_path}\n")
        f.write(f"  原始尺寸: {H0}x{W0}\n")
        f.write(f"  处理后尺寸: {H}x{W}\n")
        f.write(f"  相机内参: fx={Ks_np[0,0]:.2f}, fy={Ks_np[1,1]:.2f}, cx={Ks_np[0,2]:.2f}, cy={Ks_np[1,2]:.2f}\n")
        f.write(f"  参考深度: {z_ref:.4f}\n")
        f.write(f"  最大平移: {tx_max:.4f}\n")
        f.write(f"  对焦深度: {focus_z:.4f}\n")
        f.write(f"  使用偏移: tx={tx:.4f} (索引 {offset_idx})\n")
        f.write(f"  总偏移数: {len(offsets)}\n")
        f.write(f"  偏移列表: {offsets}\n")
        f.write(f"  temperature: {args.temperature}\n")
        f.write(f"  occlusion: {args.occlusion}\n")
        f.write(f"  hard_z_epsilon: {args.hard_z_epsilon}\n")
        f.write(f"  channels_last: {args.channels_last}\n")
        f.write(f"  空洞比例: {hole_ratio:.2f}%\n")
        f.write(f"  空洞像素数: {int(hole_pixels)} / {total_pixels}\n")
        f.write(f"  处理时间: {splat_time:.3f}秒\n")
    
    print(f"参数信息已保存到: {params_file}")
    
    return hole_visual, hole_ratio, output_filepath

def main():
    parser = argparse.ArgumentParser(description="基于深度和RGB图像生成空洞掩码")
    
    # 输入输出参数
    parser.add_argument("--rgb", required=True, help="RGB图像路径")
    parser.add_argument("--depth", required=True, help="深度图像路径")
    parser.add_argument("--output", required=True, help="输出路径或目录")
    
    # 相机和深度参数
    parser.add_argument("--manual_K", default="1402.1,1402.1,968.77,506.154", help="手动相机内参 'fx,fy,cx,cy'")
    parser.add_argument("--focus_depth", type=float, default=5.9, help="对焦深度")
    parser.add_argument("--img_size", default="1080,1920", help="处理尺寸 H,W (0,0=保持原始)")
    
    # 深度处理参数
    parser.add_argument("--depth_mode", default="auto", choices=["auto","metric","normalized"], help="深度图模式")
    parser.add_argument("--depth_scale", type=float, default=10.0, help="深度缩放因子")
    parser.add_argument("--depth_color_decode", default="auto", choices=["auto","pca","luma","rgb24","r","g","b"], help="彩色深度图解码方式")
    parser.add_argument("--far_value", default="larger", choices=["larger","smaller"], help="深度值更大是否代表更远")
    
    # 深度重标定参数
    parser.add_argument("--rescale_depth", default="none", choices=["none","linear"], help="是否重标定深度")
    parser.add_argument("--rescale_range", default="2,10", help="重标定范围 [dmin,dmax]")
    parser.add_argument("--rescale_percentiles", default="1,99", help="重标定百分位点")
    
    # 视图合成参数
    parser.add_argument("--num_per_side", type=int, default=2, help="每侧视角数")
    parser.add_argument("--spacing", default="linear", choices=["linear","cosine"], help="偏移间距")
    parser.add_argument("--tx_max", type=float, default=0.0, help="最大平移量（>0则使用该值）")
    parser.add_argument("--max_disp_px", type=float, default=25.0, help="最大视差（像素）")
    parser.add_argument("--disp_ref_percentile", type=float, default=0.5, help="参考深度百分位点")
    
    # splatting参数
    parser.add_argument("--temperature", type=float, default=30.0, help="splatting温度")
    parser.add_argument("--occlusion", default="hard", choices=["hard","soft"], help="遮挡处理方式")
    parser.add_argument("--hard_z_epsilon", type=float, default=1e-3, help="硬遮挡epsilon")
    
    # 其他参数
    parser.add_argument("--channels_last", action="store_true", help="使用channels_last内存格式")
    parser.add_argument("--offset_index", type=int, default=None, help="指定使用哪个偏移（默认使用第一个非零偏移）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    
    args = parser.parse_args()
    
    # 解析尺寸参数
    Hs, Ws = [int(x) for x in args.img_size.replace(",", " ").split()]
    args.img_size = None if (Hs == 0 or Ws == 0) else (Hs, Ws)
    
    # 解析重标定参数
    args.rescale_range = [float(x) for x in args.rescale_range.replace(",", " ").split()]
    args.rescale_percentiles = [float(x) for x in args.rescale_percentiles.replace(",", " ").split()]
    
    # 检查输入文件是否存在
    if not os.path.exists(args.rgb):
        print(f"错误: RGB图像不存在: {args.rgb}")
        return 1
    
    if not os.path.exists(args.depth):
        print(f"错误: 深度图像不存在: {args.depth}")
        return 1
    
    # 生成空洞图像
    try:
        hole_image, hole_ratio, output_filepath = generate_hole_mask(
            rgb_path=args.rgb,
            depth_path=args.depth,
            output_path=args.output,
            args=args,
            device=args.device
        )
        
        print(f"\n空洞图像生成完成!")
        print(f"输出文件: {output_filepath}")
        print(f"空洞比例: {hole_ratio:.2f}%")
        
    except Exception as e:
        print(f"生成空洞图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())