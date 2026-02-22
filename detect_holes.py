import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_stereo_holes_optimized(left_img_path, right_img_path, output_dir="stereo_output"):
    
    # --- 1. 加载双目图像与Alpha通道 ---
    print("--- 1. 加载双目图像与Alpha通道 ---")
    
    # 关键修改 1: 读取左图，包含Alpha通道 (cv2.IMREAD_UNCHANGED)
    I_L_color = cv2.imread(left_img_path, cv2.IMREAD_UNCHANGED)
    # 读取右图，灰度图 (不需要Alpha通道)
    I_R = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if I_L_color is None or I_R is None:
        print("错误：无法加载指定的左右图像，使用OpenCV模拟数据。")
        I_L_color = cv2.imread(cv2.samples.findFile("starry_night.jpg"), cv2.IMREAD_UNCHANGED)
        I_R = cv2.imread(cv2.samples.findFile("box.png"), cv2.IMREAD_GRAYSCALE)
        if I_L_color is None or I_R is None:
            print("错误：无法加载图像，请检查路径。")
            return
    
    # 关键修改 2: 提取灰度图和前景掩膜
    if I_L_color.shape[2] == 4:
        # I_L_color 是 BGRA 格式
        I_L = cv2.cvtColor(I_L_color[:, :, 0:3], cv2.COLOR_BGR2GRAY)
        foreground_mask = I_L_color[:, :, 3] # Alpha 通道作为前景掩膜
    else:
        # 如果不是RGBA，则视为普通灰度图，前景掩膜全为255
        I_L = cv2.cvtColor(I_L_color, cv2.COLOR_BGR2GRAY)
        foreground_mask = np.full(I_L.shape, 255, dtype=np.uint8)

    # 确保 I_R 和 I_L 尺寸一致
    if I_L.shape != I_R.shape:
        I_R = cv2.resize(I_R, (I_L.shape[1], I_L.shape[0]), interpolation=cv2.INTER_AREA)

    H, W = I_L.shape
    print(f"图像尺寸：{W} x {H}")

    # --- 2. 计算左-右视差 (SGBM) ---
    print("\n--- 2. 计算左-右视差 (SGBM) ---")
    num_disp = 128
    block_size = 5
    
    # 关键修改 3: 增大 P1 和 P2 提高平滑性，减少物体内部空洞
    P1_new = 16 * 1 * block_size**2 # 从 8 增加到 16
    P2_new = 64 * 1 * block_size**2 # 从 32 增加到 64
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,  
        blockSize=block_size,
        P1=P1_new,
        P2=P2_new,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    d_L = stereo.compute(I_L, I_R).astype(np.float32) / 16.0
    d_R = stereo.compute(I_R, I_L).astype(np.float32) / 16.0
    
    SGBM_INVALID = (0 - 1) * 16 / 16.0 

    print("视差计算完成。")

    # --- 3. L-R 一致性检测（向量化实现） ---
    print("\n--- 3. L-R 一致性检测（真实空洞检测 - 优化）---")
    
    # 关键修改 4: 提高阈值，容忍主体内部的微小视差不一致
    threshold = 100.0 

    hole_mask = np.zeros_like(d_L, dtype=np.uint8)
    invalid_disp_mask = (d_L <= SGBM_INVALID)
    x_coords = np.arange(W).reshape(1, W)
    xr_map = (x_coords - d_L).astype(np.int32)
    boundary_mask = (xr_map < 0) | (xr_map >= W)
    xr_safe = np.clip(xr_map, 0, W - 1) 
    dr_map = np.take_along_axis(d_R, xr_safe, axis=1) 
    consistency_mask = np.abs(d_L - dr_map) > threshold
    
    final_hole_mask = invalid_disp_mask | boundary_mask | consistency_mask
    
    # 关键修改 5: 应用前景掩膜，只在前景区域标记空洞
    # foreground_mask != 0 表示前景区域 (Alpha > 0)
    hole_mask[final_hole_mask & (foreground_mask != 0)] = 255

    # 关键修改 6: 基于前景面积计算空洞比例
    foreground_area = np.sum(foreground_mask != 0)
    if foreground_area > 0:
        hole_percentage = np.sum(hole_mask == 255) / foreground_area * 100
        print(f"前景区域空洞像素比例：{hole_percentage:.2f}%")
    else:
        hole_percentage = np.sum(hole_mask == 255) / (H*W) * 100
        print(f"图像总空洞像素比例：{hole_percentage:.2f}% (前景面积为零)")


    # ===================================================
    # --- 4. 结果生成、保存与可视化 ---
    # ===================================================
    
    # 1. 准备可视化数据
    disp_vis = cv2.normalize(d_L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    # 2. 空洞叠加图
    I_color_holes = cv2.cvtColor(I_L, cv2.COLOR_GRAY2BGR)
    I_color_holes[hole_mask == 255] = [0, 0, 255] # 红色标记空洞
    
    # 3. 空洞掩膜 (转换为BGR，用于Matplotlib显示)
    hole_mask_bgr = cv2.cvtColor(hole_mask, cv2.COLOR_GRAY2BGR)
    
    # --- 创建 $2 \times 2$ 综合图 ---
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.suptitle(f"Stereo Hole Detection Analysis (W={W}, H={H})", fontsize=16)

    # 1. 左视图 (GT 源)
    axes[0, 0].imshow(I_L, cmap="gray")
    axes[0, 0].set_title("A. Original Left Image (Grayscale)")
    axes[0, 0].axis("off")

    # 2. 视差图 (彩色)
    axes[0, 1].imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("B. Disparity Map")
    axes[0, 1].axis("off")

    # 3. 空洞掩膜 (二值图)
    axes[1, 0].imshow(hole_mask_bgr)
    axes[1, 0].set_title("C. Hole Mask (Binary - Foreground Only)")
    axes[1, 0].axis("off")

    # 4. 空洞检测结果 (叠加在左图上)
    axes[1, 1].imshow(cv2.cvtColor(I_color_holes, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"D. Detected Holes (Red, {hole_percentage:.2f}% of Foreground)")
    axes[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) #左-上-右-下边距，范围0～1
    
    # --- 保存综合图 ---
    output_path = os.path.join(output_dir, "Stereo_Analysis_Composite_Result_Optimized.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n已成功保存综合结果图至: {output_path}")

    # plt.show() # 显示窗口

if __name__ == "__main__":
    # 定义输出目录
    OUTPUT_DIRECTORY = "/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/3dTo2d/"
    
    # 请替换成你自己的左右视图路径
    LEFT_PATH = "/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/3dTo2d/left_view.png"
    RIGHT_PATH = "/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/3dTo2d/right_view.png"

    detect_stereo_holes_optimized(LEFT_PATH, RIGHT_PATH, OUTPUT_DIRECTORY)