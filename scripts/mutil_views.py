# depth_warp_vs/scripts/mutil_views.py
import os
import re
import cv2
import math
import argparse
import numpy as np

def get_mask(rows, cols, theta, N, X=4.66666, koff=5):
    """
    生成柱状（或视差）合成所需的掩膜：
    rows, cols: 输出图像的高、宽
    theta: 条纹倾斜角（弧度）
    N: 视图数量
    X, koff: 面板/光栅相关参数（需按设备微调）
    """
    # 使用float32以降低内存占用
    y, x = np.mgrid[0:rows, 0:cols * 3].astype(np.float32)
    # r ∈ [0, N)
    r = (np.mod((x + koff - 3.0 * y * np.tan(np.float32(theta))), np.float32(X))) * (np.float32(N) / np.float32(X))
    pattern = np.floor(r)  # 0..N-1
    # 拆分到3个颜色通道
    mask = np.zeros((rows, cols, 3), dtype=np.uint8)
    mask[:, :, 0] = pattern[:, 0::3].astype(np.uint8)
    mask[:, :, 1] = pattern[:, 1::3].astype(np.uint8)
    mask[:, :, 2] = pattern[:, 2::3].astype(np.uint8)
    return mask

def create_3d_img(mask, views):
    """
    根据mask将多视图图像交织成单张3D合成图。
    采用直接赋值方式，避免累加带来的潜在溢出。
    """
    # 假设所有views已与mask同尺寸
    img = np.zeros_like(views[0], dtype=np.uint8)
    for i, v in enumerate(views):
        sel = (mask == i)
        img[sel] = v[sel]
    return img

def parse_index_from_name(name):
    """
    从文件名中提取_sXX_序号（s后可以是>=2位数字），例如：
    warp_49320000_s00_tx-0.25105.png -> 0
    warp_49320000_s01_tx-0.22595.png -> 1
    """
    m = re.search(r'_s(\d{2,})\D', name)
    if not m:
        m = re.search(r'_s(\d{2,})$', name)  # 文件名可能以_sXX结尾（不常见，兜底）
    if m:
        return int(m.group(1))
    return None

def load_views_from_folder(folder, target_width, target_height):
    """
    读取指定文件夹中以warp_开头的图片，按_sXX_排序，并统一缩放到目标尺寸。
    支持常见格式：png, jpg, jpeg, bmp
    """
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(folder)
             if f.lower().startswith('warp_') and f.lower().endswith(exts)]
    if not files:
        raise FileNotFoundError("未在该文件夹中找到以warp_开头的图片文件。")

    # 解析序号与排序
    indexed = []
    for f in files:
        idx = parse_index_from_name(f)
        if idx is not None:
            indexed.append((idx, f))
    if not indexed:
        raise ValueError("未从文件名中解析到_sXX_序号，请检查文件命名格式。")

    indexed.sort(key=lambda x: x[0])

    views = []
    for idx, fname in indexed:
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"警告：无法读取图像 {path}，已跳过。")
            continue
        # 直接缩放到目标分辨率（如需等比例裁切可自行替换逻辑）
        if (img.shape[1] != target_width) or (img.shape[0] != target_height):
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        views.append(img)

    if not views:
        raise ValueError("未成功读取到任何有效图像。")

    return views

def main():
    parser = argparse.ArgumentParser(description="从文件夹中读取多视图图片并生成裸眼3D合成图")
    parser.add_argument('-i', '--input_dir', type=str, required=False, default='input_dir',
                        help='输入图片文件夹路径（包含以warp_开头且带_sXX_序号的图片）')
    parser.add_argument('-o', '--output', type=str, required=False, default='3d_output.png',
                        help='输出合成图路径')
    parser.add_argument('--width', type=int, default=3840, help='输出宽度')
    parser.add_argument('--height', type=int, default=2160, help='输出高度')
    parser.add_argument('--slope', type=float, default=0.166666,
                        help='mask斜率参数，将通过theta=atan(slope)转换为角度')
    parser.add_argument('--X', type=float, default=4.66666, help='光栅周期参数X（需按设备微调）')
    parser.add_argument('--koff', type=float, default=5.0, help='相位偏移koff（需按设备微调）')
    args = parser.parse_args()

    target_width = args.width
    target_height = args.height

    # 读取并准备多视图
    views = load_views_from_folder(args.input_dir, target_width, target_height)
    N = len(views)
    print(f"已读取到 {N} 张视图图像。")

    # 生成mask并合成
    theta = math.atan(args.slope)
    mask = get_mask(target_height, target_width, theta, N, X=args.X, koff=args.koff)
    img = create_3d_img(mask, views)

    # 保存输出
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    cv2.imwrite(args.output, img)
    print(f"合成完成：{args.output}")

if __name__ == '__main__':
    # 直接在此处设置文件夹路径也可：
    # 例如：args = argparse.Namespace(input_dir='你的图片文件夹', output='3d_output.png', width=3840, height=2160, slope=0.166666, X=4.66666, koff=5.0)
    # 然后将main()中的parser改为使用该args。
    main()
# (DWvs) D:\naked-eye 3D\video call>python -m depth_warp_vs.scripts.mutil_views -i "D:\naked-eye 3D\video call\depth_warp_vs\output" -o dwvs_output.png