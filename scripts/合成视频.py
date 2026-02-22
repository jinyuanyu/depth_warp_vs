import os
import re
import cv2
import argparse
import numpy as np

def extract_tx(filename: str):
    """
    从文件名中提取 tx 值，例如：
    warp_49320000_s00_tx-0.25105.png -> -0.25105
    """
    m = re.search(r"_tx([+-]?\d+(?:\.\d+)?)", filename)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def main():
    parser = argparse.ArgumentParser(description="按文件名中的 tx 数值排序并合成视频")
    parser.add_argument("--img_dir", default=r"D:\naked-eye 3D\video call\depth_warp_vs\data\datasets\MannequinChallenge\test\0a0c63f581b9d6e0\sweep_49320000", help="输入图片文件夹")
    parser.add_argument("--out", default="D:/naked-eye 3D/video call/depth_warp_vs/output.mp4", help="输出视频文件路径")
    parser.add_argument("--fps", type=int, default=10, help="视频帧率")
    args = parser.parse_args()

    # 获取所有图片文件
    all_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # 提取 tx 值并排序
    items = []
    for f in all_files:
        tx = extract_tx(f)
        if tx is not None:
            items.append((tx, f))
    if not items:
        raise RuntimeError("未找到符合 warp_*_txX.png 格式的图片！")

    items.sort(key=lambda x: x[0])   # 按 tx 值排序

    # 读取第一张确定分辨率
    first_img = cv2.imread(os.path.join(args.img_dir, items[0][1]))
    if first_img is None:
        raise RuntimeError("无法读取图片: " + items[0][1])
    h, w, _ = first_img.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或 "XVID"
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

    for tx, fname in items:
        img = cv2.imread(os.path.join(args.img_dir, fname))
        if img is None:
            print(f"跳过无法读取的图片: {fname}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"✅ 合成完成: {args.out}, 共 {len(items)} 帧, 分辨率 {w}x{h}, 帧率 {args.fps}")

if __name__ == "__main__":
    main()
