import cv2
import numpy as np

def show_image(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_difference(img_path1, img_path2, save_path="diff.png"):
    # 读取两张图像（确保两张图像大小一致）
    img1 = cv2.imread(r"D:\naked-eye 3D\video call\DWvs\s00_tx-0.25105.png")
    img2 = cv2.imread(r"D:\naked-eye 3D\video call\DWvs\depth_warp_vs\output\warp_49320000_s00_tx-0.25105.png")

    if img1 is None or img2 is None:
        print("无法读取输入图像，请检查路径")
        return

    if img1.shape != img2.shape:
        print(f"图像尺寸不同，img1: {img1.shape}, img2: {img2.shape}")
        return

    # 计算图像差异（绝对值差异，避免负值）
    diff = cv2.absdiff(img1, img2)

    # 也可以把差异放大一下以便观察（可选）
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=3, beta=0)  # 放大3倍

    # 保存差异图像
    cv2.imwrite(save_path, diff_enhanced)
    print(f"差异图像保存到 {save_path}")

    # 展示差异图像
    show_image("Image Difference", diff_enhanced)

if __name__ == "__main__":
    image_difference("image1.png", "image2.png")
