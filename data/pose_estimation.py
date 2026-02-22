# depth_warp_vs/data/pose_estimation.py
import cv2
import numpy as np

def _to_gray(img_bgr):
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr

def estimate_pose_pnp(src_bgr, tgt_bgr, src_depth, K, nfeatures=2000, min_depth=1e-3, max_corr=1500):
    """
    通过源帧深度+2D-2D匹配，构造3D-2D并用PnP估计ΔT(源相机->目标相机)。
    返回4x4矩阵: X_t = R * X_s + t
    """
    H, W = src_bgr.shape[:2]
    K = K.astype(np.float32)
    Ki = np.linalg.inv(K)

    g1 = _to_gray(src_bgr)
    g2 = _to_gray(tgt_bgr)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1)==0 or len(k2)==0:
        return np.eye(4, dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = bf.match(d1, d2)
    ms = sorted(ms, key=lambda x: x.distance)
    if len(ms) < 6:
        return np.eye(4, dtype=np.float32)

    obj_pts = []
    img_pts = []
    cnt = 0
    for m in ms:
        u1, v1 = k1[m.queryIdx].pt
        u2, v2 = k2[m.trainIdx].pt
        z = float(src_depth[int(round(v1)) % H, int(round(u1)) % W])
        if z <= min_depth:
            continue
        p = np.array([u1, v1, 1.0], dtype=np.float32)
        Xs = Ki @ p
        Xs = Xs * z
        obj_pts.append(Xs)
        img_pts.append([u2, v2])
        cnt += 1
        if cnt >= max_corr:
            break
    if len(obj_pts) < 6:
        return np.eye(4, dtype=np.float32)

    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, None,
        flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3.0, confidence=0.99, iterationsCount=200
    )
    if not ok:
        return np.eye(4, dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3:4] = t.astype(np.float32)
    return T

if __name__ == "__main__":
    # 仅检查函数可调用
    img = np.zeros((64,64,3), np.uint8)
    dep = np.ones((64,64), np.float32)
    K = np.array([[50,0,32],[0,50,32],[0,0,1]], np.float32)
    T = estimate_pose_pnp(img, img, dep, K, nfeatures=50)
    assert T.shape == (4,4)
    print("pose_estimation self-test passed")
