# depth_warp_vs/ui/pipeline.py
import os
import cv2
import math
import numpy as np
import torch

try:
    # 包内导入
    from depth_warp_vs.main import (
        VALID_IMG_EXTS, VALID_VID_EXTS, is_video_file, natural_key,
        parse_manual_K, is_normalized_K, scale_intrinsics, resize_intrinsics, default_K,
        load_rgb, decode_depth_array,
        build_convergent_camera_motion, compute_tx_from_disp, build_offsets,
        StreamReader, get_mask, create_3d_img, warp_and_refine_batch,
        read_meta_txt, extract_timestamp_from_frame_path,
        choose_lossless_writer, FFmpegPipeWriter, FFmpegH264Writer,
        maybe_resize_views, rescale_depth_tensor_linear
    )
except Exception:
    # 源树相对导入
    from main import (
        VALID_IMG_EXTS, VALID_VID_EXTS, is_video_file, natural_key,
        parse_manual_K, is_normalized_K, scale_intrinsics, resize_intrinsics, default_K,
        load_rgb, decode_depth_array,
        build_convergent_camera_motion, compute_tx_from_disp, build_offsets,
        StreamReader, get_mask, create_3d_img, warp_and_refine_batch,
        read_meta_txt, extract_timestamp_from_frame_path,
        choose_lossless_writer, FFmpegPipeWriter, FFmpegH264Writer,
        maybe_resize_views, rescale_depth_tensor_linear
    )

try:
    from depth_warp_vs.models.refiner import build_refiner
except Exception:
    from models.refiner import build_refiner

from .state import RunParams, ensure_dir_for_file

def parse_hw_str(hw: str):
    try:
        Hs, Ws = [int(x) for x in hw.replace(",", " ").split()]
        return Hs, Ws
    except Exception:
        return 0, 0

def run_pipeline(rp: RunParams, emit):
    # 设备
    if torch.cuda.is_available() and rp.gpu is not None and rp.gpu >= 0:
        device = f"cuda:{int(rp.gpu)}"
        torch.cuda.set_device(int(rp.gpu))
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = bool(rp.amp) and str(device).startswith("cuda")
    torch.set_grad_enabled(False)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    # 尺寸
    Hs, Ws = parse_hw_str(rp.img_size)
    resize_to = None if (Hs == 0 or Ws == 0) else (Hs, Ws)
    Hf, Wf = parse_hw_str(rp.fuse_size)
    fuse_to = None if (Hf == 0 or Wf == 0) else (Hf, Wf)

    splat_cfg = {
        "temperature": float(rp.temperature),
        "occlusion": rp.occlusion,
        "hard_z_epsilon": float(rp.hard_z_epsilon)
    }
    refiner_cfg = { "model": { "refiner": { "type": rp.refiner_type, "in_ch": 4, "out_ch": 3 } } }
    if rp.refiner_type.lower() == "mgmi":
        refiner_cfg["model"]["refiner"].update({
            "base_ch": 24, "width_mult": 1.25,
            "aspp_rates": [1,2,4,8], "act": "silu", "norm": "bn", "use_se": True
        })
    else:
        refiner_cfg["model"]["refiner"].update({ "base_ch": 48, "depth": 5 })

    # 加载Refiner
    refiner = build_refiner(refiner_cfg).to(device).eval()
    if not os.path.isfile(rp.refiner_ckpt):
        raise FileNotFoundError(f"Refiner权重不存在: {rp.refiner_ckpt}")
    state = torch.load(rp.refiner_ckpt, map_location=device, weights_only=True)
    state = state.get("model", state)
    clean = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    refiner.load_state_dict(clean, strict=False)

    mask_cache = {}
    def get_cached_mask(H, W, N):
        key = (H, W, N, rp.slope, rp.X, rp.koff)
        if key in mask_cache: return mask_cache[key]
        theta = math.atan(float(rp.slope))
        m = get_mask(H, W, theta, N, X=float(rp.X), koff=float(rp.koff))
        mask_cache[key] = m
        return m

    def unify_K_for_size(K_in: np.ndarray, src_hw, dst_hw):
        if is_normalized_K(K_in):
            return scale_intrinsics(K_in, dst_hw[0], dst_hw[1])
        else:
            return resize_intrinsics(K_in, src_hw, dst_hw)

    def make_K(H, W, src_hw=None, meta_K=None):
        if rp.manual_K:
            K_manual = parse_manual_K(rp.manual_K)
            if rp.K_units == "normalized" or (rp.K_units == "auto" and is_normalized_K(K_manual)):
                Kpx = scale_intrinsics(K_manual, H, W)
            elif src_hw is not None:
                Kpx = resize_intrinsics(K_manual, src_hw, (H, W))
            else:
                Kpx = K_manual.astype(np.float32)
        elif meta_K is not None:
            Kpx = unify_K_for_size(meta_K, src_hw, (H, W))
        else:
            Kpx = default_K(H, W)
        return torch.from_numpy(Kpx).float().unsqueeze(0).to(device)

    def compute_offsets(Ks_np, Dnp):
        fx_px = float(Ks_np[0, 0])
        valid = Dnp[np.isfinite(Dnp) & (Dnp > 1e-6)]
        if valid.size < 16:
            z_ref = float(np.maximum(Dnp.mean(), 1e-3))
        else:
            q = float(np.clip(rp.disp_ref_percentile, 0.0, 1.0))
            z_ref = float(np.quantile(valid, q))
            z_ref = max(z_ref, 1e-6)
        if rp.tx_max > 0.0:
            tx_max = float(rp.tx_max)
        else:
            tx_max = compute_tx_from_disp(fx_px, z_ref, rp.max_disp_px)
        return build_offsets(tx_max, rp.num_per_side, rp.spacing)

    rmin, rmax = [float(x) for x in rp.rescale_range.replace(",", " ").split()]
    plo, phi = [float(x) for x in rp.rescale_percentiles.replace(",", " ").split()]
    rescale_on = (rp.rescale_depth == "linear")

    def set_total(n): emit("progress_total", n)
    def step(): emit("progress_step", None)
    def status(s): emit("status", s)

    # 模式判断
    is_image_mode = (rp.mode == "image")
    is_dir_mode = (rp.mode == "dir")
    is_video_mode = (rp.mode == "video")

    if is_image_mode:
        status("读取图像/深度")
        Is, (H0, W0) = load_rgb(rp.image_path, size=resize_to)
        dep_img = cv2.imread(rp.depth_path, cv2.IMREAD_UNCHANGED)
        Ds, _ = decode_depth_array(dep_img, size=resize_to, mode=rp.depth_mode, scale=rp.depth_scale,
                                   color_decode=rp.depth_color_decode, far_value=rp.far_value)
        if rescale_on:
            Ds = rescale_depth_tensor_linear(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
        H, W = Is.shape[-2:]
        meta_map = read_meta_txt(rp.meta_path, pose_convention=rp.pose_convention) if rp.meta_path else {}
        ts = extract_timestamp_from_frame_path(rp.image_path)
        meta_K = meta_map.get(ts, (None, None))[0] if (ts is not None and ts in meta_map) else None
        Ks = make_K(H, W, src_hw=(H0, W0), meta_K=meta_K)
        Ks_np = Ks.squeeze(0).cpu().numpy()

        Dnp = Ds.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        offsets = compute_offsets(Ks_np, Dnp)
        N = len(offsets)
        focus_z = float(rp.focus_depth) if rp.focus_depth > 0 else float(np.median(Dnp[np.isfinite(Dnp) & (Dnp > 1e-6)]) or 1.0)

        Is = Is.to(device); Ds = Ds.to(device)
        status("Warp+Refine")
        set_total(1)
        views_bgr, _, _ = warp_and_refine_batch(
            Is, Ds, Ks, focus_z, offsets, device, splat_cfg, refiner, amp_enabled, chunk_size=rp.chunk_views
        )
        step()
        fuse_hw = (H, W) if fuse_to is None else fuse_to
        views_bgr_f = maybe_resize_views(views_bgr, fuse_hw)
        mask = get_cached_mask(fuse_hw[0], fuse_hw[1], N)
        out_bgr = create_3d_img(mask, views_bgr_f)
        ensure_dir_for_file(rp.out_path)
        cv2.imwrite(rp.out_path, out_bgr)
        if rp.save_views_dir:
            os.makedirs(rp.save_views_dir, exist_ok=True)
            import os as _os
            stem = _os.path.splitext(_os.path.basename(rp.image_path))[0].replace("frame_", "warp_")
            for i, (img, tx) in enumerate(zip(views_bgr, offsets)):
                cv2.imwrite(_os.path.join(rp.save_views_dir, f"{stem}_s{i:02d}_tx{tx:+.5f}.png"), img)
        return rp.out_path

    if is_dir_mode:
        clip_dir = rp.pair_dir
        depth_dir = os.path.join(clip_dir, "depth")
        if not os.path.isdir(depth_dir):
            raise FileNotFoundError(f"目录缺少 depth 子目录: {depth_dir}")
        frames = [f for f in os.listdir(clip_dir) if f.lower().startswith("frame_") and f.lower().endswith(VALID_IMG_EXTS)]
        frames.sort(key=natural_key)
        pairs = []
        for f in frames:
            ts = f.split("frame_")[-1].split(".")[0]
            dep = os.path.join(depth_dir, f"depth_{ts}.png")
            if os.path.isfile(dep):
                pairs.append((os.path.join(clip_dir, f), dep))
        if not pairs:
            raise RuntimeError("未找到任何(frame_xxx, depth/depth_xxx)对")

        status("分析首帧")
        ok_rgb = cv2.imread(pairs[0][0], cv2.IMREAD_COLOR)
        if ok_rgb is None: raise RuntimeError("首帧读取失败")
        H0, W0 = ok_rgb.shape[:2]
        H_warp, W_warp = (H0, W0) if resize_to is None else resize_to
        fuse_hw = (H_warp, W_warp) if fuse_to is None else fuse_to
        meta_map = read_meta_txt(rp.meta_path, pose_convention=rp.pose_convention) if rp.meta_path else {}
        dep0 = cv2.imread(pairs[0][1], cv2.IMREAD_UNCHANGED)
        Ds0, _ = decode_depth_array(dep0, size=(H_warp, W_warp), mode=rp.depth_mode, scale=rp.depth_scale,
                                    color_decode=rp.depth_color_decode, far_value=rp.far_value)
        if rescale_on:
            Ds0 = rescale_depth_tensor_linear(Ds0, rng=(rmin, rmax), percentiles=(plo, phi))
        ts0 = extract_timestamp_from_frame_path(pairs[0][0])
        meta_K0 = meta_map.get(ts0, (None, None))[0] if (ts0 is not None and ts0 in meta_map) else None
        Ks0 = make_K(H_warp, W_warp, src_hw=(H0, W0), meta_K=meta_K0)
        Ks0_np = Ks0.squeeze(0).cpu().numpy()
        Dnp0 = Ds0.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        offsets = compute_offsets(Ks0_np, Dnp0)
        N = len(offsets)
        mask = get_cached_mask(fuse_hw[0], fuse_hw[1], N)
        if rp.focus_depth > 0:
            focus_z = float(rp.focus_depth)
        else:
            valid = Dnp0[np.isfinite(Dnp0) & (Dnp0 > 1e-6)]
            focus_z = float(np.median(valid) if valid.size > 0 else 1.0)

        # 视频写入器
        fps_out = float(rp.fps) if rp.fps > 0 else 30.0
        writer = None
        out_path = rp.out_path
        status("打开输出")
        if rp.writer_kind == "FFmpeg H.264":
            writer = FFmpegH264Writer(out_path, fps_out, (fuse_hw[1], fuse_hw[0]),
                                      qp=rp.h264_qp, preset=rp.h264_preset,
                                      pix_fmt=rp.h264_pix_fmt, close_timeout=rp.ffmpeg_close_timeout)
            out_path = writer.out_path
        elif rp.writer_kind == "FFmpeg lossless":
            writer = FFmpegPipeWriter(out_path, fps_out, (fuse_hw[1], fuse_hw[0]), close_timeout=rp.ffmpeg_close_timeout)
            out_path = writer.out_path
        elif rp.writer_kind == "OpenCV lossless":
            writer, out2, _ = choose_lossless_writer(out_path, fps_out, (fuse_hw[1], fuse_hw[0]))
            out_path = out2
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, float(fps_out), (fuse_hw[1], fuse_hw[0]), isColor=True)
            if not writer.isOpened():
                raise RuntimeError(f"无法创建视频输出: {out_path}")

        status(f"开始处理，共 {len(pairs)} 帧")
        set_total(len(pairs))

        Ks_const = None
        if (not rp.manual_K) and (not meta_map):
            Ks_const = Ks0

        for i, (rgb_p, dep_p) in enumerate(pairs):
            bgr = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
            dep = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED)
            if bgr is None or dep is None:
                step(); continue
            if (bgr.shape[0] != H_warp) or (bgr.shape[1] != W_warp):
                bgr = cv2.resize(bgr, (W_warp, H_warp), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Is = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0)/255.0
            Ds, _ = decode_depth_array(dep, size=(H_warp, W_warp), mode=rp.depth_mode, scale=rp.depth_scale,
                                       color_decode=rp.depth_color_decode, far_value=rp.far_value)
            if rescale_on:
                Ds = rescale_depth_tensor_linear(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
            if Ks_const is not None:
                Ks_i = Ks_const
            else:
                meta_Ki = None
                if meta_map:
                    ts_i = extract_timestamp_from_frame_path(rgb_p)
                    meta_Ki = meta_map.get(ts_i, (None, None))[0] if (ts_i is not None and ts_i in meta_map) else None
                Ks_i = make_K(H_warp, W_warp, src_hw=(H0, W0), meta_K=meta_Ki)

            Is = Is.to(device); Ds = Ds.to(device)
            views_bgr, _, _ = warp_and_refine_batch(
                Is, Ds, Ks_i, focus_z, offsets, device, splat_cfg, refiner, amp_enabled, chunk_size=rp.chunk_views
            )
            views_bgr_f = maybe_resize_views(views_bgr, fuse_hw)
            out_bgr = create_3d_img(mask, views_bgr_f)
            if isinstance(writer, (FFmpegPipeWriter, FFmpegH264Writer)):
                writer.write(out_bgr)
            else:
                writer.write(out_bgr)
            if rp.save_views_dir:
                os.makedirs(rp.save_views_dir, exist_ok=True)
                import os as _os
                stem = _os.path.splitext(_os.path.basename(rgb_p))[0].replace("frame_", "warp_")
                for j, (img, tx) in enumerate(zip(views_bgr, offsets)):
                    cv2.imwrite(_os.path.join(rp.save_views_dir, f"{stem}_s{j:02d}_tx{tx:+.5f}.png"), img)
            step()

        if isinstance(writer, (FFmpegPipeWriter, FFmpegH264Writer)):
            writer.release()
        else:
            writer.release()
        return out_path

    if is_video_mode:
        status("打开视频")
        rgb_reader = StreamReader(rp.video_path, is_depth=False)
        dep_reader = StreamReader(rp.depth_video_path, is_depth=True)
        ok_rgb, rgb0 = rgb_reader.read()
        ok_dep, dep0 = dep_reader.read()
        if not ok_rgb or not ok_dep:
            rgb_reader.release(); dep_reader.release()
            raise RuntimeError("无法读取视频/深度首帧")

        H0, W0 = rgb0.shape[:2]
        H_warp, W_warp = (H0, W0) if resize_to is None else resize_to
        fuse_hw = (H_warp, W_warp) if fuse_to is None else fuse_to
        Ks0 = make_K(H_warp, W_warp, src_hw=(H0, W0), meta_K=None)
        Ks_np = Ks0.squeeze(0).cpu().numpy()
        Ds0, _ = decode_depth_array(dep0, size=(H_warp, W_warp), mode=rp.depth_mode, scale=rp.depth_scale,
                                    color_decode=rp.depth_color_decode, far_value=rp.far_value)
        if rescale_on:
            Ds0 = rescale_depth_tensor_linear(Ds0, rng=(rmin, rmax), percentiles=(plo, phi))
        Dnp0 = Ds0.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        offsets = compute_offsets(Ks_np, Dnp0)
        N = len(offsets)
        mask = get_cached_mask(fuse_hw[0], fuse_hw[1], N)
        if rp.focus_depth > 0:
            focus_z = float(rp.focus_depth)
        else:
            valid = Dnp0[np.isfinite(Dnp0) & (Dnp0 > 1e-6)]
            focus_z = float(np.median(valid) if valid.size > 0 else 1.0)

        fps_out = float(rp.fps) if rp.fps > 0 else (rgb_reader.fps or 30.0)
        status("打开输出")
        writer = None
        out_path = rp.out_path
        if rp.writer_kind == "FFmpeg H.264":
            writer = FFmpegH264Writer(out_path, fps_out, (fuse_hw[1], fuse_hw[0]),
                                      qp=rp.h264_qp, preset=rp.h264_preset,
                                      pix_fmt=rp.h264_pix_fmt, close_timeout=rp.ffmpeg_close_timeout)
            out_path = writer.out_path
        elif rp.writer_kind == "FFmpeg lossless":
            writer = FFmpegPipeWriter(out_path, fps_out, (fuse_hw[1], fuse_hw[0]), close_timeout=rp.ffmpeg_close_timeout)
            out_path = writer.out_path
        elif rp.writer_kind == "OpenCV lossless":
            writer, out2, _ = choose_lossless_writer(out_path, fps_out, (fuse_hw[1], fuse_hw[0]))
            out_path = out2
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps_out, (fuse_hw[1], fuse_hw[0]), isColor=True)
            if not writer.isOpened():
                rgb_reader.release(); dep_reader.release()
                raise RuntimeError(f"无法创建视频输出: {out_path}")

        total = (rgb_reader.frame_count or 0)
        set_total(total if total > 0 else 0)
        status("开始处理")

        def to_tensor_rgb_bgr(bgr):
            if (bgr.shape[0] != H_warp) or (bgr.shape[1] != W_warp):
                bgr = cv2.resize(bgr, (W_warp, H_warp), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ten = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0)/255.0
            return ten

        def process_one(rgb_bgr, dep_img):
            Is = to_tensor_rgb_bgr(rgb_bgr).to(device)
            Ds, _ = decode_depth_array(dep_img, size=(H_warp, W_warp), mode=rp.depth_mode, scale=rp.depth_scale,
                                       color_decode=rp.depth_color_decode, far_value=rp.far_value)
            if rescale_on:
                Ds = rescale_depth_tensor_linear(Ds, rng=(rmin, rmax), percentiles=(plo, phi))
            Ds = Ds.to(device)
            views_bgr, _, _ = warp_and_refine_batch(
                Is, Ds, Ks0, focus_z, offsets, device, splat_cfg, refiner, amp_enabled, chunk_size=rp.chunk_views
            )
            views_bgr_f = maybe_resize_views(views_bgr, fuse_hw)
            out_bgr = create_3d_img(mask, views_bgr_f)
            return out_bgr

        # 首帧
        writer.write(process_one(rgb0, dep0)); step()
        while True:
            ok_rgb, rgbn = rgb_reader.read()
            ok_dep, depn = dep_reader.read()
            if not ok_rgb or not ok_dep: break
            writer.write(process_one(rgbn, depn))
            step()

        if isinstance(writer, (FFmpegPipeWriter, FFmpegH264Writer)):
            writer.release()
        else:
            writer.release()
        rgb_reader.release(); dep_reader.release()
        return out_path

    raise ValueError("未知模式")
