# depth_warp_vs/ui/app.py
import os
import sys
import time
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
import torch
from PIL import Image, ImageTk

try:
    from depth_warp_vs.main import (
        VALID_IMG_EXTS, VALID_VID_EXTS, is_video_file,
        StreamReader, decode_depth_array
    )
except Exception:
    from main import (
        VALID_IMG_EXTS, VALID_VID_EXTS, is_video_file,
        StreamReader, decode_depth_array
    )

from .state import RunParams
from .widgets import labeled_entry, labeled_combo, labeled_check, labeled_entry_with_btn
from .pipeline import run_pipeline

def pil_from_bgr(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def colorize_depth_for_click(depth_np: np.ndarray) -> np.ndarray:
    d = depth_np.copy().astype(np.float32)
    vals = d[np.isfinite(d) & (d > 1e-8)]
    if vals.size == 0:
        dmin, dmax = 0.0, 1.0
    else:
        dmin = float(np.percentile(vals, 1.0))
        dmax = float(np.percentile(vals, 99.0))
        if dmax <= dmin: dmax = dmin + 1e-6
    v = np.clip((depth_np - dmin) / (dmax - dmin), 0.0, 1.0)
    v8 = (v * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(v8, cv2.COLORMAP_TURBO)
    return cm

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("深度引导视角合成 - 图形界面")

        # 窗口尺寸设置：
        # - 默认按屏幕分辨率的 85% 设置初始大小
        # - 设定较大的最小尺寸，避免按钮被挤到右侧不可见
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        ww = max(1200, min(int(sw * 0.85), sw))
        wh = max(800, min(int(sh * 0.85), sh))
        self.geometry(f"{ww}x{wh}")
        self.minsize(1100, 760)

        # 变量
        self.mode_var = tk.StringVar(value="image")
        self.rgb_path_var = tk.StringVar(value="")
        self.depth_path_var = tk.StringVar(value="")
        self.dir_path_var = tk.StringVar(value="")
        self.video_path_var = tk.StringVar(value="")
        self.depth_video_path_var = tk.StringVar(value="")

        self.img_size_var = tk.StringVar(value="0,0")
        self.fuse_size_var = tk.StringVar(value="2160,3840")
        self.manual_K_var = tk.StringVar(value="")
        self.K_units_var = tk.StringVar(value="auto")
        self.meta_path_var = tk.StringVar(value="")
        self.pose_convention_var = tk.StringVar(value="w2c")
        self.fps_var = tk.DoubleVar(value=30.0)

        self.depth_mode_var = tk.StringVar(value="metric")
        self.depth_scale_var = tk.DoubleVar(value=10.0)
        self.depth_color_decode_var = tk.StringVar(value="auto")
        self.far_value_var = tk.StringVar(value="larger")

        self.rescale_depth_var = tk.StringVar(value="linear")
        self.rescale_range_var = tk.StringVar(value="2,10")
        self.rescale_percentiles_var = tk.StringVar(value="1,99")

        self.num_per_side_var = tk.IntVar(value=10)
        self.spacing_var = tk.StringVar(value="linear")
        self.tx_max_var = tk.DoubleVar(value=0.0)
        self.max_disp_px_var = tk.DoubleVar(value=15.0)
        self.disp_ref_percentile_var = tk.DoubleVar(value=0.5)

        self.temperature_var = tk.DoubleVar(value=30.0)
        self.occlusion_var = tk.StringVar(value="hard")
        self.hard_z_epsilon_var = tk.DoubleVar(value=1e-3)
        self.amp_var = tk.BooleanVar(value=False)

        self.refiner_type_var = tk.StringVar(value="MGMI")
        self.refiner_ckpt_var = tk.StringVar(value="")

        self.slope_var = tk.DoubleVar(value=0.166666)
        self.X_var = tk.DoubleVar(value=4.66666)
        self.koff_var = tk.DoubleVar(value=5.0)

        self.focus_depth_var = tk.DoubleVar(value=0.0)
        self.out_path_var = tk.StringVar(value="")

        self.gpu_var = tk.IntVar(value=-1)
        self.save_views_dir_var = tk.StringVar(value="")
        self.chunk_views_var = tk.IntVar(value=0)

        self.writer_kind_var = tk.StringVar(value="OpenCV mp4v")
        self.ffmpeg_close_timeout_var = tk.DoubleVar(value=300.0)
        self.h264_qp_var = tk.IntVar(value=23)
        self.h264_preset_var = tk.StringVar(value="medium")
        self.h264_pix_fmt_var = tk.StringVar(value="yuv420p")

        # 运行状态
        self.run_thread = None
        self.msg_queue = queue.Queue()
        self.start_time = None
        self.total_steps = 0
        self.done_steps = 0
        self._elapsed_ticking = False

        self.depth_preview_np = None
        self.depth_preview_imgtk = None
        self.selected_focus_depth = 0.0

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.nb = nb

        self.page_input = ttk.Frame(nb)
        self.page_depth = ttk.Frame(nb)
        self.page_params = ttk.Frame(nb)
        self.page_run = ttk.Frame(nb)
        nb.add(self.page_input, text="1) 选择输入")
        nb.add(self.page_depth, text="2) 对焦深度")
        nb.add(self.page_params, text="3) 设置参数")
        nb.add(self.page_run, text="4) 运行")

        self._build_page_input()
        self._build_page_depth()
        self._build_page_params()
        self._build_page_run()

    def _build_page_input(self):
        frm = self.page_input
        mode_box = ttk.LabelFrame(frm, text="输入模式")
        mode_box.pack(fill="x", padx=8, pady=8)
        ttk.Radiobutton(mode_box, text="图像模式（单图+单深度）", variable=self.mode_var, value="image",
                        command=self._refresh_input_mode).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(mode_box, text="目录模式（frame_xxx + depth/depth_xxx.png）", variable=self.mode_var, value="dir",
                        command=self._refresh_input_mode).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(mode_box, text="视频模式（RGB视频 + 深度视频或序列）", variable=self.mode_var, value="video",
                        command=self._refresh_input_mode).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        self.path_box = ttk.LabelFrame(frm, text="路径设置")
        self.path_box.pack(fill="both", padx=8, pady=8, expand=True)

        # 三个子区域：仅显示当前模式所需项
        self.input_frames = {}

        # 图像模式
        f_img = ttk.Frame(self.path_box)
        self.input_frames["image"] = f_img
        r1 = ttk.Frame(f_img); r1.pack(fill="x", padx=6, pady=6)
        ttk.Label(r1, text="原图(图片):").pack(side="left")
        ttk.Entry(r1, textvariable=self.rgb_path_var, width=80).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="浏览", command=lambda: self._browse_file(self.rgb_path_var, VALID_IMG_EXTS)).pack(side="left")
        r2 = ttk.Frame(f_img); r2.pack(fill="x", padx=6, pady=6)
        ttk.Label(r2, text="深度(图片):").pack(side="left")
        ttk.Entry(r2, textvariable=self.depth_path_var, width=80).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="浏览", command=lambda: self._browse_file(self.depth_path_var, VALID_IMG_EXTS)).pack(side="left")

        # 目录模式
        f_dir = ttk.Frame(self.path_box)
        self.input_frames["dir"] = f_dir
        r3 = ttk.Frame(f_dir); r3.pack(fill="x", padx=6, pady=6)
        ttk.Label(r3, text="成对目录:").pack(side="left")
        ttk.Entry(r3, textvariable=self.dir_path_var, width=80).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r3, text="浏览", command=lambda: self._browse_dir(self.dir_path_var)).pack(side="left")

        # 视频模式
        f_vid = ttk.Frame(self.path_box)
        self.input_frames["video"] = f_vid
        r4 = ttk.Frame(f_vid); r4.pack(fill="x", padx=6, pady=6)
        ttk.Label(r4, text="RGB视频:").pack(side="left")
        ttk.Entry(r4, textvariable=self.video_path_var, width=80).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r4, text="浏览", command=lambda: self._browse_file(self.video_path_var, VALID_VID_EXTS)).pack(side="left")
        r5 = ttk.Frame(f_vid); r5.pack(fill="x", padx=6, pady=6)
        ttk.Label(r5, text="深度视频/目录:").pack(side="left")
        ttk.Entry(r5, textvariable=self.depth_video_path_var, width=80).pack(side="left", fill="x", expand=True, padx=6)
        btn = ttk.Frame(r5); btn.pack(side="left", padx=6)
        ttk.Button(btn, text="视频", command=lambda: self._browse_file(self.depth_video_path_var, VALID_VID_EXTS)).pack(side="left")
        ttk.Button(btn, text="目录", command=lambda: self._browse_dir(self.depth_video_path_var)).pack(side="left")

        ttk.Button(frm, text="下一步：选择对焦深度", command=self._go_to_depth_step).pack(pady=10)

        self._refresh_input_mode()

    def _browse_file(self, var: tk.StringVar, exts):
        p = filedialog.askopenfilename(title="选择文件")
        if not p: return
        if exts and os.path.splitext(p)[1].lower() not in exts:
            messagebox.showerror("错误", f"文件扩展名不符合要求：{exts}")
            return
        var.set(p)

    def _browse_dir(self, var: tk.StringVar):
        p = filedialog.askdirectory(title="选择目录")
        if p: var.set(p)

    def _refresh_input_mode(self):
        mode = self.mode_var.get()
        for k, f in self.input_frames.items():
            f.pack_forget()
        self.input_frames[mode].pack(fill="both", expand=True, padx=6, pady=6)

    def _build_page_depth(self):
        frm = self.page_depth
        ttk.Label(frm, text="在下方深度图中点击以选择对焦深度（显示原尺寸）。也可在参数页手动填写对焦深度。").pack(anchor="w", padx=8, pady=(8, 4))
        pv_frame = ttk.Frame(frm); pv_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.depth_canvas = tk.Canvas(pv_frame, bg="#222222")
        self.depth_canvas.pack(fill="both", expand=True)
        self.depth_canvas.bind("<Button-1>", self._on_depth_click)

        st_frame = ttk.Frame(frm); st_frame.pack(fill="x", padx=8, pady=8)
        self.depth_info_var = tk.StringVar(value="未加载深度预览")
        ttk.Label(st_frame, textvariable=self.depth_info_var, foreground="#0066aa").pack(side="left")

        nav = ttk.Frame(frm); nav.pack(fill="x", padx=8, pady=8)
        ttk.Button(nav, text="上一步", command=lambda: self.nb.select(self.page_input)).pack(side="left")
        ttk.Button(nav, text="下一步：参数设置", command=lambda: self.nb.select(self.page_params)).pack(side="right")

    def _load_depth_preview(self, dep):
        Ds, _ = decode_depth_array(dep, size=None,
                                   mode=self.depth_mode_var.get(),
                                   scale=float(self.depth_scale_var.get()),
                                   color_decode=self.depth_color_decode_var.get(),
                                   far_value=self.far_value_var.get())
        dnp = Ds.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        self.depth_preview_np = dnp
        cm = colorize_depth_for_click(dnp)
        h, w = cm.shape[:2]
        cw = max(100, self.depth_canvas.winfo_width())
        ch = max(100, self.depth_canvas.winfo_height())
        scale = min(1.0, min(float(cw) / w, float(ch) / h))
        ww = int(w * scale); hh = int(h * scale)
        cm_show = cm if scale >= 1.0 else cv2.resize(cm, (ww, hh), interpolation=cv2.INTER_AREA)
        img = pil_from_bgr(cm_show)
        self.depth_preview_imgtk = ImageTk.PhotoImage(img)
        self.depth_canvas.delete("all")
        self.depth_canvas.create_image((cw // 2, ch // 2), image=self.depth_preview_imgtk, anchor="center")
        self.depth_canvas.image_offset = ((cw - ww) // 2, (ch - hh) // 2)
        self.depth_canvas.image_size = (ww, hh)
        self.depth_canvas.src_size = (w, h)
        self.depth_info_var.set(f"深度预览: {h}x{w} | 点击选择焦点深度。当前: {self.selected_focus_depth:.6f}")

    def _go_to_depth_step(self):
        mode = self.mode_var.get()
        try:
            if mode == "image":
                img_p = self.rgb_path_var.get().strip()
                dep_p = self.depth_path_var.get().strip()
                if not (os.path.isfile(img_p) and os.path.isfile(dep_p)):
                    messagebox.showerror("错误", "图像模式需要选择“原图(图片)”与“深度(图片)”。")
                    return
                dep = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED)
                if dep is None: raise RuntimeError("无法读取深度图片")
                self._load_depth_preview(dep)
            elif mode == "dir":
                dir_p = self.dir_path_var.get().strip()
                if not os.path.isdir(dir_p):
                    messagebox.showerror("错误", "目录模式需要选择有效“成对目录”。")
                    return
                depth_dir = os.path.join(dir_p, "depth")
                if not os.path.isdir(depth_dir):
                    messagebox.showerror("错误", "目录缺少 depth 子目录。")
                    return
                dep_files = [f for f in os.listdir(depth_dir) if f.lower().endswith(".png") and f.lower().startswith("depth_")]
                if not dep_files:
                    messagebox.showerror("错误", "找不到任何 depth_*.png")
                    return
                dep = cv2.imread(os.path.join(depth_dir, dep_files[0]), cv2.IMREAD_UNCHANGED)
                if dep is None: raise RuntimeError("无法读取深度图片")
                self._load_depth_preview(dep)
            else:
                vid_p = self.video_path_var.get().strip()
                dvid_p = self.depth_video_path_var.get().strip()
                if not (os.path.isfile(vid_p) and is_video_file(vid_p)):
                    messagebox.showerror("错误", "视频模式需要有效“RGB视频”。")
                    return
                if not (os.path.isfile(dvid_p) or os.path.isdir(dvid_p)):
                    messagebox.showerror("错误", "视频模式需要有效“深度视频（文件或目录）”。")
                    return
                reader = StreamReader(dvid_p, is_depth=True)
                ok, dep0 = reader.read()
                reader.release()
                if not ok or dep0 is None:
                    messagebox.showerror("错误", "无法读取深度首帧")
                    return
                self._load_depth_preview(dep0)
        except Exception as e:
            messagebox.showerror("错误", f"加载深度预览失败：{e}")
            return
        self.nb.select(self.page_depth)

    def _on_depth_click(self, event):
        if self.depth_preview_np is None:
            return
        offx, offy = getattr(self.depth_canvas, "image_offset", (0, 0))
        ww, hh = getattr(self.depth_canvas, "image_size", (0, 0))
        W, H = getattr(self.depth_canvas, "src_size", (0, 0))
        x, y = event.x, event.y
        if not (offx <= x <= offx + ww and offy <= y <= offy + hh):
            return
        sx = (x - offx) / max(1, ww)
        sy = (y - offy) / max(1, hh)
        ix = int(round(sx * (W - 1)))
        iy = int(round(sy * (H - 1)))
        z = float(self.depth_preview_np[iy, ix])
        if not np.isfinite(z) or z <= 1e-8:
            return
        self.selected_focus_depth = z
        self.depth_info_var.set(f"选择: (x={ix}, y={iy}), z={z:.6f}")
        self.focus_depth_var.set(z)

    def _build_page_params(self):
        frm = self.page_params
        left = ttk.LabelFrame(frm, text="基础与深度")
        right = ttk.LabelFrame(frm, text="相机/编码/其他")
        left.pack(side="left", fill="both", expand=True, padx=(8,4), pady=8)
        right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)

        # 左侧：基础参数
        self.row_img_size, _ = labeled_entry(left, "Warp尺寸 (H,W)", self.img_size_var, hint="0,0=按输入")
        self.row_img_size.frame.pack(fill="x", pady=2)
        self.row_fuse_size, _ = labeled_entry(left, "合成输出 (H,W)", self.fuse_size_var, hint="0,0=跟随warp")
        self.row_fuse_size.frame.pack(fill="x", pady=2)
        self.row_occ, _ = labeled_combo(left, "遮挡模式", self.occlusion_var, ["hard", "soft"])
        self.row_occ.frame.pack(fill="x", pady=2)
        self.row_temp, _ = labeled_entry(left, "温度(temperature)", self.temperature_var)
        self.row_temp.frame.pack(fill="x", pady=2)
        self.row_hz, _ = labeled_entry(left, "硬遮挡z阈", self.hard_z_epsilon_var)
        self.row_hz.frame.pack(fill="x", pady=2)
        self.row_amp, _ = labeled_check(left, "启用AMP半精度(需CUDA)", self.amp_var)
        self.row_amp.frame.pack(fill="x", pady=2)

        # 深度
        box_depth = ttk.LabelFrame(left, text="深度与重标定")
        box_depth.pack(fill="x", padx=6, pady=6)
        self.row_dmode, _ = labeled_combo(box_depth, "深度模式", self.depth_mode_var, ["auto", "metric", "normalized"])
        self.row_dmode.frame.pack(fill="x", pady=2)
        self.row_dscale, _ = labeled_entry(box_depth, "深度尺度(scale)", self.depth_scale_var)
        self.row_dscale.frame.pack(fill="x", pady=2)
        self.row_dcd, _ = labeled_combo(box_depth, "彩色深度解码", self.depth_color_decode_var,
                                        ["auto","pca","luma","rgb24","r","g","b"])
        self.row_dcd.frame.pack(fill="x", pady=2)
        self.row_far, _ = labeled_combo(box_depth, "远近关系", self.far_value_var, ["larger","smaller"])
        self.row_far.frame.pack(fill="x", pady=2)
        self.row_rescale, _ = labeled_combo(box_depth, "深度重标定", self.rescale_depth_var, ["none","linear"])
        self.row_rescale.frame.pack(fill="x", pady=2)
        self.row_r_rng, _ = labeled_entry(box_depth, "重标定范围[a,b]", self.rescale_range_var)
        self.row_r_rng.frame.pack(fill="x", pady=2)
        self.row_r_pct, _ = labeled_entry(box_depth, "重标定分位点[p0,p1]", self.rescale_percentiles_var)
        self.row_r_pct.frame.pack(fill="x", pady=2)

        # 视角
        box_views = ttk.LabelFrame(left, text="视角/位姿")
        box_views.pack(fill="x", padx=6, pady=6)
        self.row_nps, _ = labeled_entry(box_views, "左右各视角数", self.num_per_side_var)
        self.row_nps.frame.pack(fill="x", pady=2)
        self.row_space, _ = labeled_combo(box_views, "视角间距", self.spacing_var, ["linear","cosine"])
        self.row_space.frame.pack(fill="x", pady=2)
        self.row_txmax, _ = labeled_entry(box_views, "最大tx(>0使用)", self.tx_max_var)
        self.row_txmax.frame.pack(fill="x", pady=2)
        self.row_maxdisp, _ = labeled_entry(box_views, "最大视差像素", self.max_disp_px_var)
        self.row_maxdisp.frame.pack(fill="x", pady=2)
        self.row_dispref, _ = labeled_entry(box_views, "参考深度分位点[0~1]", self.disp_ref_percentile_var)
        self.row_dispref.frame.pack(fill="x", pady=2)

        # 左侧合成
        box_mask = ttk.LabelFrame(left, text="裸眼3D合成")
        box_mask.pack(fill="x", padx=6, pady=6)
        self.row_slope, _ = labeled_entry(box_mask, "mask斜率(slope)", self.slope_var)
        self.row_slope.frame.pack(fill="x", pady=2)
        self.row_X, _ = labeled_entry(box_mask, "X(光栅周期)", self.X_var)
        self.row_X.frame.pack(fill="x", pady=2)
        self.row_koff, _ = labeled_entry(box_mask, "koff", self.koff_var)
        self.row_koff.frame.pack(fill="x", pady=2)

        # 右侧相机/设备
        box_cam = ttk.LabelFrame(right, text="相机/Meta/设备")
        box_cam.pack(fill="x", padx=6, pady=6)
        self.row_manualK, _ = labeled_entry(box_cam, "手动K 'fx,fy,cx,cy'", self.manual_K_var, hint="留空则使用Meta或默认")
        self.row_manualK.frame.pack(fill="x", pady=2)
        self.row_Kunits, _ = labeled_combo(box_cam, "K单位", self.K_units_var, ["auto","pixel","normalized"])
        self.row_Kunits.frame.pack(fill="x", pady=2)
        self.row_meta, _ = labeled_entry_with_btn(box_cam, "Meta文件(txt)", self.meta_path_var, "选择",
                                                  on_click=lambda: self._browse_file(self.meta_path_var, (".txt",)))
        self.row_meta.frame.pack(fill="x", pady=2)
        self.row_pose, _ = labeled_combo(box_cam, "位姿约定", self.pose_convention_var, ["w2c","c2w"])
        self.row_pose.frame.pack(fill="x", pady=2)

        gpus = [-1]
        if torch.cuda.is_available():
            gpus += list(range(torch.cuda.device_count()))
        self.row_gpu, _ = labeled_combo(box_cam, "GPU编号(-1自动)", self.gpu_var, gpus)
        self.row_gpu.frame.pack(fill="x", pady=2)

        self.row_fps, _ = labeled_entry(box_cam, "输出视频帧率", self.fps_var)
        self.row_fps.frame.pack(fill="x", pady=2)

        # Refiner
        box_ref = ttk.LabelFrame(right, text="Refiner模型")
        box_ref.pack(fill="x", padx=6, pady=6)
        self.row_rft, _ = labeled_combo(box_ref, "类型", self.refiner_type_var, ["MGMI","InpaintRefiner"])
        self.row_rft.frame.pack(fill="x", pady=2)
        self.row_ckpt, _ = labeled_entry_with_btn(box_ref, "权重(ckpt)", self.refiner_ckpt_var, "选择",
                                                  on_click=lambda: self._browse_file(self.refiner_ckpt_var, (".pth",".pt",".ckpt",".bin",".safetensors","")))
        self.row_ckpt.frame.pack(fill="x", pady=2)

        # 输出与编码
        box_out = ttk.LabelFrame(right, text="输出与编码")
        box_out.pack(fill="x", padx=6, pady=6)
        self.row_out, _ = labeled_entry_with_btn(box_out, "输出路径", self.out_path_var, "选择", on_click=self._choose_out)
        self.row_out.frame.pack(fill="x", pady=2)
        self.row_writer, self.cb_writer = labeled_combo(box_out, "编码方式", self.writer_kind_var,
                                                        ["OpenCV mp4v","OpenCV lossless","FFmpeg lossless","FFmpeg H.264"])
        self.row_writer.frame.pack(fill="x", pady=2)
        self.row_ffclose, _ = labeled_entry(box_out, "FFmpeg关闭超时(s)", self.ffmpeg_close_timeout_var)
        self.row_ffclose.frame.pack(fill="x", pady=2)
        self.row_qp, _ = labeled_entry(box_out, "H.264 QP(0~51)", self.h264_qp_var)
        self.row_qp.frame.pack(fill="x", pady=2)
        self.row_preset, _ = labeled_combo(box_out, "H.264 preset", self.h264_preset_var,
                                           ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"])
        self.row_preset.frame.pack(fill="x", pady=2)
        self.row_pixfmt, _ = labeled_combo(box_out, "H.264 像素格式", self.h264_pix_fmt_var, ["yuv420p","yuv444p"])
        self.row_pixfmt.frame.pack(fill="x", pady=2)

        # 其他
        box_misc = ttk.LabelFrame(right, text="其他")
        box_misc.pack(fill="x", padx=6, pady=6)
        self.row_focus, _ = labeled_entry(box_misc, "对焦深度(>0直接用)", self.focus_depth_var)
        self.row_focus.frame.pack(fill="x", pady=2)
        self.row_views, _ = labeled_entry_with_btn(box_misc, "保存视角目录(可选)", self.save_views_dir_var, "选择",
                                                   on_click=lambda: self._browse_dir(self.save_views_dir_var))
        self.row_views.frame.pack(fill="x", pady=2)
        self.row_chunk, _ = labeled_entry(box_misc, "分块视角数(0=不分块)", self.chunk_views_var)
        self.row_chunk.frame.pack(fill="x", pady=2)

        nav = ttk.Frame(frm); nav.pack(fill="x", padx=8, pady=8)
        ttk.Button(nav, text="上一步", command=lambda: self.nb.select(self.page_depth)).pack(side="left")
        ttk.Button(nav, text="下一步：运行", command=lambda: self.nb.select(self.page_run)).pack(side="right")

        # 动态显示规则
        self.depth_mode_var.trace_add("write", lambda *_: self._refresh_dynamic_fields())
        self.rescale_depth_var.trace_add("write", lambda *_: self._refresh_dynamic_fields())
        self.writer_kind_var.trace_add("write", lambda *_: self._refresh_dynamic_fields())
        self.mode_var.trace_add("write", lambda *_: self._refresh_dynamic_fields())
        self.manual_K_var.trace_add("write", lambda *_: self._refresh_dynamic_fields())
        self._refresh_dynamic_fields()

    def _refresh_dynamic_fields(self):
        mode = self.mode_var.get()
        writer = self.writer_kind_var.get()
        dmode = self.depth_mode_var.get()
        rescale = self.rescale_depth_var.get()
        manualK = self.manual_K_var.get().strip()

        # 输入模式相关
        if mode == "image":
            self.row_fps.hide()
            self.row_writer.hide()
            self.row_ffclose.hide()
            self.row_qp.hide()
            self.row_preset.hide()
            self.row_pixfmt.hide()
        else:
            self.row_fps.show()
            self.row_writer.show()
            if writer == "OpenCV mp4v":
                self.row_ffclose.hide()
                self.row_qp.hide()
                self.row_preset.hide()
                self.row_pixfmt.hide()
            elif writer == "OpenCV lossless":
                self.row_ffclose.hide()
                self.row_qp.hide()
                self.row_preset.hide()
                self.row_pixfmt.hide()
            elif writer == "FFmpeg lossless":
                self.row_ffclose.show()
                self.row_qp.hide()
                self.row_preset.hide()
                self.row_pixfmt.hide()
            else:  # FFmpeg H.264
                self.row_ffclose.show()
                self.row_qp.show()
                self.row_preset.show()
                self.row_pixfmt.show()

        # 深度模式：metric 不需要深度尺度
        if dmode == "metric":
            self.row_dscale.hide()
        else:
            self.row_dscale.show()

        # 重标定：none 不显示范围和分位点
        if rescale == "none":
            self.row_r_rng.hide()
            self.row_r_pct.hide()
        else:
            self.row_r_rng.show()
            self.row_r_pct.show()

        # Meta 仅对 image/dir 有意义；video 多为手动K或默认
        if mode in ("image","dir"):
            self.row_meta.show()
        else:
            self.row_meta.hide()

        # manual_K 为空则隐藏单位选择
        if manualK:
            self.row_Kunits.show()
        else:
            self.row_Kunits.hide()

    def _choose_out(self):
        mode = self.mode_var.get()
        if mode == "image":
            p = filedialog.asksaveasfilename(title="选择输出图片路径", defaultextension=".png",
                                             filetypes=[("PNG","*.png"),("JPG","*.jpg;*.jpeg"),("All","*.*")])
        else:
            p = filedialog.asksaveasfilename(title="选择输出视频路径", defaultextension=".mp4",
                                             filetypes=[("MP4","*.mp4"),("MKV","*.mkv"),("AVI","*.avi"),("All","*.*")])
        if p: self.out_path_var.set(p)

    def _build_page_run(self):
        frm = self.page_run
        ttk.Label(frm, text="点击下方“开始处理”开始运行。进度与耗时会实时显示。").pack(anchor="w", padx=8, pady=(8,4))
        box = ttk.LabelFrame(frm, text="运行状态"); box.pack(fill="x", padx=8, pady=8)

        self.progress = ttk.Progressbar(box, length=400, mode="determinate")
        self.progress.grid(row=0, column=0, columnspan=4, sticky="we", padx=6, pady=8)

        self.status_var = tk.StringVar(value="尚未开始")
        ttk.Label(box, textvariable=self.status_var, foreground="#007744").grid(row=1, column=0, columnspan=4, sticky="w", padx=6, pady=6)

        ttk.Label(box, text="已用时:").grid(row=2, column=0, sticky="e", padx=6, pady=6)
        self.elapsed_var = tk.StringVar(value="0.00 s")
        ttk.Label(box, textvariable=self.elapsed_var).grid(row=2, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(box, text="输出文件:").grid(row=3, column=0, sticky="e", padx=6, pady=6)
        self.final_out_var = tk.StringVar(value="-")
        # 可点击的输出路径标签：点击打开所在文件夹
        self.final_out_lbl = ttk.Label(box, textvariable=self.final_out_var, foreground="#0044aa", cursor="hand2")
        self.final_out_lbl.grid(row=3, column=1, columnspan=3, sticky="w", padx=6, pady=6)
        self.final_out_lbl.bind("<Button-1>", self._open_out_dir)

        box.columnconfigure(1, weight=1)
        box.columnconfigure(2, weight=1)
        box.columnconfigure(3, weight=1)

        btns = ttk.Frame(frm); btns.pack(fill="x", padx=8, pady=8)
        self.run_btn = ttk.Button(btns, text="开始处理", command=self._start_run)
        self.run_btn.pack(side="left")
        ttk.Button(btns, text="上一步", command=lambda: self.nb.select(self.page_params)).pack(side="left", padx=8)
        self.quit_btn = ttk.Button(btns, text="确认退出", command=self.destroy, state="disabled")
        self.quit_btn.pack(side="right")

    def _collect_params(self) -> RunParams:
        rp = RunParams(mode=self.mode_var.get())
        rp.image_path = self.rgb_path_var.get().strip()
        rp.depth_path = self.depth_path_var.get().strip()
        rp.pair_dir = self.dir_path_var.get().strip()
        rp.video_path = self.video_path_var.get().strip()
        rp.depth_video_path = self.depth_video_path_var.get().strip()

        rp.img_size = self.img_size_var.get().strip()
        rp.fuse_size = self.fuse_size_var.get().strip()
        rp.manual_K = self.manual_K_var.get().strip()
        rp.K_units = self.K_units_var.get()
        rp.meta_path = self.meta_path_var.get().strip()
        rp.pose_convention = self.pose_convention_var.get()
        rp.fps = float(self.fps_var.get())

        rp.depth_mode = self.depth_mode_var.get()
        rp.depth_scale = float(self.depth_scale_var.get())
        rp.depth_color_decode = self.depth_color_decode_var.get()
        rp.far_value = self.far_value_var.get()

        rp.rescale_depth = self.rescale_depth_var.get()
        rp.rescale_range = self.rescale_range_var.get().strip()
        rp.rescale_percentiles = self.rescale_percentiles_var.get().strip()

        rp.num_per_side = int(self.num_per_side_var.get())
        rp.spacing = self.spacing_var.get()
        rp.tx_max = float(self.tx_max_var.get())
        rp.max_disp_px = float(self.max_disp_px_var.get())
        rp.disp_ref_percentile = float(self.disp_ref_percentile_var.get())

        rp.temperature = float(self.temperature_var.get())
        rp.occlusion = self.occlusion_var.get()
        rp.hard_z_epsilon = float(self.hard_z_epsilon_var.get())
        rp.amp = bool(self.amp_var.get())

        rp.refiner_type = self.refiner_type_var.get()
        rp.refiner_ckpt = self.refiner_ckpt_var.get().strip()

        rp.slope = float(self.slope_var.get())
        rp.X = float(self.X_var.get())
        rp.koff = float(self.koff_var.get())

        rp.focus_depth = float(self.focus_depth_var.get())
        if rp.focus_depth <= 0.0 and self.selected_focus_depth > 0.0:
            rp.focus_depth = float(self.selected_focus_depth)

        rp.out_path = self.out_path_var.get().strip()

        rp.gpu = int(self.gpu_var.get())
        rp.save_views_dir = self.save_views_dir_var.get().strip()
        rp.chunk_views = int(self.chunk_views_var.get())

        rp.writer_kind = self.writer_kind_var.get()
        rp.ffmpeg_close_timeout = float(self.ffmpeg_close_timeout_var.get())
        rp.h264_qp = int(self.h264_qp_var.get())
        rp.h264_preset = self.h264_preset_var.get()
        rp.h264_pix_fmt = self.h264_pix_fmt_var.get()
        return rp

    def _start_run(self):
        # 校验
        try:
            rp = self._collect_params()
        except Exception as e:
            messagebox.showerror("错误", f"参数错误：{e}")
            return
        if not rp.refiner_ckpt or not os.path.isfile(rp.refiner_ckpt):
            messagebox.showerror("错误", "请指定有效的 Refiner 权重文件。")
            return
        if not rp.out_path:
            messagebox.showerror("错误", "请指定输出路径。")
            return
        if rp.mode == "image":
            if not (os.path.isfile(rp.image_path) and os.path.isfile(rp.depth_path)):
                messagebox.showerror("错误", "图像模式需要“原图(图片)”与“深度(图片)”。")
                return
        elif rp.mode == "dir":
            if not os.path.isdir(rp.pair_dir):
                messagebox.showerror("错误", "目录模式需要有效的成对目录。")
                return
        else:
            if not (os.path.isfile(rp.video_path) and is_video_file(rp.video_path)):
                messagebox.showerror("错误", "请提供有效的 RGB 视频文件。")
                return
            if not (os.path.isfile(rp.depth_video_path) or os.path.isdir(rp.depth_video_path)):
                messagebox.showerror("错误", "请提供有效的深度视频（文件或目录）。")
                return

        # UI状态
        self._reset_progress()
        self.status_var.set("初始化中...")
        self.run_btn.config(state="disabled")
        self.quit_btn.config(state="disabled")
        self.start_time = time.time()
        self._start_elapsed_tick()

        # 包装 emit：确保队列中始终是 (tag, payload) 二元组
        def emit(tag, payload):
            try:
                self.msg_queue.put((tag, payload))
            except Exception as e:
                self.msg_queue.put(("error", f"emit失败: {e}"))

        def worker():
            try:
                out_path = run_pipeline(rp, emit)
                self.msg_queue.put(("done", out_path))
            except Exception as e:
                self.msg_queue.put(("error", str(e)))

        self.run_thread = threading.Thread(target=worker, daemon=True)
        self.run_thread.start()

    def _reset_progress(self):
        self.done_steps = 0
        self.total_steps = 0
        self.progress.config(mode="determinate", maximum=100, value=0)
        self.elapsed_var.set("0.00 s")
        self.final_out_var.set("-")

    def _start_elapsed_tick(self):
        if self._elapsed_ticking:
            return
        self._elapsed_ticking = True
        def tick():
            if self.start_time is not None:
                el = time.time() - self.start_time
                self.elapsed_var.set(f"{el:.2f} s")
            if self.run_thread is not None and self.run_thread.is_alive():
                self.after(200, tick)
            else:
                self._elapsed_ticking = False
        self.after(200, tick)

    def _open_out_dir(self, event=None):
        """
        打开输出文件所在的文件夹（若是目录则直接打开该目录）
        """
        path = self.final_out_var.get().strip()
        if not path or path in ("-", ""):
            return
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if not folder:
            folder = os.getcwd()
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件夹：{folder}\n{e}")

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if not isinstance(msg, tuple) or len(msg) < 2:
                    tag, payload = "status", str(msg)
                else:
                    tag, payload = msg[0], msg[1]

                if tag == "progress_total":
                    self.total_steps = int(payload)
                    if self.total_steps <= 0:
                        self.progress.config(mode="indeterminate")
                        self.progress.start(10)
                    else:
                        self.progress.stop()
                        self.progress.config(mode="determinate", maximum=self.total_steps, value=0)
                elif tag == "progress_step":
                    self.done_steps += 1
                    if self.total_steps > 0:
                        self.progress.config(value=self.done_steps)
                elif tag == "status":
                    self.status_var.set(str(payload))
                elif tag == "done":
                    self.status_var.set("完成")
                    self.final_out_var.set(str(payload))
                    self.run_btn.config(state="normal")
                    self.quit_btn.config(state="normal")
                    self.progress.stop()
                elif tag == "error":
                    self.status_var.set("出错")
                    self.run_btn.config(state="normal")
                    self.quit_btn.config(state="normal")
                    self.progress.stop()
                    messagebox.showerror("错误", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

def main():
    app = App()
    app.mainloop()
