# depth_warp_vs/ui/widgets.py
import tkinter as tk
from tkinter import ttk

class Row:
    def __init__(self, frame: ttk.Frame):
        self.frame = frame
        self.visible = True
        # 统一使用 pack 管理行容器（行内部控件仍可用 grid，不冲突）
        # 初次由外部 pack() 布局；后续 show()/hide() 用 pack/pack_forget 控制
    def show(self):
        if not self.visible:
            # 使用与外部一致的缺省 pack 参数
            # 大多数行初次 pack 时使用了 fill="x", pady=2
            # 这里统一用 fill="x", pady=2，若需要自定义可在创建后再次手动 pack
            self.frame.pack(fill="x", pady=2)
            self.visible = True
    def hide(self):
        if self.visible:
            self.frame.pack_forget()
            self.visible = False

def labeled_entry(parent, label, var, width=30, hint=None):
    fr = ttk.Frame(parent)
    ttk.Label(fr, text=label + ":").grid(row=0, column=0, sticky="e", padx=6)
    e = ttk.Entry(fr, textvariable=var, width=width)
    e.grid(row=0, column=1, sticky="we", padx=6)
    if hint:
        ttk.Label(fr, text=hint, foreground="#888888").grid(row=0, column=2, sticky="w", padx=6)
    fr.columnconfigure(1, weight=1)
    return Row(fr), e

def labeled_combo(parent, label, var, values, width=30):
    fr = ttk.Frame(parent)
    ttk.Label(fr, text=label + ":").grid(row=0, column=0, sticky="e", padx=6)
    cb = ttk.Combobox(fr, textvariable=var, values=list(values), state="readonly", width=width)
    cb.grid(row=0, column=1, sticky="we", padx=6)
    fr.columnconfigure(1, weight=1)
    return Row(fr), cb

def labeled_check(parent, label, var):
    fr = ttk.Frame(parent)
    ck = ttk.Checkbutton(fr, text=label, variable=var)
    ck.grid(row=0, column=0, sticky="w", padx=6)
    return Row(fr), ck

def labeled_entry_with_btn(parent, label, var, btn_text="浏览", on_click=None, width=40):
    fr = ttk.Frame(parent)
    ttk.Label(fr, text=label + ":").grid(row=0, column=0, sticky="e", padx=6)
    e = ttk.Entry(fr, textvariable=var, width=width)
    e.grid(row=0, column=1, sticky="we", padx=6)
    b = ttk.Button(fr, text=btn_text, command=on_click)
    b.grid(row=0, column=2, sticky="w", padx=6)
    fr.columnconfigure(1, weight=1)
    return Row(fr), (e, b)

