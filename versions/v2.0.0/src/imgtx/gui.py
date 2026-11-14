"""Giao diện desktop ImgTX (Tkinter + Matplotlib) – FR11, chạy offline, đóng gói được thành .exe."""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import APP_NAME, __version__, viz
from .analysis import edge_map_topk, sobel
from .degrade import DEFAULT_PARAMS, DEGRADE_LABELS, build_pair
from .experiment import CS, GAINS, GAMMAS, load_config, run_one, sweep
from .io_utils import (read_rgb, resize_max, resource_path, sample_images, save_csv, save_image,
                       to_display, to_working, write_synthetic)

ACCENT = "#1f6fb2"
BG = "#f4f6fa"
PANEL = "#ffffff"
TEXT = "#1d2433"
MUTED = "#6b7487"

TRANSFORM_LABELS = {"negative": "Âm bản", "log": "Logarit", "gamma": "Gamma"}
METRICS = [("mean", "Mean (độ sáng)"), ("std", "Std (tương phản)"), ("entropy", "Entropy (bit)"),
           ("dyn_range", "Dải động P99−P1"), ("sat_high", "Tỉ lệ bão hoà 255"),
           ("f1_topk", "F1 biên (top-k)"), ("iou_topk", "IoU biên"), ("grad_corr", "Tương quan gradient"),
           ("mean_grad_norm", "Gradient TB chuẩn hoá"), ("edge_density_otsu", "Mật độ biên Otsu"),
           ("mse", "MSE"), ("psnr", "PSNR (dB)")]


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"


class Slider(ttk.Frame):
    """Thanh trượt có nhãn + ô giá trị."""

    def __init__(self, master, text, frm, to, value, step, command, fmt="{:.2f}"):
        super().__init__(master, style="Panel.TFrame")
        self.step, self.fmt, self.command = step, fmt, command
        self.var = tk.DoubleVar(value=value)
        top = ttk.Frame(self, style="Panel.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=text, style="Panel.TLabel").pack(side="left")
        self.lbl = ttk.Label(top, text=fmt.format(value), style="Value.TLabel")
        self.lbl.pack(side="right")
        self.scale = ttk.Scale(self, from_=frm, to=to, variable=self.var, command=self._moved)
        self.scale.pack(fill="x")

    def _moved(self, _=None):
        v = round(self.var.get() / self.step) * self.step
        self.lbl.configure(text=self.fmt.format(v))
        self.command()

    def get(self):
        return round(round(self.var.get() / self.step) * self.step, 6)

    def set(self, v):
        self.var.set(v)
        self.lbl.configure(text=self.fmt.format(v))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}  v{__version__}")
        self.geometry("1360x860")
        self.minsize(1100, 700)
        self.configure(bg=BG)
        self._set_icon()
        self._style()

        self.rgb = None
        self.ref_goc = self.ref_in = self.out = None
        self.row = self.base_row = None
        self.sweep_df = None
        self._pending = None
        self.q = queue.Queue()

        self._build_header()
        body = ttk.Frame(self, style="Bg.TFrame")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._build_controls(body)
        self._build_tabs(body)
        self.status = ttk.Label(self, text="Sẵn sàng.", style="Status.TLabel", anchor="w")
        self.status.pack(fill="x", side="bottom")

        samples = sample_images()
        self.samples = samples
        self.cb_sample["values"] = list(samples)
        if samples:
            self.cb_sample.set("camera" if "camera" in samples else list(samples)[0])
            self.load_sample()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------ giao diện
    def _set_icon(self):
        try:
            if sys.platform.startswith("win"):
                self.iconbitmap(default=resource_path("assets", "imgtx.ico"))
            self._icon_img = tk.PhotoImage(file=resource_path("assets", "logo.png")).subsample(8, 8)
            self.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        base = ("Segoe UI", 10)
        self.option_add("*Font", base)
        s.configure(".", font=base, background=BG, foreground=TEXT)
        s.configure("Bg.TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("Header.TFrame", background=ACCENT)
        s.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        s.configure("Value.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        s.configure("Section.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        s.configure("Hint.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 8))
        s.configure("Title.TLabel", background=ACCENT, foreground="white", font=("Segoe UI", 15, "bold"))
        s.configure("Sub.TLabel", background=ACCENT, foreground="#d7e6f7", font=("Segoe UI", 9))
        s.configure("Status.TLabel", background="#e6eaf2", foreground=MUTED, padding=(10, 3))
        s.configure("Panel.TRadiobutton", background=PANEL)
        s.configure("Panel.TCheckbutton", background=PANEL)
        s.configure("TButton", padding=(10, 5))
        s.configure("Accent.TButton", background=ACCENT, foreground="white", padding=(10, 6))
        s.map("Accent.TButton", background=[("active", "#185a92"), ("disabled", "#9db6cf")])
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(14, 6))
        s.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        s.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
        s.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        s.configure("Horizontal.TProgressbar", background=ACCENT)

    def _build_header(self):
        h = ttk.Frame(self, style="Header.TFrame", padding=(14, 8))
        h.pack(fill="x", pady=(0, 10))
        try:
            self._logo = tk.PhotoImage(file=resource_path("assets", "logo.png")).subsample(12, 12)
            tk.Label(h, image=self._logo, bg=ACCENT).pack(side="left", padx=(0, 10))
        except Exception:
            pass
        box = ttk.Frame(h, style="Header.TFrame")
        box.pack(side="left")
        ttk.Label(box, text="ImgTX – Khảo sát biến đổi cường độ", style="Title.TLabel").pack(anchor="w")
        ttk.Label(box, text="Âm bản · Logarit · Gamma  |  Histogram  |  So sánh bảo toàn biên Sobel",
                  style="Sub.TLabel").pack(anchor="w")

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", pady=(10, 3))

    def _build_controls(self, body):
        outer = ttk.Frame(body, style="Panel.TFrame", padding=12)
        outer.pack(side="left", fill="y", padx=(0, 10))
        p = outer

        self._section(p, "1. Nguồn ảnh")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Mở ảnh…", command=self.open_image).pack(side="left")
        self.cb_sample = ttk.Combobox(row, state="readonly", width=12)
        self.cb_sample.pack(side="left", padx=(6, 0))
        self.cb_sample.bind("<<ComboboxSelected>>", lambda e: self.load_sample())
        self.lbl_file = ttk.Label(p, text="", style="Hint.TLabel", wraplength=250)
        self.lbl_file.pack(anchor="w", pady=(2, 0))

        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Cạnh dài tối đa", style="Panel.TLabel").pack(side="left")
        self.cb_size = ttk.Combobox(row, state="readonly", width=6, values=["256", "512", "768", "1024"])
        self.cb_size.set("512")
        self.cb_size.pack(side="right")
        self.cb_size.bind("<<ComboboxSelected>>", lambda e: self.prepare())
        self.mode = tk.StringVar(value="gray")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Radiobutton(row, text="Ảnh xám", value="gray", variable=self.mode, style="Panel.TRadiobutton",
                        command=self.prepare).pack(side="left")
        ttk.Radiobutton(row, text="Màu (kênh V)", value="v", variable=self.mode, style="Panel.TRadiobutton",
                        command=self.prepare).pack(side="left", padx=(10, 0))

        self._section(p, "2. Ảnh tham chiếu (mục 8.2)")
        self.deg_keys = ["none"] + list(DEGRADE_LABELS)
        self.cb_deg = ttk.Combobox(p, state="readonly", width=32,
                                   values=["Không – ảnh thật (so với ảnh đầu vào)"] + list(DEGRADE_LABELS.values()))
        self.cb_deg.current(1)
        self.cb_deg.pack(fill="x")
        self.cb_deg.bind("<<ComboboxSelected>>", lambda e: self._deg_changed())
        self.deg_box = ttk.Frame(p, style="Panel.TFrame")
        self.deg_box.pack(fill="x")
        self.s_alpha = Slider(self.deg_box, "α", 0.1, 2.5, 0.4, 0.05, self.prepare_later)
        self.s_beta = Slider(self.deg_box, "β", 0, 100, 10, 1, self.prepare_later, "{:.0f}")
        self.s_sigma = Slider(self.deg_box, "σ nhiễu", 0, 15, 4, 0.5, self.prepare_later, "{:.1f}")
        self.s_dg = Slider(self.deg_box, "γ làm tối (đối chứng)", 1.2, 4.0, 2.5, 0.1, self.prepare_later, "{:.1f}")
        self.hint_deg = ttk.Label(p, text="", style="Hint.TLabel", wraplength=250)
        self.hint_deg.pack(anchor="w")

        self._section(p, "3. Phép biến đổi")
        self.tf = tk.StringVar(value="gamma")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x")
        for k, v in TRANSFORM_LABELS.items():
            ttk.Radiobutton(row, text=v, value=k, variable=self.tf, style="Panel.TRadiobutton",
                            command=self._tf_changed).pack(side="left", padx=(0, 8))
        self.param_box = ttk.Frame(p, style="Panel.TFrame")
        self.param_box.pack(fill="x")
        self.s_gain = Slider(self.param_box, "gain (hệ số c của log)", 0.2, 2.0, 1.0, 0.05, self.update_later)
        self.s_gamma = Slider(self.param_box, "γ", 0.1, 4.0, 0.5, 0.05, self.update_later)
        self.s_c = Slider(self.param_box, "c (hệ số gamma)", 0.25, 3.0, 1.0, 0.05, self.update_later)
        self.s_k = Slider(p, "Mật độ biên k (top-k)", 0.02, 0.30, 0.10, 0.01, self.update_later)
        self.s_k.pack(fill="x", pady=(6, 0))

        self._section(p, "4. Xuất kết quả")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Lưu ảnh kết quả", command=self.save_output).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Xuất chỉ số", command=self.save_metrics).pack(side="left", fill="x",
                                                                              expand=True, padx=(6, 0))
        self._deg_changed(initial=True)
        self._tf_changed(initial=True)

    def _figtab(self, nb, title, w, h, toolbar=False):
        frame = ttk.Frame(nb, style="Panel.TFrame", padding=6)
        nb.add(frame, text=title)
        fig = Figure(figsize=(w, h), dpi=100, layout="tight", facecolor="white")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        if toolbar:
            NavigationToolbar2Tk(canvas, frame, pack_toolbar=True).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return frame, fig, canvas

    def _build_tabs(self, body):
        nb = ttk.Notebook(body)
        nb.pack(side="left", fill="both", expand=True)
        self.nb = nb

        # Tab 1: ảnh & biên + bảng chỉ số
        f1 = ttk.Frame(nb, style="Panel.TFrame", padding=6)
        nb.add(f1, text="  Ảnh & biên Sobel  ")
        self.fig_img = Figure(figsize=(8, 5), dpi=100, layout="tight", facecolor="white")
        self.cv_img = FigureCanvasTkAgg(self.fig_img, master=f1)
        self.cv_img.get_tk_widget().pack(fill="both", expand=True)
        tv_box = ttk.Frame(f1, style="Panel.TFrame")
        tv_box.pack(fill="x")
        self.tv = ttk.Treeview(tv_box, columns=("m", "base", "out", "d"), show="headings", height=6)
        for col, txt, w in (("m", "Chỉ số", 200), ("base", "Identity (ảnh đầu vào)", 170),
                            ("out", "Sau biến đổi", 150), ("d", "Chênh lệch", 120)):
            self.tv.heading(col, text=txt)
            self.tv.column(col, width=w, anchor="w" if col == "m" else "e")
        sb = ttk.Scrollbar(tv_box, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

        # Tab 2: histogram & T(r)
        _, self.fig_hist, self.cv_hist = self._figtab(nb, "  Histogram & đường cong T(r)  ", 9, 4)

        # Tab 3: quét tham số
        f3 = ttk.Frame(nb, style="Panel.TFrame", padding=6)
        nb.add(f3, text="  Quét tham số (TN2/TN3/TN7)  ")
        bar = ttk.Frame(f3, style="Panel.TFrame")
        bar.pack(fill="x")
        ttk.Button(bar, text="Quét γ (c hiện tại) – TN2", command=lambda: self.run_sweep("gamma")).pack(side="left")
        ttk.Button(bar, text="Quét gain log – TN3", command=lambda: self.run_sweep("log")).pack(side="left", padx=6)
        ttk.Button(bar, text="Heatmap γ × c – TN7", command=lambda: self.run_sweep("grid")).pack(side="left")
        ttk.Button(bar, text="Xuất CSV lần quét", command=self.save_sweep).pack(side="right")
        self.lbl_sweep = ttk.Label(f3, text="Bấm một nút để quét trên ảnh hiện tại. Đường chấm = dòng identity (mốc).",
                                   style="Hint.TLabel")
        self.lbl_sweep.pack(anchor="w", pady=4)
        self.fig_sweep = Figure(figsize=(9, 4), dpi=100, layout="tight", facecolor="white")
        self.cv_sweep = FigureCanvasTkAgg(self.fig_sweep, master=f3)
        NavigationToolbar2Tk(self.cv_sweep, f3, pack_toolbar=True).update()
        self.cv_sweep.get_tk_widget().pack(fill="both", expand=True)

        # Tab 4: thí nghiệm hàng loạt
        self._build_batch_tab(nb)

        # Tab 5: giới thiệu
        f5 = ttk.Frame(nb, style="Panel.TFrame", padding=16)
        nb.add(f5, text="  Hướng dẫn  ")
        txt = tk.Text(f5, wrap="word", relief="flat", bg=PANEL, font=("Segoe UI", 10), height=30)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

    def _build_batch_tab(self, nb):
        f = ttk.Frame(nb, style="Panel.TFrame", padding=12)
        nb.add(f, text="  Thí nghiệm hàng loạt  ")
        ttk.Label(f, text="Chạy toàn bộ lưới TN1–TN8 như lệnh  python -m imgtx.cli  và xuất CSV + hình cho báo cáo.",
                  style="Panel.TLabel").pack(anchor="w")
        self.batch_vars = {}
        rows = [("gt", "Thư mục ảnh CÓ ảnh gốc (sẽ tạo cặp suy giảm)", "Để trống = dùng ảnh mẫu đi kèm"),
                ("real", "Thư mục ảnh thật (không có ảnh gốc) – tuỳ chọn", ""),
                ("out", "Thư mục lưu kết quả", "")]
        default_out = os.path.join(os.path.expanduser("~"), "Documents", "ImgTX_ketqua")
        for key, label, hint in rows:
            box = ttk.Frame(f, style="Panel.TFrame")
            box.pack(fill="x", pady=(10, 0))
            ttk.Label(box, text=label, style="Panel.TLabel").pack(anchor="w")
            line = ttk.Frame(box, style="Panel.TFrame")
            line.pack(fill="x")
            v = tk.StringVar(value=default_out if key == "out" else "")
            self.batch_vars[key] = v
            ttk.Entry(line, textvariable=v).pack(side="left", fill="x", expand=True)
            ttk.Button(line, text="Chọn…", command=lambda v=v: self._pick_dir(v)).pack(side="left", padx=(6, 0))
            if hint:
                ttk.Label(box, text=hint, style="Hint.TLabel").pack(anchor="w")
        opt = ttk.Frame(f, style="Panel.TFrame")
        opt.pack(fill="x", pady=(10, 0))
        self.var_syn = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Thêm ảnh tổng hợp (thang xám, bậc thang, hình học)", variable=self.var_syn,
                        style="Panel.TCheckbutton").pack(side="left")
        self.var_k = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Chạy TN6 (độ nhạy ngưỡng k)", variable=self.var_k,
                        style="Panel.TCheckbutton").pack(side="left", padx=12)
        btns = ttk.Frame(f, style="Panel.TFrame")
        btns.pack(fill="x", pady=10)
        self.btn_run = ttk.Button(btns, text="Chạy thí nghiệm", style="Accent.TButton", command=self.run_batch)
        self.btn_run.pack(side="left")
        ttk.Button(btns, text="Mở thư mục kết quả", command=self.open_out).pack(side="left", padx=6)
        self.pb = ttk.Progressbar(f, mode="determinate", style="Horizontal.TProgressbar")
        self.pb.pack(fill="x")
        self.log = tk.Text(f, height=16, wrap="word", relief="flat", bg="#f7f8fb", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, pady=(8, 0))

    # ------------------------------------------------------------------ sự kiện
    def _deg_changed(self, initial=False):
        key = self.deg_keys[self.cb_deg.current()]
        for w in (self.s_alpha, self.s_beta, self.s_sigma, self.s_dg):
            w.pack_forget()
        hints = {
            "none": "Ảnh đang chọn là ảnh đầu vào; F1/IoU chỉ đo mức giống ảnh đầu vào, không có MSE/PSNR.",
            "offset": "Ảnh chọn = ảnh gốc; đầu vào = clip(α·r − β). Suy giảm ngoài họ gamma/log.",
            "noisy": "Như offset + nhiễu Gauss σ (seed 0).",
            "bright": "Quá sáng: clip(α·r + β), vùng sáng bị cắt 255.",
            "control_gamma": "CHỈ ĐỐI CHỨNG thiên vị (TN8): gamma 1/γ khôi phục gần hoàn hảo.",
        }
        if key in ("offset", "noisy", "bright"):
            d = DEFAULT_PARAMS[key]
            self.s_alpha.set(d["alpha"])
            self.s_beta.set(d["beta"])
            self.s_alpha.pack(fill="x")
            self.s_beta.pack(fill="x")
            if key == "noisy":
                self.s_sigma.set(d["sigma"])
                self.s_sigma.pack(fill="x")
        elif key == "control_gamma":
            self.s_dg.pack(fill="x")
        self.hint_deg.configure(text=hints[key])
        if not initial:
            self.prepare()

    def _tf_changed(self, initial=False):
        for w in (self.s_gain, self.s_gamma, self.s_c):
            w.pack_forget()
        t = self.tf.get()
        if t == "log":
            self.s_gain.pack(fill="x")
        elif t == "gamma":
            self.s_gamma.pack(fill="x")
            self.s_c.pack(fill="x")
        if not initial:
            self.update_view()

    def _debounce(self, fn):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(150, fn)

    def update_later(self):
        self._debounce(self.update_view)

    def prepare_later(self):
        self._debounce(self.prepare)

    def open_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[
            ("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Tất cả", "*.*")])
        if path:
            self.load_path(path)

    def load_sample(self):
        name = self.cb_sample.get()
        if name in self.samples:
            self.load_path(self.samples[name])

    def load_path(self, path):
        try:
            self.rgb = read_rgb(path)
        except Exception as e:
            messagebox.showerror("Lỗi đọc ảnh", str(e))
            return
        self.lbl_file.configure(text=f"{os.path.basename(path)}  ({self.rgb.shape[1]}×{self.rgb.shape[0]})")
        self.prepare()

    def _deg_params(self, key):
        if key in ("offset", "bright"):
            return dict(alpha=self.s_alpha.get(), beta=self.s_beta.get())
        if key == "noisy":
            return dict(alpha=self.s_alpha.get(), beta=self.s_beta.get(), sigma=self.s_sigma.get(), seed=0)
        if key == "control_gamma":
            return dict(gamma=self.s_dg.get())
        return {}

    def prepare(self):
        """Tiền xử lý: resize, chuyển xám/kênh V, tạo cặp (ảnh gốc, ảnh đầu vào)."""
        if self.rgb is None:
            return
        self.small = resize_max(self.rgb, int(self.cb_size.get()))
        work = to_working(self.small, self.mode.get())
        key = self.deg_keys[self.cb_deg.current()]
        self.ref_goc, self.ref_in = build_pair(work, key, **self._deg_params(key))
        self.sweep_df = None
        self.update_view()

    def _params(self):
        t = self.tf.get()
        if t == "log":
            return {"gain": self.s_gain.get()}
        if t == "gamma":
            return {"gamma": self.s_gamma.get(), "c": self.s_c.get()}
        return {}

    def update_view(self):
        if self.ref_in is None:
            return
        k = self.s_k.get()
        t, p = self.tf.get(), self._params()
        self.out, self.row = run_one(t, self.ref_goc, self.ref_in, k, **p)
        _, self.base_row = run_one("identity", self.ref_goc, self.ref_in, k)
        self._draw_images(k)
        self._draw_hist(p)
        self._fill_table()
        desc = TRANSFORM_LABELS[t] + ("" if not p else " (" + ", ".join(f"{a}={b:g}" for a, b in p.items()) + ")")
        self.status.configure(text=f"{desc}  ·  tham chiếu: {'ảnh gốc' if self.ref_goc is not None else 'ảnh đầu vào'}"
                                   f"  ·  kích thước {self.ref_in.shape[1]}×{self.ref_in.shape[0]}")

    def _show(self, ax, img, title, cmap="gray"):
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=255 if cmap == "gray" else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def _draw_images(self, k):
        f = self.fig_img
        f.clear()
        color = self.mode.get() == "v"
        disp = (lambda w: to_display(self.small, w, "v")) if color else (lambda w: w)
        ref = self.ref_goc if self.ref_goc is not None else self.ref_in
        mag_ref, mag_out = sobel(ref)[0], sobel(self.out)[0]
        axes = f.subplots(2, 3)
        self._show(axes[0, 0], disp(ref), "Ảnh gốc (tham chiếu)" if self.ref_goc is not None else "Ảnh đầu vào (tham chiếu)")
        if self.ref_goc is not None:
            self._show(axes[0, 1], disp(self.ref_in), "Ảnh đầu vào (đã suy giảm)")
        else:
            self._show(axes[0, 1], mag_ref, "|∇| ảnh đầu vào", cmap="magma")
        self._show(axes[0, 2], disp(self.out), "Kết quả – " + TRANSFORM_LABELS[self.tf.get()])
        self._show(axes[1, 0], edge_map_topk(mag_ref, k), f"Biên tham chiếu (top {k:.0%})")
        self._show(axes[1, 1], edge_map_topk(mag_out, k), f"Biên kết quả – F1 = {self.row['f1_topk']:.3f}")
        self._show(axes[1, 2], mag_out, f"|∇| kết quả – tương quan = {self.row['grad_corr']:.3f}", cmap="magma")
        self.cv_img.draw_idle()

    def _draw_hist(self, p):
        f = self.fig_hist
        f.clear()
        ax1, ax2 = f.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1]})
        imgs = {}
        if self.ref_goc is not None:
            imgs["Ảnh gốc"] = self.ref_goc
        imgs["Ảnh đầu vào"] = self.ref_in
        imgs["Kết quả"] = self.out
        colors = [viz.GRAY, viz.ORANGE, viz.BLUE] if len(imgs) == 3 else [viz.ORANGE, viz.BLUE]
        for (name, img), col in zip(imgs.items(), colors):
            h = np.bincount(img.ravel(), minlength=256) / img.size
            ax1.fill_between(np.arange(256), h, step="mid", alpha=0.25, color=col)
            ax1.plot(np.arange(256), h, lw=1.1, color=col, label=f"{name}  μ={img.mean():.1f}  σ={img.std():.1f}")
        ax1.set(xlim=(0, 255), xlabel="mức xám r", ylabel="p(r)", title="Histogram chuẩn hoá")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.25)
        r = np.arange(256, dtype=np.uint8)
        ax2.plot(r, r, ":", color=viz.GRAY, label="identity")
        from .transforms import TRANSFORMS
        t = self.tf.get()
        ax2.plot(r, TRANSFORMS[t](r, **p), color=viz.BLUE, lw=2, label=TRANSFORM_LABELS[t])
        ax2.set(xlim=(0, 255), ylim=(0, 260), xlabel="r", ylabel="s = T(r)", title="Đường cong biến đổi")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.25)
        self.cv_hist.draw_idle()

    def _fill_table(self):
        self.tv.delete(*self.tv.get_children())
        for key, label in METRICS:
            b, o = self.base_row.get(key), self.row.get(key)
            try:
                d = o - b if np.isfinite(o) and np.isfinite(b) else None
            except TypeError:
                d = None
            self.tv.insert("", "end", values=(label, _fmt(b), _fmt(o), ("+" if d and d > 0 else "") + _fmt(d)
                                              if d is not None else "—"))

    # ------------------------------------------------------------------ quét tham số
    def run_sweep(self, kind):
        if self.ref_in is None:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        k = self.s_k.get()
        try:
            base = run_one("identity", self.ref_goc, self.ref_in, k)[1]
            if kind == "gamma":
                c = self.s_c.get()
                df = sweep(self.ref_goc, self.ref_in, "gamma", {"gamma": GAMMAS, "c": [c]}, k)
                panels = [[("mean", "mean")], [("entropy", "entropy")], [("f1_topk", "F1"), ("grad_corr", "grad_corr")]]
                if self.ref_goc is not None:
                    panels.append([("psnr", "PSNR (dB)")])
                viz.plot_sweep(df, "gamma", panels, base, self.s_gamma.get(), fig=self.fig_sweep)
                msg = f"TN2 – quét γ với c = {c:g}."
            elif kind == "log":
                df = sweep(self.ref_goc, self.ref_in, "log", {"gain": GAINS + [1.75, 2.0]}, k)
                panels = [[("sat_high", "tỉ lệ bão hoà 255")], [("dyn_range", "dải động")],
                          [("f1_topk", "F1"), ("grad_corr", "grad_corr")]]
                if self.ref_goc is not None:
                    panels.append([("psnr", "PSNR (dB)")])
                viz.plot_sweep(df, "gain", panels, base, self.s_gain.get(), fig=self.fig_sweep)
                msg = "TN3 – quét gain của log."
            else:
                df = sweep(self.ref_goc, self.ref_in, "gamma", {"gamma": GAMMAS, "c": CS}, k)
                vals = [("f1_topk", "F1 biên"), ("grad_corr", "Tương quan ∇"), ("mean", "Mean")]
                if self.ref_goc is not None:
                    vals.insert(2, ("psnr", "PSNR (dB)"))
                viz.plot_heatmaps(df, vals, fig=self.fig_sweep)
                g = df[df.psnr.notna()] if self.ref_goc is not None else df
                msg = "TN7 – heatmap γ × c."
                if self.ref_goc is not None and len(g):
                    b = g.loc[g.psnr.idxmax()]
                    msg += (f"  PSNR cao nhất: γ={b.gamma:g}, c={b.c:g} → {b.psnr:.1f} dB, F1 {b.f1_topk:.3f} "
                            f"(identity: {base['psnr']:.1f} dB, F1 {base['f1_topk']:.3f}).")
            self.sweep_df = df
            self.lbl_sweep.configure(text=msg + "  Đường chấm = identity, nét đứt = tham số đang chọn.")
            self.cv_sweep.draw_idle()
            self.nb.select(2)
        finally:
            self.config(cursor="")

    def save_sweep(self):
        if self.sweep_df is None:
            messagebox.showinfo("Chưa có dữ liệu", "Hãy chạy một lần quét trước.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="quet_tham_so.csv")
        if path:
            save_csv(self.sweep_df, path)
            self.status.configure(text=f"Đã lưu {path}")

    # ------------------------------------------------------------------ xuất
    def save_output(self):
        if self.out is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="ket_qua.png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            img = to_display(self.small, self.out, "v") if self.mode.get() == "v" else self.out
            save_image(path, img)
            base, ext = os.path.splitext(path)
            save_image(base + "_bien" + ext, (edge_map_topk(sobel(self.out)[0], self.s_k.get()) * 255).astype(np.uint8))
            self.status.configure(text=f"Đã lưu {path} và bản đồ biên.")

    def save_metrics(self):
        if self.row is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="chi_so.csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            save_csv(pd.DataFrame([dict(base_row="identity", **self.base_row), dict(base_row="ket_qua", **self.row)]), path)
            self.status.configure(text=f"Đã lưu {path}")

    # ------------------------------------------------------------------ hàng loạt (luồng nền)
    def _pick_dir(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def run_batch(self):
        out = self.batch_vars["out"].get().strip()
        if not out:
            messagebox.showwarning("Thiếu thư mục", "Chọn thư mục lưu kết quả.")
            return
        cfg = load_config(None)
        gt_dir = self.batch_vars["gt"].get().strip() or resource_path("data", "raw")
        images = [{"path": os.path.join(gt_dir, "*"), "group": "co_anh_goc", "has_gt": True}]
        if self.var_syn.get():
            syn = os.path.join(out, "_anh_tong_hop")
            write_synthetic(syn)
            images.append({"path": os.path.join(syn, "*"), "group": "tong_hop", "has_gt": True})
        if self.batch_vars["real"].get().strip():
            images.append({"path": os.path.join(self.batch_vars["real"].get().strip(), "*"),
                           "group": "anh_that", "has_gt": False})
        cfg["images"] = images
        cfg["edge"]["density"] = self.s_k.get()
        self.btn_run.state(["disabled"])
        self.log.delete("1.0", "end")
        self._log(f"Bắt đầu… kết quả lưu vào {out}")
        threading.Thread(target=self._batch_worker, args=(cfg, out, self.var_k.get()), daemon=True).start()

    def _batch_worker(self, cfg, out, with_k):
        from .cli import run_pipeline
        try:
            df, lines, figs = run_pipeline(cfg, out, lambda i, n, t: self.q.put(("prog", i, n, t)), with_k)
            self.q.put(("done", len(df), lines, len(figs), out))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "prog":
                    _, i, n, t = msg
                    self.pb["maximum"] = max(n, 1)
                    self.pb["value"] = i
                    self._log(f"[{i}/{n}] {t}")
                elif msg[0] == "done":
                    _, nrow, lines, nfig, out = msg
                    self.pb["value"] = self.pb["maximum"]
                    self._log("")
                    for ln in lines:
                        self._log("• " + ln)
                    self._log(f"\nXONG: {nrow} dòng kết quả, {nfig} hình. Thư mục: {out}")
                    self.btn_run.state(["!disabled"])
                elif msg[0] == "error":
                    self._log("LỖI: " + msg[1])
                    messagebox.showerror("Lỗi", msg[1])
                    self.btn_run.state(["!disabled"])
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def open_out(self):
        d = self.batch_vars["out"].get()
        if not os.path.isdir(d):
            messagebox.showinfo("Chưa có", "Thư mục kết quả chưa tồn tại – hãy chạy thí nghiệm trước.")
            return
        if sys.platform.startswith("win"):
            os.startfile(d)
        else:
            subprocess.Popen(["xdg-open", d])


HELP_TEXT = """ImgTX – Khảo sát các phép biến đổi cường độ: âm bản, logarit và gamma

CÔNG THỨC
  • Âm bản:   s = 255 − r
  • Logarit:  s = gain · (255 / ln 256) · ln(1 + r)      (gain = 1 ⇒ T(255) = 255; gain > 1 gây bão hoà)
  • Gamma:    s = 255 · c · (r / 255)^γ                    (γ < 1 làm sáng, γ > 1 làm tối; c > 1 gây bão hoà)
  • Quan hệ gradient: |∇s| = |T′(f)| · |∇f|  ⇒ âm bản giữ nguyên biên (H1).

CÁCH DÙNG
  1. Mở ảnh hoặc chọn ảnh mẫu. Chọn ảnh xám hoặc màu (biến đổi kênh V của HSV, giữ sắc độ).
  2. Ảnh tham chiếu: chọn một kiểu suy giảm để coi ảnh đang mở là ẢNH GỐC, ảnh đầu vào là ảnh đã suy giảm.
     Khi đó F1/IoU/tương quan gradient/MSE/PSNR đo mức bảo toàn so với ảnh gốc.
     Chọn "Không – ảnh thật" khi không có ảnh gốc: chỉ số biên chỉ đo độ giống ảnh đầu vào (hạn chế, mục 8.2).
  3. Chọn phép biến đổi và kéo thanh trượt. Bảng dưới ảnh so sánh với dòng identity (mốc).
  4. Tab "Quét tham số": TN2 (γ), TN3 (gain log), TN7 (heatmap γ × c) trên ảnh hiện tại.
  5. Tab "Thí nghiệm hàng loạt": chạy toàn bộ TN1–TN8 trên thư mục ảnh, xuất
       tables/results.csv, tables/tham_so_de_xuat.csv, tables/tn6_do_nhay_k.csv,
       figures/*.png, hypotheses.md (số liệu H1–H5) và run_info.json (cấu hình + phiên bản thư viện).

CHỈ SỐ
  • F1/IoU top-k: trùng khớp vị trí biên (cùng mật độ k nên P = R = F1).
  • Tương quan gradient: Pearson giữa |∇ tham chiếu| và |∇ kết quả| – không cần ngưỡng.
  • MSE/PSNR: chỉ có khi có ảnh gốc; không tính cho âm bản (đảo mức xám là chủ ý).

Lưu ý: suy giảm "Đối chứng gamma 2.5" chỉ để minh hoạ thiên vị thiết kế (TN8), không dùng để kết luận H3–H4.
"""


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
