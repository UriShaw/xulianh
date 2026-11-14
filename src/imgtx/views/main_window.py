"""VIEW – cửa sổ chính ImgTX (Tkinter + Matplotlib).

Chỉ dựng giao diện, đọc giá trị người dùng nhập và vẽ kết quả. Không tính toán:
mọi sự kiện được chuyển cho controller qua `bind_events(handlers)`.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .. import APP_NAME, __version__
from ..models.degrade import DEFAULT_PARAMS
from ..models.io_utils import resource_path
from ..models.session import DEGRADE_HINTS, DEGRADE_KEYS, DEGRADE_TITLES, TRANSFORM_LABELS, Settings
from ..models.transforms import TRANSFORMS
from . import charts
from .help_text import HELP_TEXT
from .widgets import ACCENT, PANEL, Slider, apply_theme, fmt_number

IMAGE_TYPES = [("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Tất cả", "*.*")]


class MainWindow(tk.Tk):
    TAB_RESULT, TAB_OVERVIEW, TAB_HIST, TAB_SWEEP, TAB_BATCH, TAB_HELP = range(6)

    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}  v{__version__}")
        self.geometry("1360x860")
        self.minsize(1100, 700)
        self.configure(bg="#f4f6fa")
        self._set_icon()
        apply_theme(self)
        self.handlers = {}

        self._build_header()
        body = ttk.Frame(self, style="Bg.TFrame")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._build_controls(body)
        self._build_tabs(body)
        self.status_bar = ttk.Label(self, text="Sẵn sàng.", style="Status.TLabel", anchor="w")
        self.status_bar.pack(fill="x", side="bottom")

    # ================================================================== kết nối controller
    def bind_events(self, handlers):
        """handlers: open, sample, prepare, update, sweep(kind), save_output, save_metrics,
        save_sweep, run_batch, open_out."""
        self.handlers = handlers

    def _emit(self, name, *args):
        fn = self.handlers.get(name)
        if fn:
            fn(*args)

    # ================================================================== dựng giao diện
    def _set_icon(self):
        try:
            if sys.platform.startswith("win"):
                self.iconbitmap(default=resource_path("assets", "imgtx.ico"))
            self._icon_img = tk.PhotoImage(file=resource_path("assets", "logo.png")).subsample(8, 8)
            self.iconphoto(True, self._icon_img)
        except Exception:
            pass

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
        p = ttk.Frame(body, style="Panel.TFrame", padding=12)
        p.pack(side="left", fill="y", padx=(0, 10))
        prep = lambda: self._emit("prepare")
        upd = lambda: self._emit("update")

        self._section(p, "1. Nguồn ảnh")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Mở ảnh…", command=lambda: self._emit("open")).pack(side="left")
        self.cb_sample = ttk.Combobox(row, state="readonly", width=12)
        self.cb_sample.pack(side="left", padx=(6, 0))
        self.cb_sample.bind("<<ComboboxSelected>>", lambda e: self._emit("sample", self.cb_sample.get()))
        self.lbl_file = ttk.Label(p, text="", style="Hint.TLabel", wraplength=250)
        self.lbl_file.pack(anchor="w", pady=(2, 0))

        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Cạnh dài tối đa", style="Panel.TLabel").pack(side="left")
        self.cb_size = ttk.Combobox(row, state="readonly", width=6, values=["256", "512", "768", "1024"])
        self.cb_size.set("512")
        self.cb_size.pack(side="right")
        self.cb_size.bind("<<ComboboxSelected>>", lambda e: prep())
        self.mode = tk.StringVar(value="gray")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Radiobutton(row, text="Ảnh xám", value="gray", variable=self.mode, style="Panel.TRadiobutton",
                        command=prep).pack(side="left")
        ttk.Radiobutton(row, text="Màu (kênh V)", value="v", variable=self.mode, style="Panel.TRadiobutton",
                        command=prep).pack(side="left", padx=(10, 0))

        self._section(p, "2. Ảnh tham chiếu (mục 8.2)")
        self.cb_deg = ttk.Combobox(p, state="readonly", width=32, values=DEGRADE_TITLES)
        self.cb_deg.current(1)
        self.cb_deg.pack(fill="x")
        self.cb_deg.bind("<<ComboboxSelected>>", lambda e: self._degrade_changed())
        self.deg_box = ttk.Frame(p, style="Panel.TFrame")
        self.deg_box.pack(fill="x")
        self.s_alpha = Slider(self.deg_box, "α", 0.1, 2.5, 0.4, 0.05, prep)
        self.s_beta = Slider(self.deg_box, "β", 0, 100, 10, 1, prep, "{:.0f}")
        self.s_sigma = Slider(self.deg_box, "σ nhiễu", 0, 15, 4, 0.5, prep, "{:.1f}")
        self.s_dg = Slider(self.deg_box, "γ làm tối (đối chứng)", 1.2, 4.0, 2.5, 0.1, prep, "{:.1f}")
        self.hint_deg = ttk.Label(p, text="", style="Hint.TLabel", wraplength=250)
        self.hint_deg.pack(anchor="w")

        self._section(p, "3. Phép biến đổi")
        self.tf = tk.StringVar(value="gamma")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x")
        for k, v in TRANSFORM_LABELS.items():
            ttk.Radiobutton(row, text=v, value=k, variable=self.tf, style="Panel.TRadiobutton",
                            command=self._transform_changed).pack(side="left", padx=(0, 8))
        self.param_box = ttk.Frame(p, style="Panel.TFrame")
        self.param_box.pack(fill="x")
        self.s_gain = Slider(self.param_box, "gain (hệ số c của log)", 0.2, 2.0, 1.0, 0.05, upd)
        self.s_gamma = Slider(self.param_box, "γ", 0.1, 4.0, 0.5, 0.05, upd)
        self.s_c = Slider(self.param_box, "c (hệ số gamma)", 0.25, 3.0, 1.0, 0.05, upd)
        self.s_k = Slider(p, "Mật độ biên k (top-k)", 0.02, 0.30, 0.10, 0.01, upd)
        self.s_k.pack(fill="x", pady=(6, 0))

        self._section(p, "4. Xuất kết quả")
        row = ttk.Frame(p, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Lưu ảnh kết quả", command=lambda: self._emit("save_output")).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Xuất chỉ số", command=lambda: self._emit("save_metrics")).pack(
            side="left", fill="x", expand=True, padx=(6, 0))
        self._show_degrade_sliders()
        self._show_param_sliders()

    def _build_tabs(self, body):
        nb = ttk.Notebook(body)
        nb.pack(side="left", fill="both", expand=True)
        self.nb = nb

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

        fo = ttk.Frame(nb, style="Panel.TFrame", padding=6)
        nb.add(fo, text="  Tổng quan đề 5  ")
        bar = ttk.Frame(fo, style="Panel.TFrame")
        bar.pack(fill="x")
        ttk.Button(bar, text="Vẽ tổng quan (âm bản · log · γ 0.5/1/2)", style="Accent.TButton",
                   command=lambda: self._emit("overview")).pack(side="left")
        ttk.Button(bar, text="Kiểm chứng Sobel & Otsu tự cài với OpenCV",
                   command=lambda: self._emit("verify")).pack(side="left", padx=6)
        self.lbl_overview = ttk.Label(fo, text="Một màn hình cho cả đề: ảnh – histogram – biên Sobel của ba phép biến "
                                               "đổi; F1 so với ảnh tham chiếu.", style="Hint.TLabel")
        self.lbl_overview.pack(anchor="w", pady=4)
        self.fig_over = Figure(figsize=(10, 6), dpi=100, layout="tight", facecolor="white")
        self.cv_over = FigureCanvasTkAgg(self.fig_over, master=fo)
        self.cv_over.get_tk_widget().pack(fill="both", expand=True)
        nb.bind("<<NotebookTabChanged>>", lambda e: self._emit("tab", nb.index("current")))

        f2 = ttk.Frame(nb, style="Panel.TFrame", padding=6)
        nb.add(f2, text="  Histogram & đường cong T(r)  ")
        self.fig_hist = Figure(figsize=(9, 4), dpi=100, layout="tight", facecolor="white")
        self.cv_hist = FigureCanvasTkAgg(self.fig_hist, master=f2)
        self.cv_hist.get_tk_widget().pack(fill="both", expand=True)

        f3 = ttk.Frame(nb, style="Panel.TFrame", padding=6)
        nb.add(f3, text="  Quét tham số (TN2/TN3/TN7)  ")
        bar = ttk.Frame(f3, style="Panel.TFrame")
        bar.pack(fill="x")
        ttk.Button(bar, text="Quét γ (c hiện tại) – TN2", command=lambda: self._emit("sweep", "gamma")).pack(side="left")
        ttk.Button(bar, text="Quét gain log – TN3", command=lambda: self._emit("sweep", "log")).pack(side="left", padx=6)
        ttk.Button(bar, text="Heatmap γ × c – TN7", command=lambda: self._emit("sweep", "grid")).pack(side="left")
        ttk.Button(bar, text="Xuất CSV lần quét", command=lambda: self._emit("save_sweep")).pack(side="right")
        self.lbl_sweep = ttk.Label(f3, text="Bấm một nút để quét trên ảnh hiện tại. Đường chấm = dòng identity (mốc).",
                                   style="Hint.TLabel")
        self.lbl_sweep.pack(anchor="w", pady=4)
        self.fig_sweep = Figure(figsize=(9, 4), dpi=100, layout="tight", facecolor="white")
        self.cv_sweep = FigureCanvasTkAgg(self.fig_sweep, master=f3)
        NavigationToolbar2Tk(self.cv_sweep, f3, pack_toolbar=True).update()
        self.cv_sweep.get_tk_widget().pack(fill="both", expand=True)

        self._build_batch_tab(nb)

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
        self.btn_run = ttk.Button(btns, text="Chạy thí nghiệm", style="Accent.TButton",
                                  command=lambda: self._emit("run_batch"))
        self.btn_run.pack(side="left")
        ttk.Button(btns, text="Mở thư mục kết quả", command=lambda: self._emit("open_out")).pack(side="left", padx=6)
        self.pb = ttk.Progressbar(f, mode="determinate", style="Horizontal.TProgressbar")
        self.pb.pack(fill="x")
        self.log = tk.Text(f, height=16, wrap="word", relief="flat", bg="#f7f8fb", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, pady=(8, 0))

    # ================================================================== hành vi thuần giao diện
    def _degrade_key(self):
        return DEGRADE_KEYS[self.cb_deg.current()]

    def _show_degrade_sliders(self):
        key = self._degrade_key()
        for w in (self.s_alpha, self.s_beta, self.s_sigma, self.s_dg):
            w.pack_forget()
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
        self.hint_deg.configure(text=DEGRADE_HINTS[key])

    def _degrade_changed(self):
        self._show_degrade_sliders()
        self._emit("prepare")

    def _show_param_sliders(self):
        for w in (self.s_gain, self.s_gamma, self.s_c):
            w.pack_forget()
        t = self.tf.get()
        if t == "log":
            self.s_gain.pack(fill="x")
        elif t == "gamma":
            self.s_gamma.pack(fill="x")
            self.s_c.pack(fill="x")

    def _transform_changed(self):
        self._show_param_sliders()
        self._emit("update")

    def _pick_dir(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    # ================================================================== đọc dữ liệu nhập
    def read_settings(self) -> Settings:
        key = self._degrade_key()
        if key in ("offset", "bright"):
            dp = dict(alpha=self.s_alpha.get(), beta=self.s_beta.get())
        elif key == "noisy":
            dp = dict(alpha=self.s_alpha.get(), beta=self.s_beta.get(), sigma=self.s_sigma.get(), seed=0)
        elif key == "control_gamma":
            dp = dict(gamma=self.s_dg.get())
        else:
            dp = {}
        t = self.tf.get()
        params = {"gain": self.s_gain.get()} if t == "log" else \
            {"gamma": self.s_gamma.get(), "c": self.s_c.get()} if t == "gamma" else {}
        return Settings(int(self.cb_size.get()), self.mode.get(), key, dp, t, params, self.s_k.get())

    def batch_inputs(self):
        return dict(gt=self.batch_vars["gt"].get().strip(), real=self.batch_vars["real"].get().strip(),
                    out=self.batch_vars["out"].get().strip(), synthetic=self.var_syn.get(),
                    with_k=self.var_k.get())

    def ask_open_image(self):
        return filedialog.askopenfilename(title="Chọn ảnh", filetypes=IMAGE_TYPES)

    def ask_save(self, default_name, ext, label):
        return filedialog.asksaveasfilename(defaultextension=ext, initialfile=default_name,
                                            filetypes=[(label, "*" + ext)])

    # ================================================================== hiển thị
    def set_samples(self, names, current=None):
        self.cb_sample["values"] = names
        if current:
            self.cb_sample.set(current)

    def show_file(self, name, w, h):
        self.lbl_file.configure(text=f"{name}  ({w}×{h})")

    def status(self, text):
        self.status_bar.configure(text=text)

    def error(self, title, msg):
        messagebox.showerror(title, msg)

    def info(self, title, msg):
        messagebox.showinfo(title, msg)

    def busy(self, on):
        self.config(cursor="watch" if on else "")
        self.update_idletasks()

    def show_result(self, r):
        """Vẽ tab Ảnh & biên, tab Histogram và bảng chỉ số từ một model.session.Result."""
        s = r.settings
        self._draw_images(r)
        self._draw_hist(r)
        self.tv.delete(*self.tv.get_children())
        for label, b, o, d in r.metric_rows():
            self.tv.insert("", "end", values=(label, fmt_number(b), fmt_number(o),
                                              "—" if d is None else ("+" if d > 0 else "") + fmt_number(d)))
        desc = TRANSFORM_LABELS[s.transform]
        if s.params:
            desc += " (" + ", ".join(f"{a}={b:g}" for a, b in s.params.items()) + ")"
        self.status(f"{desc}  ·  tham chiếu: {'ảnh gốc' if r.has_gt else 'ảnh đầu vào'}"
                    f"  ·  kích thước {r.ref_in.shape[1]}×{r.ref_in.shape[0]}")

    @staticmethod
    def _imshow(ax, img, title, cmap="gray"):
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        if img.ndim == 3:
            ax.imshow(img)
        elif cmap == "gray":
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def _draw_images(self, r):
        f = self.fig_img
        f.clear()
        ax = f.subplots(2, 3)
        k = r.settings.density
        self._imshow(ax[0, 0], r.shown_ref, "Ảnh gốc (tham chiếu)" if r.has_gt else "Ảnh đầu vào (tham chiếu)")
        if r.has_gt:
            self._imshow(ax[0, 1], r.shown_in, "Ảnh đầu vào (đã suy giảm)")
        else:
            self._imshow(ax[0, 1], r.mag_ref, "|∇| ảnh đầu vào", cmap="magma")
        self._imshow(ax[0, 2], r.shown_out, "Kết quả – " + TRANSFORM_LABELS[r.settings.transform])
        self._imshow(ax[1, 0], r.edge_ref, f"Biên tham chiếu (top {k:.0%})")
        self._imshow(ax[1, 1], r.edge_out, f"Biên kết quả – F1 = {r.row['f1_topk']:.3f}")
        self._imshow(ax[1, 2], r.mag_out, f"|∇| kết quả – tương quan = {r.row['grad_corr']:.3f}", cmap="magma")
        self.cv_img.draw_idle()

    def _draw_hist(self, r):
        f = self.fig_hist
        f.clear()
        ax1, ax2 = f.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1]})
        imgs = ({"Ảnh gốc": r.ref_goc} if r.has_gt else {}) | {"Ảnh đầu vào": r.ref_in, "Kết quả": r.out}
        colors = [charts.GRAY, charts.ORANGE, charts.BLUE][-len(imgs):]
        for (name, img), col in zip(imgs.items(), colors):
            h = np.bincount(img.ravel(), minlength=256) / img.size
            ax1.fill_between(np.arange(256), h, step="mid", alpha=0.25, color=col)
            ax1.plot(np.arange(256), h, lw=1.1, color=col, label=f"{name}  μ={img.mean():.1f}  σ={img.std():.1f}")
        ax1.set(xlim=(0, 255), xlabel="mức xám r", ylabel="p(r)", title="Histogram chuẩn hoá")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.25)
        rr = np.arange(256, dtype=np.uint8)
        t = r.settings.transform
        ax2.plot(rr, rr, ":", color=charts.GRAY, label="identity")
        ax2.plot(rr, TRANSFORMS[t](rr, **r.settings.params), color=charts.BLUE, lw=2, label=TRANSFORM_LABELS[t])
        ax2.set(xlim=(0, 255), ylim=(0, 260), xlabel="r", ylabel="s = T(r)", title="Đường cong biến đổi")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.25)
        self.cv_hist.draw_idle()

    def show_sweep(self, kind, df, base, settings, best=None):
        has_gt = base.get("ref_type") == "goc"
        if kind == "gamma":
            panels = [[("mean", "mean")], [("entropy", "entropy")], [("f1_topk", "F1"), ("grad_corr", "grad_corr")]]
            if has_gt:
                panels.append([("psnr", "PSNR (dB)")])
            charts.plot_sweep(df, "gamma", panels, base, settings.params.get("gamma"), fig=self.fig_sweep)
            msg = f"TN2 – quét γ với c = {settings.params.get('c', 1.0):g}."
        elif kind == "log":
            panels = [[("sat_high", "tỉ lệ bão hoà 255")], [("dyn_range", "dải động")],
                      [("f1_topk", "F1"), ("grad_corr", "grad_corr")]]
            if has_gt:
                panels.append([("psnr", "PSNR (dB)")])
            charts.plot_sweep(df, "gain", panels, base, settings.params.get("gain"), fig=self.fig_sweep)
            msg = "TN3 – quét gain của log."
        else:
            vals = [("f1_topk", "F1 biên"), ("grad_corr", "Tương quan ∇"), ("mean", "Mean")]
            if has_gt:
                vals.insert(2, ("psnr", "PSNR (dB)"))
            charts.plot_heatmaps(df, vals, fig=self.fig_sweep)
            msg = "TN7 – heatmap γ × c."
            if best is not None:
                msg += (f"  PSNR cao nhất: γ={best.gamma:g}, c={best.c:g} → {best.psnr:.1f} dB, F1 {best.f1_topk:.3f} "
                        f"(identity: {base['psnr']:.1f} dB, F1 {base['f1_topk']:.3f}).")
        self.lbl_sweep.configure(text=msg + "  Đường chấm = identity, nét đứt = tham số đang chọn.")
        self.cv_sweep.draw_idle()
        self.nb.select(self.TAB_SWEEP)

    def show_overview(self, items, has_gt, density):
        """Lưới 3 hàng × 6 cột: ảnh / histogram / biên Sobel top-k."""
        f = self.fig_over
        f.clear()
        n = len(items)
        ax = f.subplots(3, n)
        for j, (title, img, row, edge) in enumerate(items):
            self._imshow(ax[0, j], img, title)
            h = np.bincount(img.ravel(), minlength=256) / img.size
            ax[1, j].fill_between(np.arange(256), h, step="mid", color=charts.BLUE, alpha=0.85)
            ax[1, j].axvline(row["mean"], color=charts.RED, lw=1, ls="--")
            ax[1, j].set_xlim(0, 255)
            ax[1, j].set_yticks([])
            ax[1, j].tick_params(labelsize=7)
            ax[1, j].set_title(f"μ={row['mean']:.0f}  σ={row['std']:.0f}  H={row['entropy']:.2f}", fontsize=8)
            ps = f"  PSNR {row['psnr']:.1f}" if has_gt and np.isfinite(row["psnr"]) else ""
            self._imshow(ax[2, j], edge, f"F1 {row['f1_topk']:.3f}{ps}")
        f.text(0.002, 0.83, "Ảnh", rotation=90, fontsize=9, va="center")
        f.text(0.002, 0.5, "Histogram", rotation=90, fontsize=9, va="center")
        f.text(0.002, 0.17, f"Biên Sobel top {density:.0%}", rotation=90, fontsize=9, va="center")
        self.cv_over.draw_idle()
        self.lbl_overview.configure(text="F1/PSNR so với " + ("ảnh gốc" if has_gt else "ảnh đầu vào") +
                                    ". Âm bản có F1 bằng đúng ảnh đầu vào: |∇(255−f)| = |∇f| (giả thuyết H1).")

    def show_verify(self, v):
        self.lbl_overview.configure(
            text=f"Sobel tự cài vs OpenCV trên ảnh {v['shape'][1]}×{v['shape'][0]}: sai khác lớn nhất = "
                 f"{v['max_diff']:.2e} · thời gian {v['ms_manual']:.1f} ms (tự cài) / {v['ms_opencv']:.1f} ms (OpenCV)"
                 f"  |  Ngưỡng Otsu tự cài = {v['otsu_manual']}, OpenCV = {v['otsu_opencv']}")

    # ---------------------------------------------------------------- hàng loạt
    def batch_started(self, out):
        self.btn_run.state(["disabled"])
        self.log.delete("1.0", "end")
        self.pb["value"] = 0
        self.batch_log(f"Bắt đầu… kết quả lưu vào {out}")

    def batch_log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def batch_progress(self, i, n, text):
        self.pb["maximum"] = max(n, 1)
        self.pb["value"] = i
        self.batch_log(f"[{i}/{n}] {text}")

    def batch_done(self, nrow, lines, nfig, out):
        self.pb["value"] = self.pb["maximum"]
        self.batch_log("")
        for ln in lines:
            self.batch_log("• " + ln)
        self.batch_log(f"\nXONG: {nrow} dòng kết quả, {nfig} hình. Thư mục: {out}")
        self.btn_run.state(["!disabled"])

    def batch_failed(self, msg):
        self.batch_log("LỖI: " + msg)
        self.btn_run.state(["!disabled"])
        self.error("Lỗi", msg)
