"""VIEW – màu, kiểu chữ (ttk theme) và widget dùng lại: Slider có nhãn."""
import tkinter as tk
from tkinter import ttk

ACCENT = "#1f6fb2"
BG = "#f4f6fa"
PANEL = "#ffffff"
TEXT = "#1d2433"
MUTED = "#6b7487"
FONT = ("Segoe UI", 10)


def apply_theme(root):
    s = ttk.Style(root)
    try:
        s.theme_use("clam")
    except tk.TclError:
        pass
    root.option_add("*Font", FONT)
    s.configure(".", font=FONT, background=BG, foreground=TEXT)
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


class Slider(ttk.Frame):
    """Thanh trượt có nhãn + giá trị, làm tròn theo bước `step`; gọi `command()` khi kéo."""

    def __init__(self, master, text, frm, to, value, step, command=None, fmt="{:.2f}"):
        super().__init__(master, style="Panel.TFrame")
        self.step, self.fmt, self.command = step, fmt, command
        self.var = tk.DoubleVar(value=value)
        top = ttk.Frame(self, style="Panel.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=text, style="Panel.TLabel").pack(side="left")
        self.lbl = ttk.Label(top, text=fmt.format(value), style="Value.TLabel")
        self.lbl.pack(side="right")
        ttk.Scale(self, from_=frm, to=to, variable=self.var, command=self._moved).pack(fill="x")

    def _moved(self, _=None):
        self.lbl.configure(text=self.fmt.format(self.get()))
        if self.command:
            self.command()

    def get(self):
        return round(round(self.var.get() / self.step) * self.step, 6)

    def set(self, v):
        self.var.set(v)
        self.lbl.configure(text=self.fmt.format(v))


def fmt_number(v):
    """Định dạng số cho bảng chỉ số ('—' khi rỗng/NaN)."""
    import numpy as np
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
