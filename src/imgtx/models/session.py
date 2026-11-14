"""Model – phiên làm việc trên MỘT ảnh: giữ trạng thái và tính kết quả cho giao diện.

Không phụ thuộc Tkinter/Matplotlib: controller gọi các hàm ở đây rồi đưa kết quả cho view vẽ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ._lazy import pd

import time

from .analysis import edge_map_topk, otsu_threshold, sobel
from .degrade import DEFAULT_PARAMS, DEGRADE_LABELS, build_pair
from .experiment import CS, GAINS, GAMMAS, run_one, sweep
from .io_utils import read_rgb, resize_max, sample_images, save_csv, save_image, to_display, to_working

TRANSFORM_LABELS = {"negative": "Âm bản", "log": "Logarit", "gamma": "Gamma"}
DEGRADE_KEYS = ["none"] + list(DEGRADE_LABELS)
DEGRADE_TITLES = ["Không – ảnh thật (so với ảnh đầu vào)"] + list(DEGRADE_LABELS.values())
DEGRADE_HINTS = {
    "none": "Ảnh đang chọn là ảnh đầu vào; F1/IoU chỉ đo mức giống ảnh đầu vào, không có MSE/PSNR.",
    "offset": "Ảnh chọn = ảnh gốc; đầu vào = clip(α·r − β). Suy giảm ngoài họ gamma/log.",
    "noisy": "Như offset + nhiễu Gauss σ (seed 0).",
    "bright": "Quá sáng: clip(α·r + β), vùng sáng bị cắt 255.",
    "control_gamma": "CHỈ ĐỐI CHỨNG thiên vị (TN8): gamma 1/γ khôi phục gần hoàn hảo.",
}
METRICS = [("mean", "Mean (độ sáng)"), ("std", "Std (tương phản)"), ("entropy", "Entropy (bit)"),
           ("dyn_range", "Dải động P99−P1"), ("sat_high", "Tỉ lệ bão hoà 255"),
           ("f1_topk", "F1 biên (top-k)"), ("iou_topk", "IoU biên"), ("grad_corr", "Tương quan gradient"),
           ("mean_grad_norm", "Gradient TB chuẩn hoá"), ("edge_density_otsu", "Mật độ biên Otsu"),
           ("mse", "MSE"), ("psnr", "PSNR (dB)")]


@dataclass
class Settings:
    """Mọi lựa chọn của người dùng cho một lần tính (view đọc từ điều khiển, controller chuyển cho model)."""
    max_side: int = 512
    mode: str = "gray"                 # "gray" | "v" (kênh V của HSV)
    degrade: str = "offset"            # khoá trong DEGRADE_KEYS
    degrade_params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS["offset"]))
    transform: str = "gamma"           # negative | log | gamma
    params: dict = field(default_factory=lambda: {"gamma": 0.5, "c": 1.0})
    density: float = 0.10              # mật độ biên top-k


@dataclass
class Result:
    """Kết quả một lần tính – đủ để view vẽ mà không phải tính lại."""
    ref_goc: np.ndarray | None
    ref_in: np.ndarray
    out: np.ndarray
    row: dict
    base_row: dict
    shown_ref: np.ndarray
    shown_in: np.ndarray
    shown_out: np.ndarray
    mag_ref: np.ndarray
    mag_out: np.ndarray
    edge_ref: np.ndarray
    edge_out: np.ndarray
    settings: Settings

    @property
    def has_gt(self):
        return self.ref_goc is not None

    def metric_rows(self):
        """[(nhãn, identity, kết quả, chênh lệch)] cho bảng chỉ số."""
        rows = []
        for key, label in METRICS:
            b, o = self.base_row.get(key), self.row.get(key)
            d = o - b if _finite(o) and _finite(b) else None
            rows.append((label, b, o, d))
        return rows


def _finite(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)


class ImageSession:
    """Trạng thái: ảnh đang mở → cặp (ảnh gốc, ảnh đầu vào) → kết quả biến đổi → lần quét gần nhất."""

    def __init__(self):
        self.path = None
        self.rgb = None
        self.small = None
        self.ref_goc = None
        self.ref_in = None
        self.result: Result | None = None
        self.sweep_df: pd.DataFrame | None = None
        self._prepared_key = None

    # ---------------------------------------------------------------- nguồn ảnh
    @staticmethod
    def samples():
        return sample_images()

    def load(self, path):
        """Đọc ảnh; ném ValueError nếu không đọc được."""
        self.rgb = read_rgb(path)
        self.path = path
        self._prepared_key = None
        return self.rgb.shape[1], self.rgb.shape[0]

    @property
    def loaded(self):
        return self.rgb is not None

    # ---------------------------------------------------------------- tính toán
    def prepare(self, s: Settings):
        """Resize, chuyển xám/kênh V, tạo cặp (ảnh gốc, ảnh đầu vào) – chỉ làm lại khi cần."""
        key = (s.max_side, s.mode, s.degrade, tuple(sorted(s.degrade_params.items())), id(self.rgb))
        if key == self._prepared_key:
            return
        self.small = resize_max(self.rgb, s.max_side)
        work = to_working(self.small, s.mode)
        self.ref_goc, self.ref_in = build_pair(work, s.degrade, **s.degrade_params)
        self.sweep_df = None
        self._prepared_key = key

    def compute(self, s: Settings) -> Result:
        if not self.loaded:
            raise ValueError("Chưa mở ảnh.")
        self.prepare(s)
        out, row = run_one(s.transform, self.ref_goc, self.ref_in, s.density, **s.params)
        _, base = run_one("identity", self.ref_goc, self.ref_in, s.density)
        ref = self.ref_goc if self.ref_goc is not None else self.ref_in
        mag_ref, mag_out = sobel(ref)[0], sobel(out)[0]
        show = (lambda w: to_display(self.small, w, "v")) if s.mode == "v" else (lambda w: w)
        self.result = Result(self.ref_goc, self.ref_in, out, row, base, show(ref), show(self.ref_in), show(out),
                             mag_ref, mag_out, edge_map_topk(mag_ref, s.density),
                             edge_map_topk(mag_out, s.density), s)
        return self.result

    def run_sweep(self, kind, s: Settings):
        """kind: 'gamma' (TN2, c hiện tại) | 'log' (TN3) | 'grid' (TN7). Trả (DataFrame, dòng identity)."""
        self.prepare(s)
        base = run_one("identity", self.ref_goc, self.ref_in, s.density)[1]
        if kind == "gamma":
            grid, name = {"gamma": GAMMAS, "c": [s.params.get("c", 1.0)]}, "gamma"
        elif kind == "log":
            grid, name = {"gain": GAINS + [1.75, 2.0]}, "log"
        elif kind == "grid":
            grid, name = {"gamma": GAMMAS, "c": CS}, "gamma"
        else:
            raise ValueError(kind)
        self.sweep_df = sweep(self.ref_goc, self.ref_in, name, grid, s.density)
        return self.sweep_df, base

    def best_in_sweep(self):
        """Tham số có PSNR cao nhất trong lần quét (None nếu không có ảnh gốc)."""
        df = self.sweep_df
        if df is None or self.ref_goc is None or df.psnr.isna().all():
            return None
        return df.loc[df.psnr.idxmax()]

    def overview(self, s: Settings):
        """Tổng quan trên một màn hình: đầu vào, âm bản, log, γ = 0.5 / 1 / 2 (slide 18, chương 2).

        Trả list (tên, ảnh, dòng chỉ số, bản đồ biên top-k)."""
        self.prepare(s)
        gain = s.params.get("gain", 1.0) if s.transform == "log" else 1.0
        cases = [("Ảnh đầu vào", "identity", {}), ("Âm bản", "negative", {}),
                 (f"Log (gain={gain:g})", "log", {"gain": gain}),
                 ("Gamma γ=0.5", "gamma", {"gamma": 0.5}), ("Gamma γ=1.0", "gamma", {"gamma": 1.0}),
                 ("Gamma γ=2.0", "gamma", {"gamma": 2.0})]
        items = []
        for title, name, p in cases:
            out, row = run_one(name, self.ref_goc, self.ref_in, s.density, **p)
            items.append((title, out, row, edge_map_topk(sobel(out)[0], s.density)))
        return items

    def verify_core(self):
        """Đối chiếu thuật toán TỰ CÀI với OpenCV trên ảnh hiện tại (Sobel + Otsu)."""
        import cv2
        img = self.result.out if self.result is not None else self.ref_in

        def timed(fn, n=10):
            fn()
            t = time.perf_counter()
            for _ in range(n):
                r = fn()
            return r, (time.perf_counter() - t) / n * 1000

        (m1, _), t_manual = timed(lambda: sobel(img, method="manual"))
        (m2, _), t_cv = timed(lambda: sobel(img, method="opencv"))
        g = np.rint(m1 / m1.max() * 255).astype(np.uint8) if m1.max() > 0 else np.zeros_like(img)
        t_own = otsu_threshold(g)
        t_ref, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return dict(max_diff=float(np.abs(m1 - m2).max()), ms_manual=t_manual, ms_opencv=t_cv,
                    otsu_manual=t_own, otsu_opencv=int(t_ref), shape=img.shape)

    # ---------------------------------------------------------------- xuất
    def export_output(self, path):
        """Lưu ảnh kết quả và bản đồ biên (<tên>_bien.<đuôi>)."""
        r = self.result
        save_image(path, r.shown_out)
        stem, ext = path.rsplit(".", 1) if "." in path else (path, "png")
        save_image(f"{stem}_bien.{ext}", r.edge_out.astype(np.uint8) * 255)

    def export_metrics(self, path):
        r = self.result
        save_csv(pd.DataFrame([dict(dong="identity", **r.base_row), dict(dong="ket_qua", **r.row)]), path)

    def export_sweep(self, path):
        if self.sweep_df is None:
            raise ValueError("Chưa có lần quét nào.")
        save_csv(self.sweep_df, path)
