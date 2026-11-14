"""Điều phối thí nghiệm: đọc cấu hình, lặp ảnh × suy giảm × phép × tham số, gom kết quả (mục 6, 7, 8)."""
import itertools
import os
import platform

import numpy as np
import pandas as pd

from . import __version__
from .analysis import evaluate, sobel
from .degrade import DEFAULT_PARAMS, build_pair
from .io_utils import list_images, read_rgb, resize_max, to_gray
from .transforms import TRANSFORMS, apply_lut

NAN = float("nan")

DEFAULT_CONFIG = {
    "images": [
        {"path": "data/raw/*", "group": "chuan", "has_gt": True},
        {"path": "data/synthetic/*", "group": "tong_hop", "has_gt": True},
        {"path": "data/real/*", "group": "anh_that", "has_gt": False},
    ],
    "resize": 512,
    "degrade": {k: dict(v) for k, v in DEFAULT_PARAMS.items()},
    "edge": {"density": 0.10, "ksize": 3, "density_sweep": [0.05, 0.10, 0.15, 0.20]},
    "transforms": {
        "identity": {},
        "negative": {},
        "log": {"gain": [0.5, 0.75, 1.0, 1.25, 1.5]},
        "gamma": {"gamma": [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
                  "c": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]},
    },
    "select": {"mean_range": [90, 160], "f1_margin": 0.10, "weights": [0.4, 0.4, 0.2]},
    "output": "outputs",
}

GAMMAS = DEFAULT_CONFIG["transforms"]["gamma"]["gamma"]
CS = DEFAULT_CONFIG["transforms"]["gamma"]["c"]
GAINS = DEFAULT_CONFIG["transforms"]["log"]["gain"]


# ----------------------------------------------------------------- cấu hình
def load_config(path=None):
    """Đọc YAML và trộn với cấu hình mặc định."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    if path:
        import yaml
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        cfg["_root"] = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return cfg


def _image_specs(cfg):
    """Chuẩn hoá mục images: chấp nhận chuỗi glob hoặc dict {path, group, has_gt}."""
    root = cfg.get("_root", ".")
    specs = []
    for item in cfg["images"]:
        if isinstance(item, str):
            item = {"path": item}
        group = item.get("group") or os.path.basename(os.path.dirname(item["path"])) or "khac"
        for f in list_images(item["path"], root):
            specs.append(dict(path=f, group=group, has_gt=bool(item.get("has_gt", True))))
    return specs


def product_grid(grid):
    """{'gamma': [..], 'c': [..]} -> list các dict tham số (tích Descartes)."""
    if not grid:
        return [{}]
    keys = list(grid)
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


# ----------------------------------------------------------------- một lần chạy
def region_gain(ref_in, out, ksize=3):
    """Tỉ số |∇out| / |∇in| ở vùng tối và vùng sáng của ảnh đầu vào (dùng cho H2/H3)."""
    m_in, _ = sobel(ref_in, ksize)
    m_out, _ = sobel(out, ksize)
    med = np.median(ref_in)
    res = {}
    for name, mask in (("dark", ref_in <= med), ("bright", ref_in > med)):
        a = m_in[mask].mean() if mask.any() else 0.0
        res["grad_gain_" + name] = float(m_out[mask].mean() / a) if a > 0 else NAN
    return res


def run_one(name, ref_goc, ref_in, density=0.10, ksize=3, **params):
    """Biến đổi ảnh đầu vào bằng LUT rồi đánh giá. Trả (ảnh kết quả, dòng kết quả)."""
    out = apply_lut(ref_in, TRANSFORMS[name], **params)
    row = evaluate(ref_goc, ref_in, out, density, ksize)
    if name == "negative":          # đảo mức xám là chủ ý: MSE/PSNR vô nghĩa
        row["mse"] = row["psnr"] = NAN
    row.update(region_gain(ref_in, out, ksize))
    row.update(transform=name, gamma=params.get("gamma", NAN), c=params.get("c", NAN),
               gain=params.get("gain", NAN))
    return out, row


def sweep(ref_goc, ref_in, name, grid, density=0.10, ksize=3):
    """Quét một phép biến đổi trên lưới tham số -> DataFrame (dùng cho giao diện)."""
    return pd.DataFrame([run_one(name, ref_goc, ref_in, density, ksize, **p)[1]
                         for p in product_grid(grid)])


def run_grid(cfg, progress=None, images=None):
    """Lặp ảnh × suy giảm × phép × tham số (kể cả dòng identity làm mốc), gọi evaluate.

    progress: hàm callback(i, n, text) để giao diện hiển thị tiến độ.
    images: danh sách spec ghi đè (dict path/group/has_gt).
    """
    specs = images if images is not None else _image_specs(cfg)
    edge = cfg["edge"]
    jobs = []
    for s in specs:
        degs = list(cfg["degrade"].items()) if s["has_gt"] else [("none", {})]
        for deg, dp in degs:
            jobs.append((s, deg, dp))
    rows = []
    for i, (s, deg, dp) in enumerate(jobs):
        name = os.path.splitext(os.path.basename(s["path"]))[0]
        if progress:
            progress(i, len(jobs), f"{name} – {deg}")
        img = to_gray(resize_max(read_rgb(s["path"]), cfg.get("resize")))
        ref_goc, ref_in = build_pair(img, deg, **(dp or {}))
        for tname, grid in cfg["transforms"].items():
            for p in product_grid(grid):
                _, row = run_one(tname, ref_goc, ref_in, edge["density"], edge["ksize"], **p)
                row.update(image=name, group=s["group"], degrade=deg, has_gt=s["has_gt"])
                rows.append(row)
    if progress:
        progress(len(jobs), len(jobs), "Xong")
    df = pd.DataFrame(rows)
    first = ["image", "group", "degrade", "has_gt", "transform", "gamma", "c", "gain"]
    return df[first + [c for c in df.columns if c not in first]] if len(df) else df


def k_sensitivity(cfg, images=None, densities=None, degrade="offset"):
    """TN6: độ nhạy theo mật độ biên k (và Otsu) cho các phép mặc định."""
    specs = images if images is not None else [s for s in _image_specs(cfg) if s["has_gt"]]
    densities = densities or cfg["edge"]["density_sweep"]
    rows = []
    defaults = [("identity", {}), ("negative", {}), ("log", {"gain": 1.0}),
                ("gamma", {"gamma": 0.5, "c": 1.0}), ("gamma", {"gamma": 0.7, "c": 2.0})]
    for s in specs:
        img = to_gray(resize_max(read_rgb(s["path"]), cfg.get("resize")))
        ref_goc, ref_in = build_pair(img, degrade, **cfg["degrade"].get(degrade, {}))
        for k in densities:
            for tname, p in defaults:
                _, row = run_one(tname, ref_goc, ref_in, k, cfg["edge"]["ksize"], **p)
                row.update(image=os.path.basename(s["path"]), density=k)
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- chọn tham số (mục 7.3)
def add_score(df, w=(0.4, 0.4, 0.2)):
    df = df.copy()
    df["score"] = w[0] * df["f1_topk"] + w[1] * df["grad_corr"].fillna(0) + w[2] * df["entropy"] / 8.0
    return df


def select_params(df, cfg=None):
    """Mỗi (ảnh, suy giảm): chọn tham số log/gamma vừa cải thiện hiển thị vừa giữ biên, và hơn identity."""
    sel = (cfg or DEFAULT_CONFIG)["select"]
    lo, hi = sel["mean_range"]
    df = add_score(df, sel["weights"])
    out = []
    for (img, deg), g in df.groupby(["image", "degrade"], sort=False):
        base = g[g["transform"] == "identity"].iloc[0]
        cand = g[g["transform"].isin(["log", "gamma"])]
        ok = cand[(cand["mean"].between(lo, hi))
                  & (cand["f1_topk"] >= base["f1_topk"] - sel["f1_margin"])]
        if base["ref_type"] == "goc":
            ok = ok[ok["psnr"] > base["psnr"]]
        elif deg == "none":
            ok = ok[ok["dyn_range"] >= base["dyn_range"]]
        if len(ok):
            best = ok.sort_values(["score", "psnr"], ascending=False).iloc[0]
            out.append(dict(image=img, degrade=deg, transform=best["transform"], gamma=best.gamma,
                            c=best.c, gain=best.gain, mean=best["mean"], psnr=best.psnr,
                            f1_topk=best.f1_topk, grad_corr=best.grad_corr, score=best.score,
                            identity_psnr=base.psnr, identity_f1=base.f1_topk))
        else:
            out.append(dict(image=img, degrade=deg, transform="(không có tham số đạt)",
                            identity_psnr=base.psnr, identity_f1=base.f1_topk))
    return pd.DataFrame(out)


# ----------------------------------------------------------------- kiểm chứng giả thuyết
def summarize_hypotheses(df):
    """Tóm tắt số liệu cho H1–H5 (gợi ý kết luận; người viết báo cáo cần đọc lại bảng)."""
    lines = []
    gt = df[df.ref_type == "goc"]
    real = df[df.degrade.isin(["offset", "noisy", "bright"])]

    # H1
    d = []
    for _, g in df.groupby(["image", "degrade"]):
        i = g[g["transform"] == "identity"].iloc[0]
        n = g[g["transform"] == "negative"].iloc[0]
        d.append(max(abs(i.f1_topk - n.f1_topk), abs(i.grad_corr - n.grad_corr)))
    h1 = max(d) if d else NAN
    lines.append(f"H1 (âm bản giữ nguyên biên): chênh lệch lớn nhất F1/grad_corr giữa âm bản và identity = {h1:.2e} "
                 f"trên {len(d)} cặp ảnh -> {'ỦNG HỘ' if h1 < 1e-9 else 'KHÔNG ủng hộ'}.")

    # H2
    lg = real[(real["transform"] == "log") & (real.gain == 1.0)]
    if len(lg):
        dark, bright = lg.grad_gain_dark.mean(), lg.grad_gain_bright.mean()
        lines.append(f"H2 (log kéo giãn tối, nén sáng): hệ số khuếch đại gradient TB vùng tối = {dark:.2f}, "
                     f"vùng sáng = {bright:.2f} -> {'ỦNG HỘ' if dark > bright else 'KHÔNG ủng hộ'}.")

    # H3 + H4
    for deg in ("offset", "noisy", "bright"):
        g = gt[(gt.degrade == deg) & (gt["transform"] == "gamma") & (gt.c == 1.0)]
        if not len(g):
            continue
        m = g.groupby("gamma")[["psnr", "f1_topk", "entropy", "mean"]].mean()
        best_g = m.psnr.idxmax()
        lines.append(f"H3 [{deg}] (c=1): γ cho PSNR TB cao nhất = {best_g:g} ({m.psnr.max():.1f} dB); "
                     f"F1 TB tại γ=0.2 / 1.0 / 3.0 = {m.f1_topk.get(0.2, NAN):.3f} / "
                     f"{m.f1_topk.get(1.0, NAN):.3f} / {m.f1_topk.get(3.0, NAN):.3f}.")
    for deg in ("offset", "noisy"):
        g = gt[(gt.degrade == deg) & (gt["transform"] == "gamma")]
        base = gt[(gt.degrade == deg) & (gt["transform"] == "identity")]
        if not len(g):
            continue
        m = g.groupby(["gamma", "c"])[["psnr", "f1_topk"]].mean()
        bf1 = base.f1_topk.mean()
        ok = m[m.f1_topk >= bf1 - 0.10]
        if len(ok):
            (bg, bc), r = ok.psnr.idxmax(), ok.loc[ok.psnr.idxmax()]
            lines.append(f"H4 [{deg}]: vùng cân bằng γ={bg:g}, c={bc:g}: PSNR {base.psnr.mean():.1f} -> "
                         f"{r.psnr:.1f} dB, F1 {bf1:.3f} -> {r.f1_topk:.3f} (cho phép giảm ≤ 0.10).")

    # H5
    g = gt[(gt["transform"] == "gamma") & (gt.degrade.isin(["offset", "noisy"]))]
    if len(g):
        uns = g[g.sat_high < 0.001]
        spread_f1 = uns.groupby(["image", "degrade", "gamma"]).f1_topk.agg(lambda s: s.max() - s.min()).mean()
        spread_mean = uns.groupby(["image", "degrade", "gamma"])["mean"].agg(lambda s: s.max() - s.min()).mean()
        sat = g[g.sat_high >= 0.01]
        lines.append(f"H5 (c chỉ đổi thang gradient khi chưa bão hoà): biên độ F1 theo c = {spread_f1:.3f}, "
                     f"biên độ mean theo c = {spread_mean:.1f} mức xám; "
                     f"F1 TB khi bão hoà ≥1% = {sat.f1_topk.mean() if len(sat) else NAN:.3f} "
                     f"so với chưa bão hoà = {uns.f1_topk.mean():.3f}.")

    ctrl = gt[(gt.degrade == "control_gamma") & (gt["transform"] == "gamma")]
    off = gt[(gt.degrade == "offset") & (gt["transform"] == "gamma")]
    if len(ctrl) and len(off):
        lines.append(f"TN8 (đối chứng thiên vị): PSNR tốt nhất TB – control_gamma = "
                     f"{ctrl.groupby('image').psnr.max().mean():.1f} dB, offset = "
                     f"{off.groupby('image').psnr.max().mean():.1f} dB.")
    return lines


def environment_info():
    """Ghi phiên bản để tái lập (NFR3)."""
    import cv2
    import matplotlib
    return {"imgtx": __version__, "python": platform.python_version(), "numpy": np.__version__,
            "opencv": cv2.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform()}
