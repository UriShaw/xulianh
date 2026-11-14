"""VIEW – hình cho báo cáo (TN1–TN8) vẽ từ bảng kết quả, lưu PNG."""
import os

from . import charts
from ..models.degrade import build_pair
from ..models.experiment import _image_specs
from ..models.io_utils import read_rgb, resize_max, to_gray
from ..models.transforms import TRANSFORMS, apply_lut


def make_figures(df, cfg, outdir, specs=None):
    """Sinh các hình cho báo cáo từ bảng kết quả."""
    fig_dir = os.path.join(outdir, "figures")
    paths = []

    def save(fig, name):
        p = os.path.join(fig_dir, name)
        charts.save_fig(fig, p)
        paths.append(p)

    save(charts.plot_curves(1.0, 0.5, 1.0), "hinh1_duong_cong_T.png")

    # TN1: lưới ảnh – histogram – biên cho ảnh có ground truth đầu tiên
    specs = specs if specs is not None else _image_specs(cfg)
    first = next((s for s in specs if s["has_gt"]), specs[0] if specs else None)
    if first is not None:
        img = to_gray(resize_max(read_rgb(first["path"]), cfg.get("resize")))
        goc, vao = build_pair(img, "offset", **cfg["degrade"].get("offset", {}))
        outs = {"Âm bản": apply_lut(vao, TRANSFORMS["negative"]),
                "Log (gain=1)": apply_lut(vao, TRANSFORMS["log"], gain=1.0),
                "Gamma γ=0.5": apply_lut(vao, TRANSFORMS["gamma"], gamma=0.5),
                "Gamma γ=0.7, c=2": apply_lut(vao, TRANSFORMS["gamma"], gamma=0.7, c=2.0)}
        save(charts.plot_tn1_grid(vao, outs, cfg["edge"]["density"], ref_goc=goc), "tn1_luoi_anh_hist_bien.png")

    gt = df[df.ref_type == "goc"]
    g1 = gt[(gt["transform"] == "gamma") & (gt.c == 1.0)]
    if len(g1):
        save(charts.plot_metric_by_param(g1, "gamma", [("mean", "Mean"), ("entropy", "Entropy (bit)"),
                                                     ("f1_topk", "F1 biên (so ảnh gốc)"),
                                                     ("psnr", "PSNR (dB)")]), "tn2_anh_huong_gamma.png")
    lg = gt[gt["transform"] == "log"]
    if len(lg):
        save(charts.plot_metric_by_param(lg, "gain", [("sat_high", "Tỉ lệ bão hoà 255"),
                                                    ("dyn_range", "Dải động P99−P1"),
                                                    ("f1_topk", "F1 biên"), ("psnr", "PSNR (dB)")]),
             "tn3_anh_huong_gain_log.png")
    base = df[df["transform"].isin(["identity", "negative"]) | ((df["transform"] == "log") & (df.gain == 1.0))
              | ((df["transform"] == "gamma") & (df.gamma == 0.5) & (df.c == 1.0))].copy()
    if len(base):
        base["phep"] = base["transform"].where(base["transform"] != "gamma", "gamma 0.5")
        fig = charts.plot_bar(base.groupby("group").f1_topk.mean(), "TN4: F1 biên TB theo nhóm ảnh", "F1")
        save(fig, "tn4_theo_nhom_anh.png")
    for deg in ("offset", "noisy", "bright"):
        g = gt[(gt["transform"] == "gamma") & (gt.degrade == deg)]
        if len(g):
            save(charts.plot_heatmaps(g, [("f1_topk", "F1 biên"), ("grad_corr", "Tương quan ∇"),
                                       ("psnr", "PSNR (dB)"), ("mean", "Mean")]),
                 f"tn7_heatmap_gamma_c_{deg}.png")
    gg = gt[gt["transform"] == "gamma"]
    if len(gg):
        save(charts.plot_bar(gg.groupby(["degrade", "image"]).psnr.max().groupby("degrade").mean(),
                          "TN8: PSNR tốt nhất TB theo kiểu suy giảm", "dB"), "tn8_doi_chung_thien_vi.png")
    return paths
