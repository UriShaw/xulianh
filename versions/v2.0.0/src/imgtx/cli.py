"""Dòng lệnh: python -m imgtx.cli --config configs/default.yaml

Chạy toàn bộ lưới thí nghiệm TN1–TN8, xuất outputs/tables/*.csv, outputs/figures/*.png
và outputs/hypotheses.md (tóm tắt số liệu cho H1–H5).
"""
import argparse
import json
import os
import time

from . import viz
from .degrade import build_pair
from .experiment import (environment_info, k_sensitivity, load_config, run_grid, select_params,
                         summarize_hypotheses, _image_specs)
from .io_utils import read_rgb, resize_max, save_csv, to_gray, write_synthetic
from .transforms import TRANSFORMS, apply_lut


def make_figures(df, cfg, outdir, specs=None):
    """Sinh các hình cho báo cáo từ bảng kết quả."""
    fig_dir = os.path.join(outdir, "figures")
    paths = []

    def save(fig, name):
        p = os.path.join(fig_dir, name)
        viz.save_fig(fig, p)
        paths.append(p)

    save(viz.plot_curves(1.0, 0.5, 1.0), "hinh1_duong_cong_T.png")

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
        save(viz.plot_tn1_grid(vao, outs, cfg["edge"]["density"], ref_goc=goc), "tn1_luoi_anh_hist_bien.png")

    gt = df[df.ref_type == "goc"]
    g1 = gt[(gt["transform"] == "gamma") & (gt.c == 1.0)]
    if len(g1):
        save(viz.plot_metric_by_param(g1, "gamma", [("mean", "Mean"), ("entropy", "Entropy (bit)"),
                                                     ("f1_topk", "F1 biên (so ảnh gốc)"),
                                                     ("psnr", "PSNR (dB)")]), "tn2_anh_huong_gamma.png")
    lg = gt[gt["transform"] == "log"]
    if len(lg):
        save(viz.plot_metric_by_param(lg, "gain", [("sat_high", "Tỉ lệ bão hoà 255"),
                                                    ("dyn_range", "Dải động P99−P1"),
                                                    ("f1_topk", "F1 biên"), ("psnr", "PSNR (dB)")]),
             "tn3_anh_huong_gain_log.png")
    base = df[df["transform"].isin(["identity", "negative"]) | ((df["transform"] == "log") & (df.gain == 1.0))
              | ((df["transform"] == "gamma") & (df.gamma == 0.5) & (df.c == 1.0))].copy()
    if len(base):
        base["phep"] = base["transform"].where(base["transform"] != "gamma", "gamma 0.5")
        fig = viz.plot_bar(base.groupby("group").f1_topk.mean(), "TN4: F1 biên TB theo nhóm ảnh", "F1")
        save(fig, "tn4_theo_nhom_anh.png")
    for deg in ("offset", "noisy", "bright"):
        g = gt[(gt["transform"] == "gamma") & (gt.degrade == deg)]
        if len(g):
            save(viz.plot_heatmaps(g, [("f1_topk", "F1 biên"), ("grad_corr", "Tương quan ∇"),
                                       ("psnr", "PSNR (dB)"), ("mean", "Mean")]),
                 f"tn7_heatmap_gamma_c_{deg}.png")
    gg = gt[gt["transform"] == "gamma"]
    if len(gg):
        save(viz.plot_bar(gg.groupby(["degrade", "image"]).psnr.max().groupby("degrade").mean(),
                          "TN8: PSNR tốt nhất TB theo kiểu suy giảm", "dB"), "tn8_doi_chung_thien_vi.png")
    return paths


def run_pipeline(cfg, outdir, progress=None, with_k=True):
    """Chạy toàn bộ: lưới thí nghiệm + chọn tham số + TN6 + hình + tóm tắt giả thuyết."""
    t0 = time.time()
    specs = _image_specs(cfg)
    if not specs:
        raise ValueError("Không tìm thấy ảnh nào theo cấu hình.")
    df = run_grid(cfg, progress, specs)
    tab = os.path.join(outdir, "tables")
    save_csv(df, os.path.join(tab, "results.csv"))
    save_csv(select_params(df, cfg), os.path.join(tab, "tham_so_de_xuat.csv"))
    if with_k:
        if progress:
            progress(0, 1, "TN6 – độ nhạy ngưỡng biên")
        save_csv(k_sensitivity(cfg, [s for s in specs if s["has_gt"]][:5]), os.path.join(tab, "tn6_do_nhay_k.csv"))
    figs = make_figures(df, cfg, outdir, specs)
    lines = summarize_hypotheses(df)
    with open(os.path.join(outdir, "hypotheses.md"), "w", encoding="utf-8") as f:
        f.write("# Tóm tắt số liệu kiểm chứng giả thuyết\n\n")
        f.writelines(f"- {ln}\n" for ln in lines)
        f.write(f"\nSố ảnh: {len(specs)} – số dòng kết quả: {len(df)} – thời gian: {time.time() - t0:.1f} s\n")
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(os.path.join(outdir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"config": clean, "env": environment_info()}, f, ensure_ascii=False, indent=2, default=str)
    return df, lines, figs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="imgtx", description="Khảo sát âm bản – logarit – gamma + Sobel")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="thư mục kết quả (mặc định theo cấu hình)")
    ap.add_argument("--images", nargs="*", help="ghi đè danh sách ảnh (glob), coi là có ảnh gốc")
    ap.add_argument("--gen-synthetic", action="store_true", help="sinh ảnh tổng hợp vào data/synthetic rồi thoát")
    a = ap.parse_args(argv)

    cfg = load_config(a.config if os.path.exists(a.config) else None)
    root = cfg.get("_root", ".")
    if a.gen_synthetic:
        for p in write_synthetic(os.path.join(root, "data", "synthetic")):
            print("Đã tạo", p)
        return 0
    if a.images:
        cfg["images"] = [{"path": p, "group": "tuy_chon", "has_gt": True} for p in a.images]
    out = a.out or os.path.join(root, cfg.get("output", "outputs"))

    def prog(i, n, text):
        print(f"[{i}/{n}] {text}", flush=True)

    df, lines, figs = run_pipeline(cfg, out, prog)
    print("\n".join(lines))
    print(f"Đã ghi {len(df)} dòng -> {os.path.join(out, 'tables', 'results.csv')}; {len(figs)} hình.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
