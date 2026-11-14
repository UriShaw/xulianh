"""CONTROLLER – điều phối một lần chạy thí nghiệm hàng loạt (dùng chung cho CLI và giao diện).

Gọi model (run_grid, select_params, k_sensitivity, summarize_hypotheses) rồi view (report_figures)
và ghi kết quả: tables/*.csv, figures/*.png, hypotheses.md, run_info.json.
"""
import json
import os
import time

from ..models.experiment import (_image_specs, environment_info, k_sensitivity, run_grid, select_params,
                                 summarize_hypotheses)
from ..models.io_utils import save_csv
from ..views.report_figures import make_figures


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
