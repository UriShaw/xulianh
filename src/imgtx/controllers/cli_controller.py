"""CONTROLLER – dòng lệnh: python -m imgtx.cli --config configs/default.yaml

Chạy toàn bộ lưới thí nghiệm TN1–TN8, xuất outputs/tables/*.csv, outputs/figures/*.png
và outputs/hypotheses.md (tóm tắt số liệu cho H1–H5).
"""
import argparse
import os

from ..models.experiment import load_config
from ..models.io_utils import write_synthetic
from .experiment_controller import run_pipeline


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
