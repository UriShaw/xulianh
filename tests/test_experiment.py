"""T16, T17, T22, T23: tích hợp, hồi quy, thiên vị thiết kế, dòng identity."""
import os
import numpy as np
import pandas as pd
import pytest
from imgtx.models.degrade import build_pair, darken_gamma_control, darken_noisy, darken_offset
from imgtx.models.experiment import (CS, DEFAULT_CONFIG, GAMMAS, load_config, product_grid, run_grid,
                              select_params, summarize_hypotheses, sweep)
from imgtx.models.io_utils import save_image, synthetic_images

from conftest import ROOT

GOLDEN = os.path.join(ROOT, "tests", "golden_camera_offset.csv")


def _cfg(tmp_path, n=3):
    for name, img in list(synthetic_images(64).items())[:n]:
        save_image(str(tmp_path / f"{name}.png"), img)
    cfg = load_config(None)
    cfg["images"] = [{"path": str(tmp_path / "*.png"), "group": "syn", "has_gt": True}]
    cfg["resize"] = 64
    return cfg


def test_T16_pipeline(tmp_path):
    cfg = _cfg(tmp_path)
    df = run_grid(cfg)
    per = 1 + 1 + 5 + 9 * 8
    assert len(df) == 3 * len(cfg["degrade"]) * per
    num = df[[c for c in df.columns if c not in ("mse", "psnr", "grad_corr", "mean_grad_norm",
                                                  "gamma", "c", "gain", "grad_gain_dark", "grad_gain_bright")]]
    assert not num.isna().any().any()
    assert len(select_params(df)) == 3 * len(cfg["degrade"])
    assert any(l.startswith("H1") for l in summarize_hypotheses(df))


def test_T23_identity_row(tmp_path, camera):
    cfg = _cfg(tmp_path, 1)
    cfg["degrade"] = {"offset": {"alpha": 0.4, "beta": 10}}
    df = run_grid(cfg)
    idr = df[df["transform"] == "identity"]
    assert len(idr) == 1
    img = synthetic_images(64)[list(synthetic_images(64))[0]]
    assert abs(idr.iloc[0]["mean"] - darken_offset(img).mean()) < 1e-9


def test_real_image_no_gt(tmp_path):
    cfg = _cfg(tmp_path, 1)
    cfg["images"][0]["has_gt"] = False
    df = run_grid(cfg)
    assert (df.degrade == "none").all() and (df.ref_type == "dau_vao").all() and df.psnr.isna().all()


def _best_psnr(camera, deg_fn, **kw):
    x = deg_fn(camera, **kw)
    return sweep(camera, x, "gamma", {"gamma": GAMMAS, "c": CS}).psnr.max()


def test_T22_degrade_not_in_family(camera):
    assert _best_psnr(camera, darken_offset) < 30
    assert _best_psnr(camera, darken_noisy) < 30
    assert _best_psnr(camera, darken_gamma_control) >= 35


@pytest.mark.slow
def test_T17_regression_golden(camera):
    goc, vao = build_pair(camera, "offset")
    df = pd.concat([sweep(goc, vao, "identity", {}), sweep(goc, vao, "negative", {}),
                    sweep(goc, vao, "log", {"gain": [0.5, 1.0, 1.5]}),
                    sweep(goc, vao, "gamma", {"gamma": [0.4, 0.7, 2.0], "c": [1.0, 2.0]})], ignore_index=True)
    cols = ["mean", "std", "entropy", "f1_topk", "grad_corr", "psnr"]
    if not os.path.exists(GOLDEN):
        df[["transform", "gamma", "c", "gain"] + cols].to_csv(GOLDEN, index=False)
        pytest.skip("Đã tạo golden values lần đầu.")
    gold = pd.read_csv(GOLDEN)
    np.testing.assert_allclose(df[cols].to_numpy(float), gold[cols].to_numpy(float), atol=1e-6)


def test_product_grid():
    assert len(product_grid({"gamma": [1, 2], "c": [1, 2, 3]})) == 6
    assert product_grid({}) == [{}]


def test_cli_pipeline_outputs(tmp_path):
    """Tích hợp: run_pipeline xuất đủ CSV, hình, hypotheses.md, run_info.json (dùng cả viz)."""
    from imgtx.cli import main
    from imgtx.controllers.experiment_controller import run_pipeline
    (tmp_path / "img").mkdir()
    cfg = _cfg(tmp_path / "img", 2)
    out = tmp_path / "out"
    df, lines, figs = run_pipeline(cfg, str(out))
    for f in ("tables/results.csv", "tables/tham_so_de_xuat.csv", "tables/tn6_do_nhay_k.csv",
              "hypotheses.md", "run_info.json"):
        assert (out / f).exists(), f
    assert len(figs) >= 6 and all(os.path.exists(p) for p in figs)
    # chạy lại cho cùng CSV (tái lập, mục 9.3)
    df2, _, _ = run_pipeline(cfg, str(tmp_path / "out2"), with_k=False)
    pd.testing.assert_frame_equal(df, df2)
    assert main(["--config", "khong_ton_tai.yaml", "--images", str(tmp_path / "img" / "*.png"),
                 "--out", str(tmp_path / "out3")]) == 0
