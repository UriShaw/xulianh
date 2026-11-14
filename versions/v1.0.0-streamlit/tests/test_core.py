import numpy as np
import pytest

from imgtx import transforms as T
from imgtx.analysis import (edge_map_topk, evaluate, hist_stats, mse, psnr, sobel)
from imgtx.degrade import darken_offset
from imgtx.experiment import run_one

RAMP = np.arange(256, dtype=np.uint8)


@pytest.fixture
def rnd():
    return np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)


def test_negative_involution(rnd):
    assert np.array_equal(T.negative(T.negative(rnd)), rnd)
    assert list(T.negative(np.array([0, 255], np.uint8))) == [255, 0]


def test_log_endpoints_monotone():
    o = T.log_transform(RAMP)
    assert o[0] == 0 and o[255] == 255 and np.all(np.diff(o.astype(int)) >= 0)


def test_gamma_identity_and_direction():
    assert np.array_equal(T.gamma_transform(RAMP, 1.0), RAMP)
    r = np.array([64], np.uint8)
    assert T.gamma_transform(r, 0.5)[0] > 64 > T.gamma_transform(r, 2.0)[0]


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_params(bad):
    with pytest.raises(ValueError):
        T.gamma_transform(RAMP, bad)
    with pytest.raises(ValueError):
        T.log_transform(RAMP, bad)


def test_input_type(rnd):
    with pytest.raises(ValueError):
        T.negative(rnd.astype(float))
    with pytest.raises(ValueError):
        T.negative(np.zeros((4, 4, 3), np.uint8))


def test_lut_equals_direct(rnd):
    assert np.array_equal(T.apply_lut(rnd, T.gamma_transform, gamma=0.4), T.gamma_transform(rnd, 0.4))
    for f in (T.negative, T.log_transform):
        o = T.apply_lut(rnd, f)
        assert o.dtype == np.uint8 and o.shape == rnd.shape


def test_sobel_constant_and_negative_invariance(rnd):
    mag, _ = sobel(np.full((16, 16), 100, np.uint8))
    assert mag.max() == 0
    assert np.allclose(sobel(rnd)[0], sobel(T.negative(rnd))[0])


def test_step_edge_position():
    img = np.zeros((20, 20), np.uint8)
    img[:, 10:] = 255
    cols = np.where(edge_map_topk(sobel(img)[0], 0.08).any(axis=0))[0]
    assert cols.min() >= 9 and cols.max() <= 10


def test_hist_stats():
    assert hist_stats(np.full((8, 8), 7, np.uint8))["entropy"] == 0
    two = np.array([[0, 255]] * 8, np.uint8)
    assert abs(hist_stats(two)["entropy"] - 1.0) < 1e-12
    s, n = hist_stats(rnd_img := np.random.default_rng(1).integers(0, 256, (32, 32), dtype=np.uint8)), hist_stats(T.negative(rnd_img))
    assert abs(n["mean"] - (255 - s["mean"])) < 1e-9 and abs(n["std"] - s["std"]) < 1e-9


def test_mse_psnr():
    a = np.full((8, 8), 100, np.uint8)
    assert mse(a, a + 10) == 100 and abs(psnr(a, a + 10) - 28.1308) < 1e-3 and psnr(a, a) == 100


def test_evaluate_self_and_no_gt(rnd):
    r = evaluate(rnd, rnd, rnd)
    assert r["ref_type"] == "goc" and abs(r["f1_topk"] - 1) < 1e-12 and abs(r["grad_corr"] - 1) < 1e-12
    r = evaluate(None, rnd, T.log_transform(rnd))
    assert r["ref_type"] == "dau_vao" and np.isnan(r["mse"]) and np.isnan(r["psnr"])
    assert not np.isnan(r["f1_topk"])


def test_h1_negative_equals_identity(rnd):
    goc = rnd
    vao = darken_offset(goc)
    _, a = run_one("identity", goc, vao)
    _, b = run_one("negative", goc, vao)
    assert abs(a["f1_topk"] - b["f1_topk"]) < 1e-12 and np.isnan(b["psnr"])


def test_gamma_improves_psnr_on_dark():
    from skimage import data
    goc = data.camera()
    vao = darken_offset(goc)
    _, ident = run_one("identity", goc, vao)
    _, g = run_one("gamma", goc, vao, gamma=0.7, c=2.0)
    assert g["psnr"] > ident["psnr"] and g["psnr"] < 30
