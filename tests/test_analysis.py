"""T10–T15, T19–T21, T24: histogram, Sobel, độ đo, evaluate."""
import numpy as np
from imgtx.models.analysis import edge_map_topk, evaluate, hist_stats, mse, psnr, sobel
from imgtx.models.degrade import darken_offset
from imgtx.models.experiment import run_one
from imgtx.models.transforms import gamma_transform, negative


def test_T10_sobel_constant_zero():
    m, _ = sobel(np.full((32, 32), 100, np.uint8))
    assert np.allclose(m[2:-2, 2:-2], 0)


def test_T11_sobel_negative_invariant(rand_img):
    assert np.allclose(sobel(rand_img)[0], sobel(negative(rand_img))[0])


def test_T12_evaluate_self(rand_img):
    r = evaluate(rand_img, rand_img, rand_img)
    assert r["f1_topk"] > 0.999 and r["grad_corr"] > 0.999 and r["ref_type"] == "goc"


def test_T13_step_edge_location():
    img = np.zeros((32, 32), np.uint8)
    img[:, 16:] = 255
    e = edge_map_topk(sobel(img)[0], density=2 * 32 / 1024)
    cols = np.where(e.any(axis=0))[0]
    assert set(cols) <= {14, 15, 16, 17} and len(cols) > 0


def test_T14_entropy():
    assert hist_stats(np.full((8, 8), 7, np.uint8))["entropy"] == 0
    two = np.zeros((8, 8), np.uint8)
    two[:, 4:] = 200
    assert abs(hist_stats(two)["entropy"] - 1.0) < 1e-12


def test_T15_negative_mean_std(rand_img):
    a, b = hist_stats(rand_img), hist_stats(negative(rand_img))
    assert abs(b["mean"] - (255 - a["mean"])) < 1e-9 and abs(a["std"] - b["std"]) < 1e-9


def test_T19_mse_psnr_known():
    a = np.full((8, 8), 100, np.uint8)
    b = a + 10
    assert mse(a, b) == 100.0
    assert abs(psnr(a, b) - 28.1308) < 1e-3
    assert psnr(a, a) == 100.0


def test_T20_no_ground_truth(rand_img):
    r = evaluate(None, rand_img, negative(rand_img))
    assert r["ref_type"] == "dau_vao" and np.isnan(r["mse"]) and np.isnan(r["psnr"])
    assert r["f1_topk"] > 0.999
    assert not any(np.isnan(v) for k, v in r.items() if k not in ("mse", "psnr", "ref_type"))


def test_T21_negative_f1_equals_identity():
    gt = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
    x = darken_offset(gt)
    a, b = evaluate(gt, x, x), evaluate(gt, x, negative(x))
    assert abs(a["f1_topk"] - b["f1_topk"]) < 1e-12


def test_T24_c_invariance_unsaturated(camera):
    x = darken_offset(camera)
    _, base = run_one("gamma", camera, x, gamma=0.5, c=1.0)
    ref = gamma_transform(x, 0.5, 1.0)
    for c in (0.5, 1.5):
        out = gamma_transform(x, 0.5, c)
        assert (out == 255).sum() == 0
        r = evaluate(ref, x, out)          # so với kết quả c = 1
        assert r["grad_corr"] >= 0.99


# ---------------------------------------------------------------- thuật toán tự cài so với OpenCV
import cv2
from imgtx.models.analysis import correlate3x3, otsu_threshold, SOBEL_GX


def test_manual_sobel_equals_opencv(camera, rand_img):
    for img in (camera, rand_img, np.full((9, 9), 7, np.uint8)):
        m1, t1 = sobel(img, method="manual")
        m2, t2 = sobel(img, method="opencv")
        assert np.allclose(m1, m2, atol=1e-9) and np.allclose(t1, t2, atol=1e-9)


def test_correlate_known_value():
    f = np.array([[0, 0, 10], [0, 0, 10], [0, 0, 10]], np.float64)
    assert correlate3x3(f, SOBEL_GX)[1, 1] == 40.0          # (1+2+1)·10


def test_sobel_l1_norm(rand_img):
    m_l1, _ = sobel(rand_img, norm="l1")
    m_l2, _ = sobel(rand_img)
    assert np.all(m_l1 >= m_l2 - 1e-9) and np.all(m_l1 <= np.sqrt(2) * m_l2 + 1e-9)


def test_manual_otsu_equals_opencv(camera, rand_img):
    two = np.zeros((20, 20), np.uint8)
    two[:, 10:] = 200
    for img in (camera, rand_img, two):
        t_cv, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        assert otsu_threshold(img) == int(t_cv)
