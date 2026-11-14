"""T01–T09, T18: phép biến đổi và LUT."""
import time
import numpy as np
import pytest
from imgtx.models.transforms import (TRANSFORMS, apply_lut, gamma_transform, identity, log_transform,
                              negative, register)


def test_T01_negative_involution(ramp):
    assert np.array_equal(negative(negative(ramp)), ramp)


def test_T02_negative_endpoints():
    assert negative(np.array([[0, 255]], np.uint8)).tolist() == [[255, 0]]


def test_T03_log_endpoints_monotonic(ramp):
    y = log_transform(ramp)[0]
    assert y[0] == 0 and y[255] == 255 and np.all(np.diff(y.astype(int)) >= 0)


def test_T04_gamma_identity(ramp):
    assert np.array_equal(gamma_transform(ramp, 1.0), ramp)


def test_T05_gamma_direction(ramp):
    assert gamma_transform(ramp, 0.5)[0, 64] > 64 > gamma_transform(ramp, 2.0)[0, 64]


@pytest.mark.parametrize("g", [0, -1.0])
def test_T06_gamma_invalid(ramp, g):
    with pytest.raises(ValueError):
        gamma_transform(ramp, g)


def test_T06b_log_gain_invalid(ramp):
    with pytest.raises(ValueError):
        log_transform(ramp, gain=0)


@pytest.mark.parametrize("func,kw", [(negative, {}), (log_transform, {"gain": 1.3}),
                                     (gamma_transform, {"gamma": 0.4}),
                                     (gamma_transform, {"gamma": 2.0, "c": 1.5})])
def test_T07_lut_equals_direct(rand_img, func, kw):
    assert np.array_equal(apply_lut(rand_img, func, **kw), func(rand_img, **kw))


def test_T08_dtype_shape(ramp):
    for y in (negative(ramp), log_transform(ramp), gamma_transform(ramp, 2), identity(ramp)):
        assert y.dtype == np.uint8 and y.shape == ramp.shape


@pytest.mark.parametrize("bad", [np.zeros((8, 8), np.float64), np.zeros((8, 8, 3), np.uint8)])
def test_T09_bad_input(bad):
    with pytest.raises(ValueError):
        negative(bad)
    with pytest.raises(ValueError):
        gamma_transform(bad, 0.5)


def test_registry_extensible(ramp):
    @register("_test_half")
    def half(img):
        return (img // 2).astype(np.uint8)
    try:
        assert np.array_equal(apply_lut(ramp, TRANSFORMS["_test_half"]), ramp // 2)
    finally:
        TRANSFORMS.pop("_test_half")


@pytest.mark.slow
def test_T18_lut_performance():
    img = np.random.default_rng(0).integers(0, 256, (3000, 4000), dtype=np.uint8)
    apply_lut(img, gamma_transform, gamma=0.5)          # khởi động
    t = time.perf_counter()
    apply_lut(img, gamma_transform, gamma=0.5)
    assert time.perf_counter() - t < 0.5
