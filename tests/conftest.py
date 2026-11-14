import os
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def ramp():
    return np.tile(np.arange(256, dtype=np.uint8), (16, 1))


@pytest.fixture
def rand_img():
    return np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture(scope="session")
def camera():
    from imgtx.models.io_utils import read_rgb, resize_max, to_gray
    return to_gray(resize_max(read_rgb(os.path.join(ROOT, "data", "raw", "camera.png")), 512))
