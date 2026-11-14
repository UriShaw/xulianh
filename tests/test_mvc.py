"""Kiểm thử model phiên làm việc và controller bằng một view giả (không cần mở cửa sổ Tk)."""
import os

import numpy as np
import pytest

from imgtx.controllers.main_controller import MainController
from imgtx.models.session import ImageSession, Settings

from conftest import ROOT

CAMERA = os.path.join(ROOT, "data", "raw", "camera.png")


class FakeView:
    """Ghi lại mọi lệnh controller gửi cho view."""

    def __init__(self, settings=None):
        self.settings = settings or Settings(max_side=128)
        self.calls = []
        self.handlers = {}
        self.save_path = None
        self.batch = None

    def bind_events(self, h):
        self.handlers = h

    def __getattr__(self, name):          # set_samples, show_file, show_result, status, error, ...
        def rec(*a, **k):
            self.calls.append((name, a))
        return rec

    def read_settings(self):
        return self.settings

    def ask_save(self, *a):
        return self.save_path

    def batch_inputs(self):
        return self.batch

    def after(self, ms, fn=None, *a):
        return "id"

    def names(self):
        return [c[0] for c in self.calls]


def test_session_compute_and_sweep():
    m = ImageSession()
    w, h = m.load(CAMERA)
    assert (w, h) == (512, 512)
    r = m.compute(Settings(max_side=128))
    assert r.has_gt and r.out.shape == (128, 128) and r.out.dtype == np.uint8
    assert len(r.metric_rows()) == 12
    df, base = m.run_sweep("grid", Settings(max_side=128))
    assert len(df) == 72 and base["ref_type"] == "goc"
    assert m.best_in_sweep() is not None


def test_session_real_image_and_color():
    m = ImageSession()
    m.load(os.path.join(ROOT, "data", "raw", "coffee.png"))
    r = m.compute(Settings(max_side=96, mode="v", degrade="none", degrade_params={},
                           transform="log", params={"gain": 1.2}))
    assert not r.has_gt and r.shown_out.ndim == 3 and np.isnan(r.row["psnr"])
    with pytest.raises(ValueError):
        m.compute(Settings(max_side=96, transform="gamma", params={"gamma": 0, "c": 1}))


def test_controller_flow(tmp_path):
    v = FakeView()
    c = MainController(ImageSession(), v)
    c.start(run_mainloop=False)
    assert {"set_samples", "show_file", "show_result"} <= set(v.names())
    v.handlers["sweep"]("gamma")
    assert "show_sweep" in v.names()
    v.save_path = str(tmp_path / "kq.png")
    v.handlers["save_output"]()
    assert (tmp_path / "kq.png").exists() and (tmp_path / "kq_bien.png").exists()
    v.save_path = str(tmp_path / "quet.csv")
    v.handlers["save_sweep"]()
    assert (tmp_path / "quet.csv").exists()


def test_controller_batch_config(tmp_path):
    c = MainController(ImageSession(), FakeView())
    cfg = c.build_batch_config({"gt": "", "real": str(tmp_path), "out": str(tmp_path / "o"),
                                "synthetic": True, "with_k": False}, 0.15)
    assert [i["has_gt"] for i in cfg["images"]] == [True, True, True, True, False]
    assert cfg["edge"]["density"] == 0.15 and len(os.listdir(tmp_path / "o" / "_anh_tong_hop")) == 4


def test_controller_bad_image(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_bytes(b"khong phai anh")
    v = FakeView()
    c = MainController(ImageSession(), v)
    c._load(str(bad))
    assert v.calls[-1][0] == "error"


def test_overview_and_verify():
    v = FakeView()
    c = MainController(ImageSession(), v)
    c.start(run_mainloop=False)
    v.handlers["overview"]()
    name, args = [x for x in v.calls if x[0] == "show_overview"][-1]
    items = args[0]
    assert [t for t, *_ in items][:2] == ["Ảnh đầu vào", "Âm bản"] and len(items) == 6
    assert abs(items[0][2]["f1_topk"] - items[1][2]["f1_topk"]) < 1e-12        # H1
    v.handlers["verify"]()
    res = [x for x in v.calls if x[0] == "show_verify"][-1][1][0]
    assert res["max_diff"] < 1e-9 and res["otsu_manual"] == res["otsu_opencv"]
