"""CONTROLLER – nối cửa sổ chính (view) với phiên làm việc (model).

Nhận sự kiện từ view → đọc Settings → gọi model → đưa kết quả cho view vẽ.
Thí nghiệm hàng loạt chạy trong luồng nền; tiến độ gửi về luồng giao diện qua hàng đợi.
"""
import os
import queue
import subprocess
import sys
import threading

from ..models.experiment import load_config
from ..models.io_utils import resource_path, write_synthetic
from ..models.session import ImageSession
from .experiment_controller import run_pipeline

DEBOUNCE_MS = 150


class MainController:
    def __init__(self, model: ImageSession, view):
        self.model = model
        self.view = view
        self.samples = {}
        self._pending = None
        self._queue = queue.Queue()
        view.bind_events({
            "open": self.on_open,
            "sample": self.on_sample,
            "prepare": self.on_settings_changed,
            "update": self.on_settings_changed,
            "sweep": self.on_sweep,
            "save_output": self.on_save_output,
            "save_metrics": self.on_save_metrics,
            "save_sweep": self.on_save_sweep,
            "run_batch": self.on_run_batch,
            "open_out": self.on_open_out,
            "overview": self.on_overview,
            "verify": self.on_verify,
            "tab": self.on_tab,
        })
        self._overview_key = None

    # ---------------------------------------------------------------- khởi động
    def start(self, run_mainloop=True):
        self.samples = self.model.samples()
        default = "camera" if "camera" in self.samples else next(iter(self.samples), None)
        self.view.set_samples(list(self.samples), default)
        if default:
            self._load(self.samples[default])
        self.view.after(100, self._poll_queue)
        if run_mainloop:
            self.view.mainloop()

    # ---------------------------------------------------------------- ảnh
    def on_open(self):
        path = self.view.ask_open_image()
        if path:
            self._load(path)

    def on_sample(self, name):
        if name in self.samples:
            self._load(self.samples[name])

    def _load(self, path):
        try:
            w, h = self.model.load(path)
        except Exception as e:
            self.view.error("Lỗi đọc ảnh", str(e))
            return
        self.view.show_file(os.path.basename(path), w, h)
        self.refresh()

    def on_settings_changed(self):
        """Thanh trượt kéo liên tục → chỉ tính lại sau khi dừng DEBOUNCE_MS."""
        if self._pending:
            self.view.after_cancel(self._pending)
        self._pending = self.view.after(DEBOUNCE_MS, self.refresh)

    def refresh(self):
        self._pending = None
        if not self.model.loaded:
            return
        try:
            self.view.show_result(self.model.compute(self.view.read_settings()))
        except ValueError as e:
            self.view.error("Tham số không hợp lệ", str(e))

    # ---------------------------------------------------------------- quét tham số
    def on_sweep(self, kind):
        if not self.model.loaded:
            return
        s = self.view.read_settings()
        self.view.busy(True)
        try:
            df, base = self.model.run_sweep(kind, s)
            self.view.show_sweep(kind, df, base, s, self.model.best_in_sweep())
        finally:
            self.view.busy(False)

    # ---------------------------------------------------------------- tổng quan đề 5
    def on_overview(self):
        if not self.model.loaded:
            return
        s = self.view.read_settings()
        self.view.busy(True)
        try:
            items = self.model.overview(s)
            self.view.show_overview(items, self.model.ref_goc is not None, s.density)
            self._overview_key = self.model._prepared_key
        finally:
            self.view.busy(False)

    def on_tab(self, index):
        """Mở tab Tổng quan lần đầu (hoặc sau khi đổi ảnh) thì tự vẽ."""
        if index == getattr(self.view, "TAB_OVERVIEW", -1) and self._overview_key != self.model._prepared_key:
            self.on_overview()

    def on_verify(self):
        if self.model.loaded:
            self.view.show_verify(self.model.verify_core())

    # ---------------------------------------------------------------- xuất
    def on_save_output(self):
        if self.model.result is None:
            return
        path = self.view.ask_save("ket_qua.png", ".png", "PNG")
        if path:
            self.model.export_output(path)
            self.view.status(f"Đã lưu {path} và bản đồ biên.")

    def on_save_metrics(self):
        if self.model.result is None:
            return
        path = self.view.ask_save("chi_so.csv", ".csv", "CSV")
        if path:
            self.model.export_metrics(path)
            self.view.status(f"Đã lưu {path}")

    def on_save_sweep(self):
        if self.model.sweep_df is None:
            self.view.info("Chưa có dữ liệu", "Hãy chạy một lần quét trước.")
            return
        path = self.view.ask_save("quet_tham_so.csv", ".csv", "CSV")
        if path:
            self.model.export_sweep(path)
            self.view.status(f"Đã lưu {path}")

    # ---------------------------------------------------------------- thí nghiệm hàng loạt
    def build_batch_config(self, inp, density):
        """Dựng cấu hình thí nghiệm từ các ô nhập của tab hàng loạt."""
        cfg = load_config(None)
        if inp["gt"]:
            images = [{"path": os.path.join(inp["gt"], "*"), "group": "co_anh_goc", "has_gt": True}]
        else:                       # bộ ảnh đi kèm ứng dụng, chia nhóm như configs/default.yaml
            images = [{"path": resource_path("data", f, "*"), "group": g, "has_gt": True}
                      for f, g in (("raw", "chuan"), ("detail", "nhieu_chi_tiet"), ("medical", "y_te"))]
        if inp["synthetic"]:
            syn = os.path.join(inp["out"], "_anh_tong_hop")
            write_synthetic(syn)
            images.append({"path": os.path.join(syn, "*"), "group": "tong_hop", "has_gt": True})
        real = inp["real"] or (resource_path("data", "real") if not inp["gt"] else "")
        if real:
            images.append({"path": os.path.join(real, "*"), "group": "anh_that", "has_gt": False})
        cfg["images"] = images
        cfg["edge"] = {**cfg["edge"], "density": density}
        return cfg

    def on_run_batch(self):
        inp = self.view.batch_inputs()
        if not inp["out"]:
            self.view.error("Thiếu thư mục", "Chọn thư mục lưu kết quả.")
            return
        cfg = self.build_batch_config(inp, self.view.read_settings().density)
        self.view.batch_started(inp["out"])
        threading.Thread(target=self._batch_worker, args=(cfg, inp["out"], inp["with_k"]), daemon=True).start()

    def _batch_worker(self, cfg, out, with_k):
        try:
            df, lines, figs = run_pipeline(cfg, out, lambda i, n, t: self._queue.put(("prog", i, n, t)), with_k)
            self._queue.put(("done", len(df), lines, len(figs), out))
        except Exception as e:
            self._queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "prog":
                    self.view.batch_progress(*msg[1:])
                elif msg[0] == "done":
                    self.view.batch_done(*msg[1:])
                else:
                    self.view.batch_failed(msg[1])
        except queue.Empty:
            pass
        self.view.after(150, self._poll_queue)

    def on_open_out(self):
        d = self.view.batch_inputs()["out"]
        if not os.path.isdir(d):
            self.view.info("Chưa có", "Thư mục kết quả chưa tồn tại – hãy chạy thí nghiệm trước.")
        elif sys.platform.startswith("win"):
            os.startfile(d)
        else:
            subprocess.Popen(["xdg-open", d])
