"""Nạp thư viện nặng khi dùng lần đầu, không nạp lúc mở ứng dụng (giảm thời gian khởi động).

pandas chỉ cần khi quét tham số, chạy thí nghiệm hàng loạt hoặc xuất CSV; nạp nó ngay lúc mở
ứng dụng tốn ~0,4 s trên Linux và lâu hơn trên Windows.
"""
import importlib


class LazyModule:
    def __init__(self, name):
        self._name = name
        self._mod = None

    def __getattr__(self, attr):
        if self._mod is None:
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)


pd = LazyModule("pandas")
