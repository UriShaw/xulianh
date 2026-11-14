# ImgTX 2.1.0 – MVC

Tách `gui.py` (một file ~900 dòng vừa vẽ vừa tính) thành:
- `models/session.py` – trạng thái phiên làm việc (`ImageSession`, `Settings`, `Result`);
- `views/main_window.py`, `widgets.py`, `charts.py`, `report_figures.py`, `help_text.py` – chỉ hiển thị;
- `controllers/main_controller.py`, `experiment_controller.py`, `cli_controller.py` – xử lý sự kiện.
Thêm `tests/test_mvc.py` chạy controller với view giả. 39 test đạt, độ phủ 85%.
