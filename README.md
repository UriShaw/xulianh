# ImgTX – Khảo sát âm bản, logarit, gamma + so sánh biên Sobel

Bài tập lớn học phần **Xử lý ảnh và thị giác máy tính (IT3243)** – Trường Đại học Công nghệ Đông Á, đề tài số 5.
Phiên bản 2.2.0 · Mã nguồn: https://github.com/UriShaw/xulianh · Báo cáo: `docs/BaoCao_De5.pdf` (bản Word: `docs/BaoCao_De5.docx`)

Làm lại theo **Tài liệu phân tích & thiết kế v2** (đề tài 5, môn Xử lý ảnh).

## Chạy ngay (không cần cài Python)
Tải **`ImgTX.exe`** ở mục *Releases* của repo (hoặc dùng file trong thư mục dự án) rồi bấm đúp. Lần mở đầu mất khoảng 5–15 giây vì exe tự giải nén thư viện.
Nếu Windows SmartScreen cảnh báo (exe chưa ký số): chọn *More info → Run anyway*.

## Chạy từ mã nguồn
```
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_gui.py                   # giao diện desktop
python -m pytest -q                 # 44 test (T01–T24, MVC, đối chiếu Sobel/Otsu tự cài với OpenCV)
python -m pytest -q -m "not slow" --cov=src/imgtx   # độ phủ 85% (bỏ phần vẽ cửa sổ)
set PYTHONPATH=src && python -m imgtx.cli --config configs/default.yaml   # chạy TN1–TN8
```
Kết quả CLI: `outputs/tables/results.csv`, `tham_so_de_xuat.csv`, `tn6_do_nhay_k.csv`,
`outputs/figures/*.png`, `outputs/hypotheses.md`, `outputs/run_info.json`.

## Tự đóng gói lại exe
Bấm đúp `build_exe.bat` (cần Python 3.10–3.12) → `dist\ImgTX.exe`. Logo: `assets/imgtx.ico`
(sinh bằng `tools_make_logo.py`), thông tin tệp: `version_info.txt`.

## Cấu trúc MVC
```
src/imgtx/
├── main.py                     ghép Model – View – Controller và chạy giao diện
├── cli.py                      lệnh trong tài liệu: python -m imgtx.cli → controllers/cli_controller
├── __main__.py                 python -m imgtx (giao diện) | python -m imgtx --config ... (dòng lệnh)
├── models/                     MODEL – dữ liệu + nghiệp vụ, không phụ thuộc giao diện
│   ├── transforms.py           identity, âm bản, log, gamma, LUT, registry @register, kênh V
│   ├── degrade.py              offset, noisy, bright, control_gamma (đối chứng), build_pair
│   ├── analysis.py             histogram, thống kê, Sobel, top-k, Otsu, F1/IoU, grad_corr, MSE/PSNR, evaluate
│   ├── experiment.py           cấu hình YAML, run_grid (có dòng identity), sweep, select_params, H1–H5
│   ├── io_utils.py             nạp/lưu ảnh (đường dẫn có dấu), CSV, ảnh mẫu, ảnh tổng hợp
│   └── session.py              ImageSession + Settings + Result: trạng thái phiên làm việc trên 1 ảnh
├── views/                      VIEW – chỉ hiển thị, không tính toán
│   ├── main_window.py          cửa sổ Tkinter: đọc dữ liệu nhập, vẽ kết quả, phát sự kiện
│   ├── widgets.py              theme ttk, Slider, định dạng số
│   ├── charts.py               biểu đồ Matplotlib (T(r), histogram, quét, heatmap)
│   ├── report_figures.py       bộ hình TN1–TN8 cho báo cáo
│   └── help_text.py            nội dung tab Hướng dẫn
└── controllers/                CONTROLLER – nhận sự kiện, gọi model, cập nhật view
    ├── main_controller.py      điều khiển cửa sổ chính (debounce, quét, xuất, chạy nền)
    ├── experiment_controller.py run_pipeline: lưới thí nghiệm → CSV + hình + hypotheses.md
    └── cli_controller.py       tham số dòng lệnh
configs/default.yaml            danh sách ảnh, lưới γ/c/gain, suy giảm, k biên, quy tắc chọn tham số
data/raw (6 ảnh chuẩn) · data/detail (4 ảnh nhiều chi tiết) · data/medical (4 ảnh y tế)
data/synthetic (4 ảnh tổng hợp) · data/real (4 ảnh thật không có ảnh gốc)
docs/                           báo cáo bài tập lớn (.pdf, .docx)
versions/                       CHANGELOG.md + ghi chú và ảnh chụp mã nguồn từng phiên bản
tests/                          T01–T24 + test_mvc.py (controller với view giả, không mở cửa sổ)
```
Luồng một thao tác: kéo thanh trượt → `MainWindow` phát sự kiện `update` → `MainController.refresh()`
đọc `Settings` từ view → `ImageSession.compute()` trả `Result` → `MainWindow.show_result()` vẽ.

## Thuật toán tự cài đặt (yêu cầu của đề thi)
Ba phép biến đổi + LUT, histogram và thống kê, **Sobel** (tương quan 3×3, `models/analysis.py: correlate3x3, sobel`),
**ngưỡng Otsu** (`otsu_threshold`), F1/IoU, tương quan gradient, MSE/PSNR – đều viết bằng NumPy.
OpenCV chỉ dùng để đọc/ghi ảnh và để đối chiếu: Sobel tự cài khớp `cv2.Sobel` với sai khác lớn nhất 0,
Otsu tự cài trùng ngưỡng `cv2.threshold(..., THRESH_OTSU)` (test `test_manual_sobel_equals_opencv`,
`test_manual_otsu_equals_opencv`; trong ứng dụng: tab *Tổng quan đề 5* → *Kiểm chứng*).

Ghi chú tái lập: top-k dùng sắp xếp ổn định nên kết quả không đổi giữa các phiên bản NumPy;
phiên bản thư viện được ghim trong `requirements.txt` và ghi vào `run_info.json` mỗi lần chạy.

## Kết quả chính (outputs/hypotheses.md)
Trên 22 ảnh, 6.004 tổ hợp: âm bản giữ nguyên biên tuyệt đối (H1); log khuếch đại gradient vùng tối ~2,1 lần (H2);
γ tốt nhất cho ảnh thiếu sáng ≈ 0,4–0,7, cho ảnh quá sáng ≈ 3 (H3); điểm cân bằng γ = 0,7, c = 2 nâng PSNR 10,5 → 27,6 dB
với F1 biên 0,959 → 0,894 (H4); khi chưa bão hoà, c gần như không đổi biên (H5).

## Phiên bản
Xem `versions/CHANGELOG.md`. Thư mục `versions/` trên máy còn chứa `ImgTX.exe` của từng bản (không đưa lên Git; tải ở Releases).
