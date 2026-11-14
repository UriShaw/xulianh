"""MODEL – dữ liệu và nghiệp vụ, không phụ thuộc giao diện.

transforms  : âm bản, logarit, gamma, LUT, registry
degrade     : mô phỏng ảnh thiếu/quá sáng, cặp (ảnh gốc, ảnh đầu vào)
analysis    : histogram, thống kê, Sobel, độ đo biên, MSE/PSNR, evaluate
experiment  : cấu hình, run_grid, quét tham số, chọn tham số, tóm tắt giả thuyết
io_utils    : nạp/lưu ảnh, CSV, ảnh mẫu, ảnh tổng hợp
session     : trạng thái phiên làm việc trên một ảnh (dùng cho giao diện)
"""
