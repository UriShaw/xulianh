# ImgTX 2.2.0 – bản nộp bài

- **Thuật toán tự cài**: Sobel (tương quan 3×3, đệm biên phản xạ, chuẩn L2 hoặc L1 |Gx|+|Gy|) và ngưỡng Otsu
  từ histogram; khớp OpenCV với sai khác lớn nhất 0 (`tests/test_analysis.py`).
- **Tab “Tổng quan đề 5”**: ảnh – histogram – biên Sobel của âm bản, log, γ = 0,5 / 1 / 2 trên một màn hình;
  nút *Kiểm chứng* so Sobel/Otsu tự cài với OpenCV (sai khác, thời gian).
- **Dữ liệu**: 22 ảnh chia 5 nhóm – chuẩn (6), nhiều chi tiết (4), y tế (4), tổng hợp (4), ảnh thật không có ảnh gốc (4).
- **Lưới c** của gamma mở rộng thành {0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4} để điểm tối ưu không rơi vào mép lưới (H4).
- 44 test đạt.
