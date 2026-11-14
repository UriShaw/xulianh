# Lịch sử phiên bản – ImgTX (đề tài 5)

Mỗi phiên bản có một thư mục trong `versions/`. Mã nguồn hiện hành luôn ở `src/`; thư mục phiên bản giữ
ghi chú thay đổi, ảnh chụp mã nguồn của các bản cũ cần đối chiếu, và (chỉ trên máy, không đưa lên Git)
file `ImgTX.exe` của bản đó. File exe được đính kèm ở mục *Releases* trên GitHub.

| Phiên bản | Ngày | Nội dung chính | Thư mục |
| --- | --- | --- | --- |
| 2.2.1 | 14/11/2026 | Mở ứng dụng nhanh gần gấp 3 (12,5 s → 4,5 s): đóng gói kiểu thư mục + bộ cài `ImgTX_Setup`, chỉ nạp pandas khi cần, hiện cửa sổ trước rồi mới nạp ảnh mẫu; bỏ chữ “đề 5” và tham chiếu mục tài liệu khỏi giao diện | `v2.2.1/` |
| 2.2.0 | 14/11/2026 | Sobel và Otsu tự cài (khớp OpenCV, sai khác 0); tab *Tổng quan đề 5*; nút kiểm chứng; bộ ảnh 4 nhóm (22 ảnh); lưới c mở rộng tới 4 | `v2.2.0/` |
| 2.1.0 | 14/11/2026 | Tổ chức lại mã nguồn theo MVC (models / views / controllers); test controller với view giả | `v2.1.0-mvc/` |
| 2.0.0 | 14/11/2026 | Làm lại theo tài liệu thiết kế v2: CLI, cấu hình YAML, `run_grid` có dòng identity, T01–T24, giao diện desktop, đóng gói `ImgTX.exe` | `v2.0.0/` (kèm `src/`) |
| 1.0.0 | 11/2026 | Bản đầu: giao diện Streamlit chạy trên trình duyệt, 13 test | `v1.0.0-streamlit/` (mã nguồn gốc) |
