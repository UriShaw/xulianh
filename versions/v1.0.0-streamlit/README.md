# Khảo sát âm bản – logarit – gamma + Sobel

## Cài đặt & chạy
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q          # chạy test
```
Mở http://localhost:8501. Camera trình duyệt chỉ hoạt động trên `localhost` hoặc HTTPS.

## Cách dùng
1. Thanh bên: chọn **Thư viện** (tải ảnh), **Camera** (chụp) hoặc **Ảnh mẫu**.
2. Chọn xám hoặc màu (áp dụng lên kênh V của HSV).
3. Muốn so biên với ảnh gốc: bật **Mô phỏng suy giảm** (ảnh chọn = ảnh gốc, ảnh đầu vào = ảnh suy giảm).
4. Kéo thanh trượt gain / γ / c, mật độ biên k.

## Cấu trúc
`imgtx/transforms.py` (3 phép + LUT), `analysis.py` (histogram, Sobel, F1/IoU, PSNR),
`degrade.py`, `experiment.py` (quét tham số), `viz.py`, `io_utils.py`, `app.py` (giao diện).
