# ImgTX 2.2.1 – mở nhanh hơn

| Cách đóng gói | Thời gian tới khi hiện cửa sổ* |
| --- | --- |
| 2.2.0 – một file `ImgTX.exe` (giải nén ~95 MB vào thư mục tạm mỗi lần mở) | ~12,5 s |
| 2.2.1 – một file, đã tối ưu nạp thư viện | ~10 s |
| **2.2.1 – thư mục / bộ cài `ImgTX_Setup_2.2.1.exe`** | **~4,5 s** |

\*Đo trong môi trường Windows giả lập (wine); trên máy Windows thật thường nhanh hơn.

Thay đổi:
- Đóng gói kiểu thư mục (`--onedir`) + bộ cài NSIS (`installer/imgtx.nsi`): cài vào `%LOCALAPPDATA%\Programs\ImgTX`,
  không cần quyền Administrator, tạo shortcut Desktop và Start Menu, có mục gỡ cài đặt trong Settings → Apps.
- `models/_lazy.py`: pandas chỉ được nạp khi quét tham số / chạy hàng loạt / xuất CSV (nạp mã lúc mở: 5,2 s → 2,4 s).
- Cửa sổ hiện trước, ảnh mẫu nạp ngay sau đó.
- Bỏ chữ “đề 5”, “(mục 8.2)” khỏi giao diện; thuộc tính file exe ghi “ImgTX”.
