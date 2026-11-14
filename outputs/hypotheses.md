# Tóm tắt số liệu kiểm chứng giả thuyết

- H1 (âm bản giữ nguyên biên): chênh lệch lớn nhất F1/grad_corr giữa âm bản và identity = 0.00e+00 trên 76 cặp ảnh -> ỦNG HỘ.
- H2 (log kéo giãn tối, nén sáng): hệ số khuếch đại gradient TB vùng tối = 2.13, vùng sáng = 1.00 -> ỦNG HỘ.
- H3 [offset] (c=1): γ cho PSNR TB cao nhất = 0.4 (22.6 dB); F1 TB tại γ=0.2 / 1.0 / 3.0 = 0.671 / 0.959 / 0.536.
- H3 [noisy] (c=1): γ cho PSNR TB cao nhất = 0.4 (21.9 dB); F1 TB tại γ=0.2 / 1.0 / 3.0 = 0.455 / 0.728 / 0.471.
- H3 [bright] (c=1): γ cho PSNR TB cao nhất = 3 (13.8 dB); F1 TB tại γ=0.2 / 1.0 / 3.0 = 0.705 / 0.756 / 0.722.
- H4 [offset]: vùng cân bằng γ=0.7, c=2: PSNR 10.5 -> 27.6 dB, F1 0.959 -> 0.894 (cho phép giảm ≤ 0.10).
- H4 [noisy]: vùng cân bằng γ=0.7, c=2: PSNR 10.5 -> 24.5 dB, F1 0.728 -> 0.705 (cho phép giảm ≤ 0.10).
- H5 (c chỉ đổi thang gradient khi chưa bão hoà): biên độ F1 theo c = 0.025, biên độ mean theo c = 61.8 mức xám; F1 TB khi bão hoà ≥1% = 0.423 so với chưa bão hoà = 0.687.
- TN8 (đối chứng thiên vị): PSNR tốt nhất TB – control_gamma = 44.5 dB, offset = 29.4 dB.

Ghi chú: H3–H5 và TN8 tính trên ảnh tự nhiên (bỏ nhóm tổng hợp).

Số ảnh: 22 – số dòng kết quả: 6004 – thời gian: 353.6 s
