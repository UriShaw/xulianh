"""VIEW – nội dung tab Hướng dẫn."""

HELP_TEXT = """ImgTX – Khảo sát các phép biến đổi cường độ: âm bản, logarit và gamma

CÔNG THỨC
  • Âm bản:   s = 255 − r
  • Logarit:  s = gain · (255 / ln 256) · ln(1 + r)      (gain = 1 ⇒ T(255) = 255; gain > 1 gây bão hoà)
  • Gamma:    s = 255 · c · (r / 255)^γ                    (γ < 1 làm sáng, γ > 1 làm tối; c > 1 gây bão hoà)
  • Quan hệ gradient: |∇s| = |T′(f)| · |∇f|  ⇒ âm bản giữ nguyên biên (H1).

THUẬT TOÁN TỰ CÀI ĐẶT (yêu cầu bắt buộc của đề thi)
  • Ba phép biến đổi + bảng tra LUT 256 phần tử (models/transforms.py).
  • Histogram, mean, std, entropy, P1/P99 (models/analysis.py – histogram, hist_stats).
  • Sobel: tương quan 3×3 với Gx = [[-1,0,1],[-2,0,2],[-1,0,1]], Gy = Gxᵀ, đệm biên phản xạ;
    M = sqrt(Gx² + Gy²) hoặc |Gx| + |Gy| (models/analysis.py – correlate3x3, sobel).
  • Ngưỡng Otsu: cực đại phương sai giữa hai lớp từ histogram (otsu_threshold).
  • F1/IoU, tương quan gradient, MSE, PSNR.
  OpenCV chỉ dùng để đọc/ghi ảnh và để ĐỐI CHIẾU: tab "Tổng quan đề 5" → nút Kiểm chứng.

CÁCH DÙNG
  1. Mở ảnh hoặc chọn ảnh mẫu. Chọn ảnh xám hoặc màu (biến đổi kênh V của HSV, giữ sắc độ).
  2. Ảnh tham chiếu: chọn một kiểu suy giảm để coi ảnh đang mở là ẢNH GỐC, ảnh đầu vào là ảnh đã suy giảm.
     Khi đó F1/IoU/tương quan gradient/MSE/PSNR đo mức bảo toàn so với ảnh gốc.
     Chọn "Không – ảnh thật" khi không có ảnh gốc: chỉ số biên chỉ đo độ giống ảnh đầu vào (hạn chế, mục 8.2).
  3. Chọn phép biến đổi và kéo thanh trượt. Bảng dưới ảnh so sánh với dòng identity (mốc).
  4. Tab "Tổng quan đề 5": ảnh – histogram – biên của âm bản, log, γ = 0.5 / 1 / 2 trên một màn hình.
  5. Tab "Quét tham số": TN2 (γ), TN3 (gain log), TN7 (heatmap γ × c) trên ảnh hiện tại.
  6. Tab "Thí nghiệm hàng loạt": chạy toàn bộ TN1–TN8 trên thư mục ảnh, xuất
       tables/results.csv, tables/tham_so_de_xuat.csv, tables/tn6_do_nhay_k.csv,
       figures/*.png, hypotheses.md (số liệu H1–H5) và run_info.json (cấu hình + phiên bản thư viện).

CHỈ SỐ
  • F1/IoU top-k: trùng khớp vị trí biên (cùng mật độ k nên P = R = F1).
  • Tương quan gradient: Pearson giữa |∇ tham chiếu| và |∇ kết quả| – không cần ngưỡng.
  • MSE/PSNR: chỉ có khi có ảnh gốc; không tính cho âm bản (đảo mức xám là chủ ý).

Lưu ý: suy giảm "Đối chứng gamma 2.5" chỉ để minh hoạ thiên vị thiết kế (TN8), không dùng để kết luận H3–H4.
"""
