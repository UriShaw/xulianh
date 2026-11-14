"""Sinh logo ImgTX: assets/logo.png (512) và assets/imgtx.ico (16–256)."""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

S = 1024
# nền gradient chéo xanh đậm -> ngọc
y, x = np.mgrid[0:S, 0:S] / S
t = np.clip((x + y) / 2, 0, 1)[..., None]
c1, c2 = np.array([24, 44, 110]), np.array([32, 178, 160])
bg = (c1 * (1 - t) + c2 * t).astype(np.uint8)
img = Image.fromarray(np.dstack([bg, np.full((S, S), 255, np.uint8)]), "RGBA")
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=220, fill=255)
img.putalpha(mask)

d = ImageDraw.Draw(img)
m0, m1 = 210, S - 210                      # khung đồ thị
# lưới mờ
for k in range(1, 4):
    p = m0 + (m1 - m0) * k / 4
    d.line([(p, m0), (p, m1)], fill=(255, 255, 255, 40), width=6)
    d.line([(m0, p), (m1, p)], fill=(255, 255, 255, 40), width=6)
# trục
d.line([(m0, m1), (m1 + 30, m1)], fill=(255, 255, 255, 230), width=22)
d.line([(m0, m1), (m0, m0 - 30)], fill=(255, 255, 255, 230), width=22)
# đường identity nét mảnh
d.line([(m0, m1), (m1, m0)], fill=(255, 255, 255, 110), width=12)
# đường gamma 0.4 (dày, trắng) và gamma 2.5 (cam)
def curve(g):
    r = np.linspace(0, 1, 3000)
    return [(m0 + (m1 - m0) * a, m1 - (m1 - m0) * a ** g) for a in r]
def stroke(pts, w, col):
    for (px, py) in pts:
        d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2], fill=col)
stroke(curve(2.5), 34, (255, 176, 74, 255))
stroke(curve(0.4), 44, (255, 255, 255, 255))
# chấm đầu mút
for (px, py), col in (((m1, m0), (255, 255, 255, 255)), ((m0, m1), (255, 255, 255, 255))):
    d.ellipse([px - 34, py - 34, px + 34, py + 34], fill=col)

img.resize((512, 512), Image.LANCZOS).save("assets/logo.png")
img.resize((256, 256), Image.LANCZOS).save("assets/imgtx.ico",
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("ok")
