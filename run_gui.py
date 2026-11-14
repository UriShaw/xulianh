"""Điểm vào của ứng dụng desktop (dùng cho PyInstaller):  python run_gui.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from imgtx.main import main  # noqa: E402

if __name__ == "__main__":
    main()
