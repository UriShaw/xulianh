"""Điểm vào dòng lệnh (giữ đúng lệnh trong tài liệu): python -m imgtx.cli --config configs/default.yaml

Logic nằm ở controllers/cli_controller.py.
"""
from .controllers.cli_controller import main

if __name__ == "__main__":
    raise SystemExit(main())
