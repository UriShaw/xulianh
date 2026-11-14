"""python -m imgtx  → mở giao diện;  python -m imgtx --config ...  → chạy dòng lệnh."""
import sys

if len(sys.argv) > 1:
    from .controllers.cli_controller import main
    raise SystemExit(main())
from .main import main

main()
