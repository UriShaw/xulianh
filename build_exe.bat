@echo off
REM Dong goi ImgTX.exe (chay tren Windows co Python 3.10-3.12)
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-build.txt
python -m pytest -q -m "not slow" || (echo Test that bai & pause & exit /b 1)
pyinstaller --noconfirm --clean --onefile --windowed --name ImgTX ^
  --icon assets\imgtx.ico --version-file version_info.txt ^
  --add-data "assets;assets" --add-data "data\raw;data\raw" --add-data "data\detail;data\detail" --add-data "data\medical;data\medical" --add-data "data\real;data\real" ^
  --paths src --exclude-module pytest --exclude-module IPython ^
  run_gui.py
echo.
echo Xong: dist\ImgTX.exe
pause
