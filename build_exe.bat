@echo off
REM Dong goi ImgTX (chay tren Windows co Python 3.10-3.12)
REM   dist\ImgTX\ImgTX.exe          ban thu muc - mo nhanh (khong giai nen moi lan chay)
REM   dist\ImgTX_Setup_x.y.z.exe    bo cai dat (neu da cai NSIS: https://nsis.sourceforge.io)
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-build.txt
python -m pytest -q -m "not slow" || (echo Test that bai & pause & exit /b 1)
pyinstaller --noconfirm --clean --onedir --windowed --name ImgTX ^
  --icon assets\imgtx.ico --version-file version_info.txt ^
  --add-data "assets;assets" --add-data "data\raw;data\raw" --add-data "data\detail;data\detail" ^
  --add-data "data\medical;data\medical" --add-data "data\real;data\real" ^
  --paths src --exclude-module pytest --exclude-module IPython ^
  --exclude-module pandas.tests --exclude-module numpy.tests --exclude-module matplotlib.tests ^
  run_gui.py
where makensis >nul 2>nul && makensis installer\imgtx.nsi
echo.
echo Xong: dist\ImgTX\ImgTX.exe  (va dist\ImgTX_Setup_*.exe neu co NSIS)
pause
