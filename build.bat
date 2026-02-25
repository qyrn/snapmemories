@echo off
REM ────────────────────────────────────────────────────────────
REM  SnapMemories — Build script (Windows)
REM  Requires: pip install -r requirements.txt
REM ────────────────────────────────────────────────────────────

echo [1/3] Cleaning previous builds...
if exist build rmdir /s /q build
if exist SnapMemories.spec del SnapMemories.spec
if exist SnapMemories.exe del SnapMemories.exe

echo [2/3] Building executable...
pyinstaller ^
  --onefile ^
  --noconsole ^
  --name SnapMemories ^
  --distpath . ^
  --icon "static\favicon.ico" ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --hidden-import=flask ^
  --hidden-import=requests ^
  --hidden-import=PIL ^
  --hidden-import=piexif ^
  app.py

echo [3/3] Done.
if exist SnapMemories.exe (
  echo.
  echo  Executable ready: SnapMemories.exe
  echo  Double-click it to launch the app.
) else (
  echo.
  echo  ERROR: build failed. Check the logs above.
)
pause
