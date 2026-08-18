@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo Casting Defect Inspector - Flask Server
echo ==================================================

if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment is missing.
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)

if not exist "casting_defect_model.keras" (
    echo WARNING: casting_defect_model.keras is not in this folder.
    echo The website will open, but predictions will remain disabled.
    echo.
)

echo Opening http://127.0.0.1:5000
start "" "http://127.0.0.1:5000"
.venv\Scripts\python.exe app.py
pause
