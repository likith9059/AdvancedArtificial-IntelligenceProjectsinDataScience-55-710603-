@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo Casting Defect Inspector - Windows Setup
echo ==================================================

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher was not found.
    echo Install Python 3.11 from python.org and enable Add Python to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto :failed
)

echo Updating pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo Installing Flask and TensorFlow dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Setup completed successfully.
echo Copy casting_defect_model.keras into this folder before running the app.
echo Then double-click run_windows.bat.
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the error above.
pause
exit /b 1
