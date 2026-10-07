@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" verify_project.py
if errorlevel 1 exit /b 1
echo Setup and tests passed. Run start_demo.cmd to open the application.
pause
