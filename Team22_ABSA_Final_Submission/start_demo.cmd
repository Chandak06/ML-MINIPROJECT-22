@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run setup.cmd to create the project environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --browser.gatherUsageStats false
