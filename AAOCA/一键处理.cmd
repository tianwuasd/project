@echo off
setlocal
cd /d "%~dp0"
set "AAOCA_PY=%AAOCA_PYTHON%"
if not defined AAOCA_PY set "AAOCA_PY=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%AAOCA_PY%" set "AAOCA_PY=%~dp0..\..\..\project\AAOCA\.venv\Scripts\pythonw.exe"
if not exist "%AAOCA_PY%" (
  echo Python environment was not found. Install .venv or set AAOCA_PYTHON.
  pause
  exit /b 1
)
start "AAOCA" "%AAOCA_PY%" "%~dp0pipeline_gui.py"
exit /b 0
