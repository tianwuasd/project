@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0start.py" %*
    goto finished
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0start.py" %*
    goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0start.py" %*
    goto finished
)
echo Python 3.11+ not found. Install Python and enable "Add Python to PATH".
set "result=1"
goto end
:finished
set "result=%errorlevel%"
:end
if "%~1"=="" pause
exit /b %result%
