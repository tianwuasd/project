@echo off
setlocal
if defined CHD_V2_CONTROL_PYTHON (
  "%CHD_V2_CONTROL_PYTHON%" -B -X utf8 "%~dp0start_server_v2.py" %*
) else (
  python -B -X utf8 "%~dp0start_server_v2.py" %*
)
