@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 未找到运行环境，请按照 一键处理说明.md 安装依赖。
  pause
  exit /b 1
)
if not "%~1"=="" goto commandline
".venv\Scripts\python.exe" -X utf8 -c "import pipeline_gui" >nul
if errorlevel 1 (
  echo 窗口启动检查失败，请检查运行环境或查看一键处理说明。
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -X utf8 "pipeline_gui.py"
exit /b 0
:commandline
".venv\Scripts\python.exe" -X utf8 "run_pipeline.py" %*
set "pipelineExit=%errorlevel%"
if "%pipelineExit%"=="2" echo 已生成结果，但有需要查看的问题，请打开输出目录中的 查看结果.html 和 问题清单.csv。
pause
exit /b %pipelineExit%
