# 独立环境

`setup_env.py`可从基础Python调用，也可通过`dataset1282_analysis.environment`中的`check_environment`和`install_environment`作为函数复用。标准库负责环境创建和命令编排，真正的数据依赖只在目标解释器中检查。

- `--plan`仅显示环境路径、锁摘要与安装步骤。
- `--check`不创建/修复环境，查询全部锁定包版本并用临时合成文件检查NIfTI读写、Excel、连通域、资源采集和无GUI绘图。
- `--install`只在配置的专属目录创建venv，再安装锁定二进制依赖。已有非venv目录拒绝覆盖；已有专属venv可以再次安装同一锁文件。

安装日志在work_root/environment_logs；验证结果在environment-report.json。已有研究、Slicer或系统环境不作为安装目标。失败保留日志，不自动清理或切换其他环境。API检查使用临时文件，不写入原始数据。

本机验证Python3.11。Linux路径使用`bin/python`，Windows使用`Scripts/python.exe`；需要可创建venv的Python基础安装。依赖锁同时固定直接与传递依赖，安装只接受二进制wheel；目标平台缺wheel时明确失败，不偷偷改用源码编译或换版本。

安装网络只作用子进程。默认PyPI；可显式传入`--index-url`和`--proxy`。本地中国网络访问国外服务可用`--proxy http://127.0.0.1:7897`，服务器应按其实际路由配置。没有代理监听时需先修复该路由；程序不改全局Git、pip或系统代理。

当前范围不含GPU依赖或驱动安装。Windows实测不等于目标Linux设备实测；部署后应依次运行环境check、synthetic smoke及真实capacity smoke。
