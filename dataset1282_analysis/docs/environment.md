# 独立环境

`setup_env.py`可从基础Python调用，也可通过`dataset1282_analysis.environment`中的`check_environment`和`install_environment`作为函数复用。标准库负责环境创建和命令编排，真正的数据依赖只在目标解释器中检查。

- `--plan`仅显示环境路径、锁摘要与安装步骤。
- `--check`不创建/修复环境，查询全部锁定包版本并用临时合成文件检查NIfTI读写、Excel、连通域、资源采集和无GUI绘图。
- `--install`只在配置的专属目录创建venv，再安装锁定二进制依赖。已有非venv目录拒绝覆盖；已有专属venv可以再次安装同一锁文件。

安装日志在work_root/environment_logs；验证结果在environment-report.json。已有研究、Slicer或系统环境不作为安装目标。失败保留日志，不自动清理或切换其他环境。API检查使用临时文件，不写入原始数据。

本机验证Python3.11。Linux路径使用`bin/python`，Windows使用`Scripts/python.exe`；需要可创建venv的Python基础安装。依赖锁同时固定直接与传递依赖，安装只接受二进制wheel；目标平台缺wheel时明确失败，不偷偷改用源码编译或换版本。

安装网络只作用子进程。默认PyPI；可显式传入`--index-url`和`--proxy`。本地中国网络访问国外服务可用`--proxy http://127.0.0.1:7897`，服务器应按其实际路由配置。没有代理监听时需先修复该路由；程序不改全局Git、pip或系统代理。

上述 `setup_env.py` 只负责审计环境。Windows实测不等于目标Linux设备实测；部署后应依次运行环境check、synthetic smoke及真实capacity smoke。

## Xu GPU 训练环境

从项目目录用现有基础 Python 调用独立入口；无需先激活 Conda/venv：

```bash
python3 scripts/setup_gpu.py --check --runtime /path/to/dataset1282_runtime
python3 scripts/setup_gpu.py --install --runtime /path/to/dataset1282_runtime
source /path/to/dataset1282_runtime/training-env.sh
```

`--check` 只检查、不安装。`--install` 也先检查，按显式 `--reuse-python`、本项目 `training`、本项目 `research`、相邻 CHD 项目 `.venv_preprocess` 的顺序选取首个合格解释器；不修改被复用的环境。只有没有合格环境，且 NVIDIA 驱动/GPU 可见时，才在指定 runtime 的 `training` 子目录安装。已有无本安装器归属标记的目录拒绝覆盖。GPU 被隐藏、CUDA PyTorch 已安装但 GPU 不可用时停止并报告，不反复下载，也不安装系统驱动。

新建环境采用本机已经验证的 Python 3.11 / PyTorch 2.12.1+cu126 基线；基础解释器支持 3.11–3.13，可用 `--base-python` 指定。CUDA wheel 来源为 [PyTorch 官方历史版本安装页](https://pytorch.org/get-started/previous-versions/)。数据依赖固定在 `configs/training-requirements.txt`；传递依赖由 pip 解析，最终完整版本写入 `training-freeze.txt`，该文件不是跨平台锁。复用环境可采用其他通过导入/API/CUDA检查的版本，实际版本记入报告；其训练数值和容量仍需短测。

检查包含实际库导入、CPU反向传播、Blosc2数组、NIfTI、缩放、骨架/分水岭、绘图API及CUDA可见性；Linux多卡要求NCCL构建。没有在检查阶段加载患者数据、申请模型显存或启动训练。后续仍需通过 `scripts/xu_trial.sh` 中的真实尺寸GPU检查，环境通过不等于多卡训练验收。

安装模式写出 `training-environment.json`；完整安装输出追加到 `training-install.log`。只有检查成功才写 `training-env.sh`，其中保存实际 `PY` 路径，保留 Linux venv 的 Python 符号链接。失败会撤去旧的成功入口。网络默认直连；需要时显式加 `--proxy http://HOST:PORT` 重试，代理只作用于本次安装，不改全局设置。环境安装与原始数据、已完成预处理及现有审计/研究环境分别保存。
