#!/usr/bin/env bash
# Linux/Ubuntu 一键入口；路径全部加引号，可包含空格。不要用 sudo 运行。
set -Eeuo pipefail
trap 'printf "\n启动失败（第 %s 行）。请查看上方错误；没有自动重试或删除数据。\n" "$LINENO" >&2' ERR
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
内部计算环境安装入口。用户请运行 bash start_server.sh 打开中文任务面板。
仅安装固定依赖：bash bootstrap_server.sh --install-only
不在此入口自动选择数据或GPU。运行环境由面板的runtime配置指定。
HELP
  exit 0
fi
[[ "$(uname -s)" == "Linux" ]] || { echo "此入口用于 Linux/Ubuntu；Windows 上请勿直接运行。" >&2; exit 2; }
# 安装和CPU预处理不要求显卡；显卡在实际GPU步骤入口检查。

# 使用 Conda 可执行文件，不要求先执行 conda activate 或修改 shell 配置。
CONDA_BIN="${CONDA_EXE:-}"
if [[ ! -x "$CONDA_BIN" ]]; then
  CONDA_BIN="$(type -P conda || true)"
fi
if [[ ! -x "$CONDA_BIN" ]]; then
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
    if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
  done
fi
[[ -x "$CONDA_BIN" ]] || { echo "找不到 Conda。请先安装 Miniconda，或设置 CONDA_EXE=/实际路径/bin/conda。" >&2; exit 2; }
ENV_PREFIX="${WHOLE_HEART_ENV_PREFIX:-$SCRIPT_DIR/.server_env}"

# 防止两次启动同时修改环境；训练阶段由 Python 工作区锁保护。
command -v flock >/dev/null || { echo "缺少 flock，请安装 util-linux。" >&2; exit 2; }
mkdir -p "$(dirname -- "$ENV_PREFIX")"
exec 9>"${ENV_PREFIX}.setup.lock"
flock -n 9 || { echo "正在安装环境，请等待安装结束。" >&2; exit 2; }

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "首次启动：创建独立 Conda 环境，下载依赖可能需要一些时间。"
  "$CONDA_BIN" env create --prefix "$ENV_PREFIX" --file "$SCRIPT_DIR/06_configs/environment_server.yml" --yes
fi
if [[ ! -f "$ENV_PREFIX/.baseline_ready" ]]; then
  echo "安装固定的 GPU 训练依赖（不会改动其他环境）。"
  "$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  "$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -m pip install nnunetv2==2.8.1 --index-url https://pypi.org/simple -c "$SCRIPT_DIR/06_configs/training_constraints.txt"
  "$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -m pip check
  touch "$ENV_PREFIX/.baseline_ready"
fi
flock -u 9
exec 9>&-
export WHOLE_HEART_CONDA_EXE="$CONDA_BIN"
export WHOLE_HEART_ENV_PREFIX="$ENV_PREFIX"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
if [[ "${1:-}" == "--install-only" ]]; then
  echo "独立训练环境准备完成；尚未申请GPU。"
  exit 0
fi
# 原样传递参数，绝不把数据路径拼成 shell 命令或使用 eval。
"$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -B -X utf8 "$SCRIPT_DIR/05_src/server_launch.py" "$@"
