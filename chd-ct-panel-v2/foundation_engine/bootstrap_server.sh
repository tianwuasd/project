#!/usr/bin/env bash
# Linux/Ubuntu 一键入口；路径全部加引号，可包含空格。不要用 sudo 运行。
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_PYTHON="${WHOLE_HEART_CONTROL_PYTHON:-python3}"
progress() { "$CONTROL_PYTHON" -B -X utf8 "$SCRIPT_DIR/05_src/runtime_setup.py" "$@"; }
download_run() { "$CONTROL_PYTHON" -B -X utf8 "$SCRIPT_DIR/05_src/runtime_setup.py" --download "$@"; }
trap 'progress "" "环境安装失败（第 $LINENO 行），请查看任务总日志。" --status failed || true' ERR

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
ENV_PREFIX="${WHOLE_HEART_ENV_PREFIX:-$HOME/.local/share/imagechd_stunet_runtime/envs/imagechd_stunet_py311}"
export WHOLE_HEART_ENV_PREFIX="$ENV_PREFIX"

# 防止两次启动同时修改环境；训练阶段由 Python 工作区锁保护。
command -v flock >/dev/null || { echo "缺少 flock，请安装 util-linux。" >&2; exit 2; }
mkdir -p "$(dirname -- "$ENV_PREFIX")"
exec 9>"${ENV_PREFIX}.setup.lock"
flock -n 9 || { echo "正在安装环境，请等待安装结束。" >&2; exit 2; }

CREATED=0
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  if [[ -d "$ENV_PREFIX" ]]; then
    echo "环境目录不完整：$ENV_PREFIX；已保留现场，请核查或在面板选择新的 runtime。" >&2
    exit 2
  fi
  progress python_env "创建独立 Python 3.11 环境；此阶段可能等待依赖解析和下载。"
  download_run "$CONDA_BIN" env create --prefix "$ENV_PREFIX" --file "$SCRIPT_DIR/06_configs/environment_server.yml" --yes
  CREATED=1
fi
FINGERPRINT="$( { cat "$SCRIPT_DIR/06_configs/environment_server.yml" "$SCRIPT_DIR/06_configs/training_constraints.txt"; printf '%s' 'torch2.8.0+cu128 torchvision0.23.0+cu128 nnunetv2=2.8.1'; } | sha256sum | cut -d' ' -f1)"
READY=0
if [[ -f "$ENV_PREFIX/.baseline_ready" ]] && [[ "$(cat "$ENV_PREFIX/.baseline_ready")" == "$FINGERPRINT" ]]; then
  progress verify "已有环境仍需实际导入与资源采集检查；安装凭证不能代替当前检查。"
  if "$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -B "$SCRIPT_DIR/05_src/runtime_setup.py" --check-dependencies; then
    READY=1
  else
    progress python_env "检测到缺失或损坏依赖，修复本项目独立环境；详情见上述检查输出。"
  fi
fi
if [[ "$READY" == 0 ]]; then
  if [[ "$CREATED" == 0 ]]; then
    progress python_env "依赖配置或旧凭证已变化；更新本项目独立 Conda 环境。"
    download_run "$CONDA_BIN" env update --prefix "$ENV_PREFIX" --file "$SCRIPT_DIR/06_configs/environment_server.yml"
  fi
  progress pytorch "安装固定 PyTorch / CUDA 运行库；下载明细见总日志，不需要现在分配显卡。"
  download_run "$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  progress nnunet "安装 nnU-Net 2.8.1 及训练约束依赖。"
  download_run "$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -m pip install nnunetv2==2.8.1 threadpoolctl==3.6.0 --index-url https://pypi.org/simple -c "$SCRIPT_DIR/06_configs/training_constraints.txt"
fi
progress verify "核验固定版本、实际导入、psutil资源采集和pip依赖一致性；CPU检查不申请GPU。"
"$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -B "$SCRIPT_DIR/05_src/runtime_setup.py" --check-dependencies
"$CONDA_BIN" run --no-capture-output --prefix "$ENV_PREFIX" python -m pip check
printf '%s\n' "$FINGERPRINT" > "$ENV_PREFIX/.baseline_ready.pending"
mv -- "$ENV_PREFIX/.baseline_ready.pending" "$ENV_PREFIX/.baseline_ready"
progress complete "独立训练环境安装与依赖检查通过；尚未申请 GPU。" --status completed
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
