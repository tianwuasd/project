#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_PYTHON="${VISTA_CONTROL_PYTHON:-python3}"
command -v "$CONTROL_PYTHON" >/dev/null || { echo "未找到 Python；请提供 Python 3.11+，或设置 VISTA_CONTROL_PYTHON。" >&2; exit 2; }
"$CONTROL_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 2)' || { echo "面板需要 Python 3.11+；可用 VISTA_CONTROL_PYTHON=python3.11 bash start_server.sh，不需要先安装训练环境。" >&2; exit 2; }
exec "$CONTROL_PYTHON" -B -X utf8 "$SCRIPT_DIR/05_src/task_panel.py" "$@"
