#!/usr/bin/env bash
# 标准库任务面板：查询不安装环境、不启动训练、不要求GPU。
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -B -X utf8 "$SCRIPT_DIR/05_src/task_panel.py" "$@"
