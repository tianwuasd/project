#!/usr/bin/env bash
# 只调用系统 Python 的标准库向导；不要求事先有 Conda。
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -B -X utf8 "$SCRIPT_DIR/05_src/task_panel.py" "$@"
