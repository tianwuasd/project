#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_PYTHON="${CHD_V2_CONTROL_PYTHON:-python3}"
"$CONTROL_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 2)' || {
  echo "CHD V2 requires Python 3.11+; set CHD_V2_CONTROL_PYTHON." >&2
  exit 2
}
exec "$CONTROL_PYTHON" -B -X utf8 "$SCRIPT_DIR/start_server_v2.py" "$@"
