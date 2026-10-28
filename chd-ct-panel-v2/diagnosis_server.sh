#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# V2 diagnosis graph requires an explicit completed predict-all source task.
exec bash "$SCRIPT_DIR/start_server.sh" --stage diagnosis "$@"
