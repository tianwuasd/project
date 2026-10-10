#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Select exactly one diagnosis operation; default is prediction with explanations.
exec bash "$SCRIPT_DIR/start_server.sh" --task diagnose "$@"
