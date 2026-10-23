#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${AAOCA_PYTHON:-python3}" "$root/start_segmentation.py" "$@"
