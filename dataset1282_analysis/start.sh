#!/usr/bin/env sh
set -eu
exec python3 -B "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/run.py" "$@"
