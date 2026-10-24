#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${AAOCA_PREPROCESS_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$ROOT/.venv-preprocess/bin/python" ]]; then
        PYTHON="$ROOT/.venv-preprocess/bin/python"
    elif [[ -x "${AAOCA_PREPROCESS_ENV:-/data5/zhougaowei/zhangruichen_workspace/aaoca_runtime/venv}/bin/python" ]]; then
        PYTHON="${AAOCA_PREPROCESS_ENV:-/data5/zhougaowei/zhangruichen_workspace/aaoca_runtime/venv}/bin/python"
    else
        PYTHON=python3
    fi
fi
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
exec "$PYTHON" "$ROOT/start_aaoca_server.py" "$@"
