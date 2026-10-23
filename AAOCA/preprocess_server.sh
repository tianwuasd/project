#!/usr/bin/env bash
# Explicit foreground execution. For SSH-independent runs see PREPROCESSING.md.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${AAOCA_PREPROCESS_PYTHON:-python3}"
# Conservative per-process library threads; case parallelism is --workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
exec "$python_bin" -u "$script_dir/preprocess_aaoca.py" "$@"
