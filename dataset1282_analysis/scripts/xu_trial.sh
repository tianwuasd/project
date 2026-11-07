#!/usr/bin/env bash
# Existing research Python, completed prepared cache, NEW run directory,
# authorized GPU pool (auto/all-visible/cuda:...), and pause epoch (1 or 2).
set -euo pipefail
umask 077
if [ "$#" -ne 5 ]; then
  echo 'Usage: bash scripts/xu_trial.sh PYTHON PREPARED NEW_RUN GPU_POOL PAUSE_EPOCH' >&2
  exit 2
fi
# Preserve a venv's final python symlink: resolving it would select the base
# interpreter and lose that environment's installed research dependencies.
PY="$(cd -- "$(dirname -- "$1")" && pwd -P)/$(basename -- "$1")"
D=$(realpath "$2")
RUN=$(realpath -m "$3")
GPU_POOL=$4
PAUSE_EPOCHS=$5
case "$PAUSE_EPOCHS" in 1|2) ;; *) echo 'Pause epoch must be 1 or 2' >&2; exit 2;; esac
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
case "$RUN/" in "$D/"*|"$CODE/"*) echo 'Run directory must be outside prepared data and code' >&2; exit 2;; esac
cd "$CODE"
test -x "$PY"
test -s "$D/complete.json"
"$PY" -B -c 'import torch,numpy,scipy,skimage,nibabel,blosc2,psutil; assert torch.cuda.is_available() and torch.cuda.device_count() >= 1'
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -B -c 'from pathlib import Path; from dataset1282_analysis.maintenance import verify_release; print("verified files",len(verify_release(Path.cwd())["files"]))'
mkdir "$RUN"
O="$RUN/training"
printf 'CODE=%q\nPY=%q\nD=%q\nRUN=%q\nO=%q\nGPU_POOL=%q\nPAUSE_EPOCHS=%q\n' "$CODE" "$PY" "$D" "$RUN" "$O" "$GPU_POOL" "$PAUSE_EPOCHS" > "$RUN/settings.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo 'Checking prepared cohort and checkpoint recovery'
"$PY" -B scripts/xu_seg.py inspect --prepared "$D"
PYTHONPATH="src:tests${PYTHONPATH:+:$PYTHONPATH}" "$PY" -B -m unittest test_xu_training test_xu_epoch_pause -v
for stage in inference 3d64 3d128; do
  echo "GPU short check: $stage"
  "$PY" -B -u scripts/xu_fleet_probe.py --devices "$GPU_POOL" --stage "$stage" --threads 4 --output "$RUN/check-$stage"
done
echo "Short checks passed; starting both paper localizers through epoch $PAUSE_EPOCHS"
"$PY" -B -u scripts/xu_pipeline.py run --prepared "$D" --output "$O" \
  --profile paper --supervision marginal --devices "$GPU_POOL" --threads 4 \
  --prefetch --wait-seconds 60 --pause-after-epoch "$PAUSE_EPOCHS"
echo 'Trial command ended; inspect training/status.json for the recorded outcome'
