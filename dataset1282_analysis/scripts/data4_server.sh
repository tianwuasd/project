set -euo pipefail
B=/data4/zhangruichen
C=$B/code/dataset1282_analysis
PY=$B/env/dataset1282_analysis/bin/python
H=${CHD_CHT_CODE:-$B/code/chd-segmentation}
D=${CHD_PREPROCESSED:-$B/data/chd-segmentation/preprocessed/CHD1282-all-20261125}
M=$B/result/chd1282/models
P=${CHD_PREDICTION_OUTPUT:-$B/result/chd1282/prediction-sixway}
L=$B/course/chd1282
R=$B/back/dataset1282_preprocessing
V=$B/data/chd-segmentation/preprocessed/CHD1282-XuImport-20261125-v4/fold_0
cd "$C"
test -x "$PY"
test -s "$D/prepared.json"
mkdir -p "$L" "$B/result/chd1282"
case "${1:-}" in
accept-b2nd)
  N=${CHD_COMPRESSED_OUTPUT:-$D-b2nd}
  exec "$PY" -B scripts/accept_b2nd.py --source "$D/prepared.json" --compressed "$N/prepared.json" \
    --cht-code "$H" --output "$L/xu-b2nd-read.json"
  ;;
delete-npy)
  N=${CHD_COMPRESSED_OUTPUT:-$D-b2nd}
  test -s "$N/compression-complete.json"
  exec "$PY" -B -u scripts/compress_preprocessed.py --preprocessed "$D/prepared.json" \
    --cht-code "$H" --output "$N" --resume --delete-npy-after-verify \
    --acceptance "$L/xu-b2nd-read.json" "$L/cht-b2nd-read.json"
  ;;
compress)
  N=${CHD_COMPRESSED_OUTPUT:-$D-b2nd}
  exec "$PY" -B -u scripts/compress_preprocessed.py --preprocessed "$D/prepared.json" \
    --cht-code "$H" --output "$N" "${@:2}"
  ;;
check)
  "$PY" -B scripts/xu_data4.py check --preprocessed "$D/prepared.json" --cht-code "$H"
  ;;
models)
  "$PY" -B scripts/xu_data4.py models --preprocessed "$D/prepared.json" --cht-code "$H" \
    --legacy-prepared "$R/work/prepared/zmic-source-v030/hu-marginal" \
    --prediction "$R/experiments/xu-source-test-1794903218" "$R/experiments/xu-target-test-1794998445" \
    --training "$R/experiments/xu-train-1794048634" --output "$M"
  ;;
short)
  S=$B/result/chd1282/.short-one
  test ! -e "$S"
  exec "$PY" -B -u scripts/xu_data4.py short --preprocessed "$D/prepared.json" --cht-code "$H" \
    --models "$M" --output "$S" --threads 4 --receipt "$L/acceptance.json" "${@:2}"
  ;;
predict)
  O=$P
  "$PY" -B -u scripts/xu_data4.py predict --preprocessed "$D/prepared.json" --cht-code "$H" \
    --models "$M" --output "$O" --threads 4 "${@:2}"
  ;;
evaluate)
  "$PY" -B -u scripts/xu_predict.py evaluate --output "$P" "${@:2}"
  ;;
preview|clean)
  flags=()
  if [ "$1" = clean ]; then flags+=(--apply); fi
  "$PY" -B -u scripts/retire_chd1282.py --preprocessed "$D/prepared.json" --cht-code "$H" \
    --models "$M" --receipt "$L/acceptance.json" --old-export "$V" "${flags[@]}"
  ;;
*)
  echo '用法：bash scripts/data4_server.sh compress|accept-b2nd|delete-npy|check|models|short|predict|evaluate|preview|clean' >&2
  exit 2
  ;;
esac
