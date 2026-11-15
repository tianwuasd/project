set -euo pipefail

dataset=${1:-imagechd}
fold=${2:-0}
action=${3:-prepare}
case "$dataset" in
  imagechd) label=ImageCHD ;;
  chd1282) label=CHD1282 ;;
  *) printf '数据集仅允许 imagechd 或 chd1282\n' >&2; exit 2 ;;
esac
case "$fold" in 0|1|2) ;; *) printf 'fold 仅允许 0、1、2\n' >&2; exit 2 ;; esac

root=${CHD_ROOT:-/data4/zhangruichen}
code=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py=${CHD_PYTHON:-"$root/env/chd-segmentation/bin/python"}
course="$root/course/chd-segmentation"
cfg=${CHD_CONFIG:-"$course/inputs/${dataset}_stunet_fold${fold}.yaml"}
name=${CHD_RUN_NAME:-"${label}-STUNet-20261115_fold${fold}"}
run="$course/runs/$name"
source_file="$course/prepared/${name}_source.json"
ready="$course/prepared/${name}_offline.json"
data=${CHD_PREPROCESSED_ROOT:-"$root/data/chd-segmentation/preprocessed"}
result="$root/result/chd-segmentation/$name"
common=(--config "$cfg" --run-dir "$run" --fold "$fold")
training=(--device cuda --gpus "${CHD_GPUS:-2}" --min-gpus "${CHD_GPUS:-2}"
  --batch "${CHD_BATCH:-4}" --epochs "${CHD_EPOCHS:-200}" --workers "${CHD_WORKERS:-auto}"
  --no-share-gpu --wait-seconds forever)
if [ -n "${CHD_GPU_IDS:-}" ]; then training+=(--gpu-ids "$CHD_GPU_IDS"); fi

prepare_data() {
  "$py" -u -B "$code/run.py" prepare "${common[@]}" --output "$source_file"
  "$py" -u -B "$code/run.py" preprocess "${common[@]}" --prepared "$source_file" --output "$ready" --root "$data"
}
run_tests() {
  "$py" -u -B "$code/run.py" test "${common[@]}" --set internal
  if [ "$dataset" = chd1282 ]; then "$py" -u -B "$code/run.py" test "${common[@]}" --set external; fi
}
preprocess_data() {
  "$py" -u -B "$code/run.py" preprocess "${common[@]}" --prepared "$source_file" --output "$ready" --root "$data"
}
case "$action" in
  prepare) prepare_data ;;
  preprocess) preprocess_data ;;
  check) "$py" -B "$code/run.py" check "${common[@]}" --stage preprocess --prepared "$source_file" ;;
  train) "$py" -u -B "$code/run.py" train "${common[@]}" --prepared "$ready" "${training[@]}" ;;
  resume) "$py" -u -B "$code/run.py" train "${common[@]}" --prepared "$ready" "${training[@]}" --resume ;;
  predict) "$py" -u -B "$code/run.py" predict "${common[@]}" --output "$result/predictions" ;;
  evaluate) "$py" -u -B "$code/run.py" evaluate "${common[@]}" --prepared "$ready" --predictions "$result/predictions" --output "$result/metrics.json" ;;
  test) run_tests ;;
  info) "$py" -B "$code/run.py" data-info --prepared "$ready" ;;
  all)
    prepare_data
    "$py" -u -B "$code/run.py" train "${common[@]}" --prepared "$ready" "${training[@]}"
    run_tests
    ;;
  *) printf '阶段允许 prepare/preprocess/check/train/resume/predict/evaluate/test/info/all\n' >&2; exit 2 ;;
esac
