set -euo pipefail

dataset=${1:-imagechd}
action=${2:-prepare}
case "$dataset" in imagechd) label=ImageCHD ;; chd1282) label=CHD1282 ;; *) exit 2 ;; esac
root=${CHD_ROOT:-/data4/zhangruichen}
code=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py=${CHD_PYTHON:-"$root/env/chd-segmentation/bin/python"}
course="$root/course/chd-segmentation"
configs="$course/configs/${label}-MedSAM2/20261115"
model=${MEDSAM2_CONFIG:-"$configs/model.yaml"}
prepare_config=${MEDSAM2_PREPARE_CONFIG:-"$configs/prepare.yaml"}
source_file="$course/prepared/${label}-MedSAM2/20261115/source.json"
ready="$course/prepared/${label}-MedSAM2/20261115/offline.json"
data=${CHD_PREPROCESSED_ROOT:-"$root/data/chd-segmentation/preprocessed"}
case "$action" in
  prepare)
    "$py" -u -B "$code/run.py" prepare --config "$prepare_config" \
      --run-dir "$course/prepare/${label}-MedSAM2/20261115" --output "$source_file"
    "$py" -u -B "$code/run.py" models preprocess --config "$model" --prepare-config "$prepare_config" \
      --prepared "$source_file" --root "$data" --output "$ready"
    ;;
  train|resume)
    resume=()
    if [ "$action" = resume ]; then resume+=(--resume); fi
    "$py" -u -B "$code/run.py" models train --config "$model" --prepared "$ready" \
      --stage sequence --device cuda --gpus "${CHD_GPUS:-2}" --batch "${MEDSAM2_BATCH:-2}" \
      --workers auto --no-share-gpu "${resume[@]}"
    ;;
  label-box-internal|label-box-external)
    test_set=${action#label-box-}
    official=()
    if [ "${MEDSAM2_OFFICIAL_WEIGHTS:-0}" = 1 ]; then official+=(--official-weights); fi
    "$py" -u -B "$code/run.py" models predict --config "$model" --prepared "$ready" \
      --checkpoint "${MEDSAM2_CHECKPOINT:?请填写官方权重或本项目检查点路径}" \
      --prompt-mode label-box --test-set "$test_set" \
      --output "$root/result/chd-segmentation/${label}-MedSAM2/20261115-offline/$test_set" --no-share-gpu "${official[@]}"
    ;;
  info) "$py" -B "$code/run.py" data-info --prepared "$ready" ;;
  *) printf '阶段允许prepare/train/resume/label-box-internal/label-box-external/info\n' >&2; exit 2 ;;
esac
