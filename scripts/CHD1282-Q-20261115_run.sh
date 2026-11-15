set -euo pipefail

action=${1:-prepare}
root=${CHD_ROOT:-/data4/zhangruichen}
code=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py=${CHD_PYTHON:-"$root/env/chd-segmentation/bin/python"}
course="$root/course/chd-segmentation"
models=${Q_CONFIG_DIR:-"$course/configs/CHD1282-Q/20261115"}
prepare_config=${Q_PREPARE_CONFIG:-"$models/prepare.yaml"}
source_file="$course/prepared/CHD1282-Q/20261115/source.json"
ready="$course/prepared/CHD1282-Q/20261115/offline.json"
data=${CHD_PREPROCESSED_ROOT:-"$root/data/chd-segmentation/preprocessed"}

case "$action" in
  prepare)
    "$py" -u -B "$code/run.py" prepare --config "$prepare_config" \
      --run-dir "$course/prepare/CHD1282-Q/20261115" --output "$source_file"
    "$py" -u -B "$code/run.py" models preprocess --config "$models/base.yaml" \
      --prepare-config "$prepare_config" --prepared "$source_file" --root "$data" --output "$ready"
    ;;
  base|localizer|local)
    "$py" -u -B "$code/run.py" models train --config "$models/${action}.yaml" \
      --prepared "$ready" --stage "$action" --device cuda --gpus "${CHD_GPUS:-2}" --no-share-gpu
    ;;
  resume)
    stage=${2:?请指定base、localizer或local}
    case "$stage" in base|localizer|local) ;; *) exit 2 ;; esac
    "$py" -u -B "$code/run.py" models train --config "$models/${stage}.yaml" \
      --prepared "$ready" --stage "$stage" --device cuda --gpus "${CHD_GPUS:-2}" --no-share-gpu --resume
    ;;
  internal|external)
    checkpoint=${Q_CHECKPOINT:?请填写局部阶段best.pt的完整路径}
    "$py" -u -B "$code/run.py" models predict --config "$models/local.yaml" \
      --prepared "$ready" --checkpoint "$checkpoint" --test-set "$action" \
      --output "$root/result/chd-segmentation/CHD1282-Q/20261115-offline/$action" --no-share-gpu
    ;;
  info) "$py" -B "$code/run.py" data-info --prepared "$ready" ;;
  *) printf '阶段允许prepare/base/localizer/local/resume/internal/external/info\n' >&2; exit 2 ;;
esac
