set -euo pipefail

root=${CHD_ROOT:-/data4/zhangruichen}
code=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py=${CHD_PYTHON:-"$root/env/chd-segmentation/bin/python"}
export PYTHONDONTWRITEBYTECODE=1
action=install
if [ "${1:-}" = check ]; then action=check; shift; fi
"$py" -u -B -c 'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("chd_segmentation.delivery",run_name="__main__")' \
  "$code/src" "$action" --root "$root" --code "$code" "$@"
