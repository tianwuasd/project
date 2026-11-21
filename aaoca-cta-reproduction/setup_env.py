from importlib.metadata import version
import json
import sys

import nibabel
import numpy
import psutil
import scipy
import SimpleITK


if __name__ == "__main__":
    print(json.dumps({"state": "passed", "python": sys.executable,
                      "packages": {name: version(name) for name in ("numpy", "scipy", "nibabel", "SimpleITK", "psutil")},
                      "scope": "定位裁剪依赖导入；不证明作者分类训练或CUDA可用"}, ensure_ascii=False))
