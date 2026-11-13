import argparse
from importlib.metadata import version
import json
import platform
import sys


def check():
    import monai
    import nibabel
    import nnunetv2
    import numpy
    import scipy
    import torch
    import yaml
    from filelock import FileLock
    from monai.inferers import sliding_window_inference
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    if version("monai") != "1.5.2" or version("nnunetv2") != "2.8.1":
        raise ValueError("需要 MONAI 1.5.2 和 nnU-Net 2.8.1")
    if not callable(sliding_window_inference) or not callable(DefaultPreprocessor.run_case_npy):
        raise ValueError("所需数据处理或推断 API 不可用")
    return {"state": "passed", "python": sys.executable, "platform": platform.platform(),
            "packages": {name: version(name) for name in
                         ("torch", "monai", "nnunetv2", "numpy", "scipy", "nibabel", "PyYAML", "psutil", "filelock")},
            "scope": "依赖与 API 检查；不代表模型运行或 CUDA 验收"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查先心分割处理的独立运行环境")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(check(), ensure_ascii=False, indent=2))
