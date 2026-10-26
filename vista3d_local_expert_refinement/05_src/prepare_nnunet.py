"""仅用来源训练子集生成指纹/网络规划，再处理来源训练与开发病例。

目标中心不出现在 raw 训练目录；开发集不参与强度统计或间距规划。
用 nnU-Net 官方 API，不手写近似的重采样/归一化替代它。
"""
import argparse
import os
import time
from pathlib import Path
from data_pipeline import ROOT, FOLDS, load_json, save_json


def configure_paths():
    for key, folder in [("nnUNet_raw", "04_data/derived/nnUNet_raw"), ("nnUNet_preprocessed", "04_data/derived/nnUNet_preprocessed"), ("nnUNet_results", "07_experiments/nnUNet_results")]:
        os.environ[key] = str(ROOT / folder)
    os.environ["nnUNet_compile"] = "false"
    os.environ["nnUNet_wandb_enabled"] = "false"
    os.environ["nnUNet_n_proc_DA"] = "0"
    os.environ.setdefault('OMP_NUM_THREADS',os.environ.get('WHOLE_HEART_CPU_THREADS','1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fold', choices=list(FOLDS))
    parser.add_argument('--gpu-memory', type=float, default=20)
    args = parser.parse_args()
    from server_prepare import prepare
    prepare(args.fold, args.gpu_memory)


if __name__ == '__main__':
    main()
