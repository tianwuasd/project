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
    os.environ["nnUNet_n_proc_DA"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", choices=list(FOLDS))
    parser.add_argument("--resume-after-fix", action="store_true", help="仅在保留失败日志、修正具体问题后使用已有指纹继续")
    args = parser.parse_args()
    configure_paths()
    # nnU-Net 在导入时读取环境变量，所以必须先设置路径。
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset
    split = load_json(ROOT / "04_data/manifests/splits_v1.json")["folds"][args.fold]
    name = f"Dataset{split['dataset_id']:03d}_{args.fold}"
    destination = Path(os.environ["nnUNet_preprocessed"]) / name
    if destination.exists() and not args.resume_after_fix:
        raise FileExistsError("预处理目录已存在；检查先前运行状态，禁止静默覆盖或重试")
    start = time.perf_counter()
    if args.resume_after_fix:
        provenance = load_json(destination / "fingerprint_provenance.json")
        assert provenance["fit_ids"] == split["train"]
        assert provenance["excluded_val"] == split["val"] and provenance["excluded_target"] == split["target"]
        if (ROOT / f"09_reports/{args.fold}_preparation.json").exists():
            raise FileExistsError("已完成准备，不能重复运行")
    else:
        extractor = DatasetFingerprintExtractor(split["dataset_id"], num_processes=1, verbose=False)
        extractor.dataset = {i: extractor.dataset[i] for i in split["train"]}
        extractor.dataset_json = dict(extractor.dataset_json, numTraining=len(split["train"]))
        extractor.run(overwrite_existing=False)
        save_json(destination / "fingerprint_provenance.json", {"fit_ids": sorted(extractor.dataset), "excluded_val": split["val"], "excluded_target": split["target"], "foreground_samples": extractor.num_foreground_voxels_for_intensitystats})
    planner = ExperimentPlanner(split["dataset_id"], gpu_memory_target_in_gb=5, plans_name="Project1Plans")
    planner.dataset = {i: planner.dataset[i] for i in split["train"]}
    planner.dataset_json = dict(planner.dataset_json, numTraining=len(split["train"]))
    plans = planner.plan_experiment()
    # 原始 dataset.json 记录来源总数，规划过程的训练分母另在 provenance 中明确。
    save_json(destination / "splits_final.json", [{"train": split["train"], "val": split["val"]}])
    preprocess_dataset(split["dataset_id"], plans_identifier="Project1Plans", configurations=["3d_fullres"], num_processes=[1], verbose=False)
    save_json(ROOT / f"09_reports/{args.fold}_preparation.json", {"status": "passed", "seconds": time.perf_counter()-start, "source_train": len(split["train"]), "source_val": len(split["val"]), "target_used": False, "gpu_memory_target_gb": 5, "configuration": plans["configurations"]["3d_fullres"]})


if __name__ == "__main__":
    main()
