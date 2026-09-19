"""服务器可继续的数据准备，逐病例保存完成标记，保留已完成预处理。"""
import argparse
import importlib.metadata
from pathlib import Path
import shutil
import time
from data_pipeline import ROOT, FOLDS, export_fold, load_json
from prepare_nnunet import configure_paths
from server_data import digest, write_json


def prepare(fold, gpu_memory):
    configure_paths()
    import torch
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    torch.set_num_threads(2)
    split = load_json(ROOT / "04_data/manifests/splits_v1.json")["folds"][fold]
    cases = load_json(ROOT / "04_data/manifests/case_inventory.json")["cases"]
    name = f"Dataset{split['dataset_id']:03d}_{fold}"
    raw = ROOT / "04_data/derived/nnUNet_raw" / name
    prep = ROOT / "04_data/derived/nnUNet_preprocessed" / name
    signature = {"split": split, "gpu_memory_gb": gpu_memory, "inventory_sha256": digest(ROOT / "04_data/manifests/case_inventory.json"), "implementation_sha256": {name: digest(Path(__file__).with_name(name)) for name in ["server_prepare.py", "data_pipeline.py", "prepare_nnunet.py"]}, "nnunet_version": importlib.metadata.version("nnunetv2")}
    marker = prep / "server_preparation_signature.json"
    if marker.exists() and load_json(marker) != signature:
        raise ValueError("已有预处理输入或代码不同，请使用新的工作区")
    for path in [raw, prep]:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("数据准备路径超出工作区")
    complete = ROOT / f"09_reports/{fold}_preparation.json"
    if complete.exists() and marker.exists():
        plan = load_json(prep / "Project1Plans.json")
        prov = load_json(prep / "fingerprint_provenance.json")
        if load_json(complete)["plan_sha256"] != digest(prep / "Project1Plans.json") or prov["sha256"] != digest(prep / "dataset_fingerprint.json") or prov["fit_ids"] != split["train"]:
            raise ValueError("已完成的计划或指纹被改动，停止复用")
        cache = prep / plan["configurations"]["3d_fullres"]["data_identifier"]
        if all((cache / f"{i}{suffix}").is_file() for i in split["train"]+split["val"] for suffix in [".b2nd", "_seg.b2nd", ".pkl"]):
            print(f"{fold} 已准备完成，复用同一份来源数据。", flush=True)
            return
    started = time.perf_counter()
    export_fold(ROOT, fold, split, cases, resume=raw.exists())
    write_json(marker, signature)
    fingerprint = prep / "dataset_fingerprint.json"
    provenance = prep / "fingerprint_provenance.json"
    if not fingerprint.exists() or not provenance.exists():
        extractor = DatasetFingerprintExtractor(split["dataset_id"], num_processes=1, verbose=False)
        extractor.dataset = {i: extractor.dataset[i] for i in split["train"]}
        extractor.dataset_json = dict(extractor.dataset_json, numTraining=len(split["train"]))
        extractor.run(overwrite_existing=True)
        write_json(provenance, {"fit_ids": split["train"], "excluded_val": split["val"], "excluded_target": split["target"], "sha256": digest(fingerprint)})
    prov = load_json(provenance)
    if prov["fit_ids"] != split["train"] or prov["sha256"] != digest(fingerprint):
        raise ValueError("来源指纹验证失败")
    plans_path = prep / "Project1Plans.json"
    if not plans_path.exists():
        planner = ExperimentPlanner(split["dataset_id"], gpu_memory_target_in_gb=gpu_memory, plans_name="Project1Plans")
        planner.dataset = {i: planner.dataset[i] for i in split["train"]}
        planner.dataset_json = dict(planner.dataset_json, numTraining=len(split["train"]))
        planner.plan_experiment()
    write_json(prep / "splits_final.json", [{"train": split["train"], "val": split["val"]}])
    manager = PlansManager(load_json(plans_path))
    config = manager.get_configuration("3d_fullres")
    processor = config.preprocessor_class(verbose=False)
    cache = prep / config.data_identifier
    cache.mkdir(parents=True, exist_ok=True)
    gt = prep / "gt_segmentations"
    gt.mkdir(exist_ok=True)
    dataset_json = load_json(raw / "dataset.json")
    plan_hash = digest(plans_path)
    for i, case_id in enumerate(split["train"] + split["val"], 1):
        done = cache / f"{case_id}.complete.json"
        files_ok = all((cache / f"{case_id}{s}").exists() for s in [".b2nd", "_seg.b2nd", ".pkl"])
        if not (done.exists() and files_ok and load_json(done).get("plan_sha256") == plan_hash):
            # 使用官方逐病例接口，避免官方整批入口先删除已有缓存目录。
            processor.run_case_save(str(cache / case_id), [str(raw / "imagesTr" / f"{case_id}_0000.nii.gz")], str(raw / "labelsTr" / f"{case_id}.nii.gz"), manager, config, dataset_json)
            write_json(done, {"plan_sha256": plan_hash})
        shutil.copyfile(raw / "labelsTr" / f"{case_id}.nii.gz", gt / f"{case_id}.nii.gz")
        print(f"{fold} 预处理 {i}/{len(split['train'])+len(split['val'])}：{case_id}", flush=True)
    write_json(complete, {"status": "passed", "seconds": time.perf_counter()-started, "source_train": len(split["train"]), "source_val": len(split["val"]), "target_used": False, "gpu_memory_target_gb": gpu_memory, "plan_sha256": plan_hash})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", choices=list(FOLDS))
    parser.add_argument("--gpu-memory", type=float, default=5)
    args = parser.parse_args()
    prepare(args.fold, args.gpu_memory)
