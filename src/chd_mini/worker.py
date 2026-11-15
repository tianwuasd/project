from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter

from .contracts import file_sha, read_json, select_cases, verify_engine, write_json


def select_inputs(request):
    import nibabel as nib
    import numpy as np
    from chd_segmentation.datasets.splits import apply_split
    from chd_segmentation.datasets.manifest import load_manifest

    settings, work = request["settings"], Path(request["work"])
    source = read_json(settings["manifest"])
    if source["dataset"] != settings["dataset"]:
        raise ValueError("原始清单与数据集不符")
    split_config = {"split": {"path": settings["split"], "format": "auto", "id_field": "case_id", "unit": "patient"},
                    "fold": settings["fold"]}
    assigned = apply_split(source, split_config)
    rows = select_cases(assigned["cases"], settings["seed"])
    root = Path(settings["data_root"])
    for row in rows:
        for kind in ("image", "label"):
            relative = row.get("preferred_files", {}).get(kind, row[kind])
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
                raise ValueError("病例路径不在原始数据根目录")
            row[kind] = str(path)
            if file_sha(path) != row["sha256"][kind]:
                raise ValueError("所选病例来源摘要不符")
    shapes = [list(nib.load(row["image"]).shape) for row in rows]
    minimum = np.min(shapes, axis=0)
    if len(minimum) != 3 or min(minimum) < 32:
        raise ValueError("所选病例无法容纳32立方技术子区域")
    roi = [[int(size // 2 - 16), int(size // 2 + 16)] for size in minimum]
    scope = "technical" if settings["profile"] == "technical" else "formal"
    manifest = {"dataset": source["dataset"], "scope": scope, "plan_scope": scope,
                "data_root": str(root), "cases": rows, "source_scope": source["scope"],
                "frozen_split": assigned["split_source"]}
    manifest_path = work / "inputs" / "selected.json"
    write_json(manifest_path, manifest)
    manifest = load_manifest(manifest_path, source["dataset"], scope)
    write_json(manifest_path, manifest)
    train_manifest = {**manifest, "cases": [row for row in manifest["cases"] if row["roles"] == ["train"]]}
    train_path = work / "inputs" / "training-only.json"
    write_json(train_path, train_manifest)
    workspace = work / "workspace"
    config = {"dataset": settings["dataset"], "model": "stunet", "scope": scope,
              "manifest": str(manifest_path), "plan": str(workspace / "plans" / "train-only.json"),
              "pretrained": settings["pretrained"], "work_root": str(workspace),
              "run_dir": str(workspace / "training"), "result_root": str(workspace / "results"),
              "fold": settings["fold"], "planning": {"mode": "existing"},
              "training": {"patch": [32, 32, 32] if scope == "technical" else [128, 128, 128],
                           "batch": 4, "epochs": settings["epochs"], "steps_per_epoch": settings["steps_per_epoch"],
                           "validation_steps": 1, "save_every_attempts": 1, "seed": settings["seed"],
                           "precision": "fp16" if scope == "formal" and settings["device"] == "cuda" else "fp32"},
              "resources": {"device": settings["device"], "gpu_count": 1, "min_gpus": 1,
                            "cpu_threads": 2, "cpu_processes": 1, "workers": 0, "max_workers": 2,
                            "prefetch": 1, "readers": 1, "max_readers": 1, "cache_bytes": 64 * 1024 ** 2,
                            "reserve_ram_bytes": 2 * 1024 ** 3, "wait_seconds": 30, "share": False,
                            "pressure_retries": 0},
              "technical": {"roi_xyz": roi if scope == "technical" else None, "allow_role_overlap": False}}
    fit = deepcopy(config)
    fit["manifest"] = str(train_path)
    fit["planning"] = {"mode": "auto"}
    write_json(work / "config.json", config)
    write_json(work / "fit-config.json", fit)
    result = {"seed": settings["seed"], "source_manifest_sha256": file_sha(settings["manifest"]),
              "source_split_sha256": file_sha(settings["split"]), "selection_order": "sha256(seed:role:case_id)",
              "patient_independence": True, "source_readonly": True, "scope": scope,
              "roi_xyz": config["technical"]["roi_xyz"], "source_shapes_xyz": shapes,
              "cases": [{"case_id": r["case_id"], "roles": r["roles"], "source_role": r.get("source_role"),
                         "patient_sha256": sha256(r["patient"].encode()).hexdigest(),
                         "geometry_policy": r["geometry_policy"], "unknown_classes": r["unknown_classes"],
                         "sha256": r["sha256"]} for r in rows],
              "versions": {name: version(name) for name in ("torch", "monai", "nnunetv2", "nibabel", "numpy", "psutil")}}
    write_json(Path(request["logs"]) / "selection.json", result)
    return result


def execute(request_path, name):
    request = read_json(request_path)
    settings, work, logs = request["settings"], Path(request["work"]), Path(request["logs"])
    sys.path.insert(0, str(Path(settings["engine_root"]) / "src"))
    baseline = verify_engine(settings["engine_root"], settings["engine_commit"])
    started = perf_counter()
    if name == "select":
        result = select_inputs(request)
    else:
        from chd_segmentation.config import load_config
        from chd_segmentation.preprocessing.prepare import prepare, load_prepared

        config = load_config(work / ("fit-config.json" if name == "fit" else "config.json"))
        prepared_path = work / "workspace" / "prepared.json"
        if name == "fit":
            prepared = prepare(config, work / "workspace" / "fit-prepared.json")
            result = {"fit_ids": prepared["plan"]["fit_ids"], "scope": prepared["scope"],
                      "generation": prepared["plan"]["generation"], "plan_sha256": file_sha(config["plan"])}
        elif name == "prepare":
            prepared = prepare(config, prepared_path)
            expected = sorted(r["case_id"] for r in prepared["manifest"]["cases"] if r["roles"] == ["train"])
            if sorted(prepared["plan"]["fit_ids"]) != expected:
                raise ValueError("拟合病例包含所选训练病例以外的病例")
            result = {"identity": prepared["identity"], "fit_ids": expected, "cases": len(prepared["manifest"]["cases"])}
        elif name == "preprocess":
            import gc
            from chd_segmentation.data.dataset import CaseDataset
            prepared = load_prepared(prepared_path)
            dataset = CaseDataset(prepared, config["resources"], log_dir=config["run_dir"], logging=config["logging"])
            result = {"cases": []}
            for row in prepared["manifest"]["cases"]:
                mark = perf_counter()
                entry = dataset.get(row["case_id"])
                result["cases"].append({"case_id": row["case_id"], "roles": row["roles"],
                                        "shape": list(entry["image"].shape), "timing": entry["timing"],
                                        "seconds": perf_counter() - mark})
                dataset.clear()
                del entry
                gc.collect()
        elif name in ("train_start", "train_resume"):
            from chd_segmentation.training.engine import train
            result = train(config, prepared_path, resume=name == "train_resume", stop_after=1 if name == "train_start" else None)
            target = 1 if name == "train_start" else settings["epochs"] * settings["steps_per_epoch"]
            if result["attempts"] != target or result["global_batch"] != 4:
                raise ValueError("实际训练尝试或全局batch与小样本范围不符")
            if name == "train_resume" and (result["state"] != "completed" or not result["restored_training_state"] or
                                           not result["restored_exact"] or result["updates"] + result["skipped"] != target):
                raise ValueError("完整恢复或有界训练终态核验失败")
        elif name == "test":
            from chd_segmentation.prediction.testing import run_test
            result = run_test(config, "internal")
            if result["cases"] != 1 or result["new_predictions"] != 1:
                raise ValueError("测试必须恰好预测一个冻结测试病例")
            prediction = read_json(Path(result["results"]) / "predictions" / "predictions.json")
            metrics = read_json(Path(result["results"]) / "metrics.json")
            expected = [r["case_id"] for r in load_prepared(prepared_path)["manifest"]["cases"] if r["roles"] == ["test"]]
            if sorted(prediction["cases"]) != sorted(expected) or [r["case_id"] for r in metrics["cases"]] != expected:
                raise ValueError("测试预测或评价病例不符合冻结选择")
            result = {**result, "prediction": prediction, "metrics": metrics}
        else:
            raise ValueError("未知工作阶段")
    if baseline != verify_engine(settings["engine_root"], settings["engine_commit"]):
        raise ValueError("执行期间生产依赖发生改变")
    result = {"stage": name, "worker_seconds": perf_counter() - started, "result": result}
    write_json(logs / (name + ".json"), result)
    return result
