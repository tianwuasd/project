from copy import deepcopy
import json
from pathlib import Path

from .records import classification_for_task, read_json, write_json


def _absolute(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def generate_run_configs(config, base_path, output, gpus=1, batch_size=1,
                         epochs=300, device="cuda:0", check=False):
    base_file = Path(base_path).resolve(strict=True)
    base = base_file.parent
    cfg = deepcopy(config)
    if not 1 <= gpus <= batch_size or not 2 <= epochs <= 300:
        raise ValueError("需1≤显卡数≤全局batch，轮数2到300")
    if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("device须为cpu或cuda:非负整数")
    if device == "cpu" and gpus != 1:
        raise ValueError("CPU配置仅允许gpu_count=1")
    for name in ("state_root", "data_root", "course_root", "input_manifest", "diagnoses", "aaoca_root", "aaoca_pairs", "imagechd_root", "imagechd_manifest"):
        if cfg.get(name) is not None:
            cfg[name] = str((base / cfg[name]).resolve())
    output = Path(output).resolve()
    results = {}
    payloads = {}
    task = cfg.get("classification", {}).get("task", "aaoca_present")
    if task not in {"aaoca_present", "origin_left"}:
        raise ValueError("当前仅生成检测或侧别的配置")
    for task in (task,):
        existing = deepcopy(config.get("classification", cfg.get("classification", {})))
        strategy = "local_from_scratch" if task == "aaoca_present" else existing.get("strategy")
        if task == "origin_left" and strategy not in {"author_detection_transfer", "local_detection_transfer"}:
            raise ValueError("侧别训练须声明真实检测父权重；检测配置可独立生成")
        if task == "origin_left" and not existing.get("pretrained_detection"):
            raise ValueError("侧别缺少实际父检测权重声明")
        prep = deepcopy(cfg)
        prep.update(classification_recipe="canonical_ras_v1", offset_ras_mm=[10, 0, 10],
                    prepared_input_policy="compatible", training_policy="local_adaptation")
        old = prep.pop("prepared_inputs", None)
        if old is not None:
            prep["provenance_prepared_inputs"] = old
        for key in ("direction_review", "coverage_review", "direction_reviews", "coverage_reviews"):
            if key in prep:
                prep["provenance_" + key] = prep.pop(key)
        prep["classification"] = classification_for_task(existing, task, existing.get("prediction_case_ids", []))
        prep["classification"].update(task=task, strategy=strategy)
        prep["prepared_inputs"] = {}

        train = deepcopy(prep)
        train["classification"].update(device=device, gpu_count=gpus, batch_size=batch_size,
            epochs=epochs, learning_rate_recipe="paper_lr_1e-3", member_index=0,
            seed=20261121, run_directory=str(_absolute(cfg["course_root"], base) /
                "runs" / f"{task}-canonical-v1"), checkpoint_interval_updates=10)
        train["classification"].pop("gpu_ids", None)
        results[task] = {"preprocess": prep, "train": train}
        for kind, value in results[task].items():
            path = output / f"project.{task}.{kind}.json"
            payloads[path] = json.loads(json.dumps(value, default=str))
    size = sum(len(json.dumps(v, ensure_ascii=False).encode("utf-8")) for v in payloads.values())
    if size > 128 * 1024:
        raise ValueError("两份正式配置超过128KiB预算")
    for path, value in payloads.items():
        if path.exists() and read_json(path) != value:
            raise FileExistsError("已有配置内容不同，保留原件：" + str(path))
    if not check:
        for path, value in payloads.items():
            if not path.exists():
                write_json(path, value)
    return {"check_only": check, "files": [str(p) for p in payloads], "bytes_estimated": size,
            "configurations": {p.name: v for p, v in payloads.items()}}
