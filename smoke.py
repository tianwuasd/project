from copy import deepcopy
import atexit
import gc
import json
from pathlib import Path
import shutil
import sys
import threading
from time import perf_counter
from uuid import uuid4

from update import code_guard

_code_guard = code_guard(__file__)
_code_guard.__enter__()
atexit.register(_code_guard.__exit__, None, None, None)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import psutil
import torch
import yaml

from chd_segmentation.artifacts.store import file_digest, read_json, write_json
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.data.dataset import CaseDataset
from chd_segmentation.prediction.infer import predict
from chd_segmentation.prediction.metrics import evaluate
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.engine import train


class SmokeWorkspace:
    def __init__(self, root, report):
        self.root, self.report = Path(root).resolve(), Path(report).resolve()
        self.nonce = uuid4().hex
        self.process = psutil.Process()
        self.birth = self.process.create_time()
        self.started = perf_counter()
        self.peak_rss, self.min_available = 0, psutil.virtual_memory().available
        self.stop = threading.Event()
        self.results = []

    def observe(self):
        while not self.stop.wait(.2):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)

    def __enter__(self):
        if self.root == self.report or self.report.is_relative_to(self.root) or self.root.is_relative_to(self.report):
            raise ValueError("测试工作区与保留报告目录必须独立")
        self.root.mkdir(parents=True, exist_ok=False)
        self.report.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "owner.json", {"nonce": self.nonce, "pid": self.process.pid, "birth": self.birth})
        self.thread = threading.Thread(target=self.observe, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.stop.set()
        self.thread.join()
        owner = read_json(self.root / "owner.json")
        if owner != {"nonce": self.nonce, "pid": self.process.pid, "birth": self.birth}:
            raise ValueError("测试工作区归属不匹配，拒绝清理")
        if self.root.is_symlink() or any(path.is_symlink() for path in self.root.rglob("*")):
            raise ValueError("测试工作区含符号链接，拒绝递归清理")
        files = [{"path": str(path.relative_to(self.root)), "bytes": path.stat().st_size}
                 for path in self.root.rglob("*") if path.is_file()]
        result = {"state": "passed" if exception is None else "failed", "scope": "technical",
                  "error": None if exception is None else str(exception), "experiments": self.results,
                  "seconds": perf_counter() - self.started, "peak_rss_bytes": self.peak_rss,
                  "minimum_available_ram_bytes": self.min_available, "peak_vram_bytes": None,
                  "device": "cpu", "artifacts_before_cleanup": files,
                  "artifact_count": len(files), "artifact_bytes": sum(row["bytes"] for row in files),
                  "work_root": str(self.root), "report_root": str(self.report)}
        write_json(self.report / "acceptance.json", result)
        # 目录由本次独占创建；原件、已有环境和共享权重均在目录外。
        shutil.rmtree(self.root)
        result["cleanup"] = {"deleted_files": len(files), "released_bytes": result["artifact_bytes"],
                             "work_root_exists": self.root.exists()}
        write_json(self.report / "acceptance.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return False


def run(spec_path, root, report, validate_only=False):
    experiments = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))["experiments"]
    if not 1 <= len(experiments) <= 2 or len({row["case"]["image"] for row in experiments}) != len(experiments):
        raise ValueError("短测只允许总计一至两个不同真实病例")
    with SmokeWorkspace(root, report) as workspace:
        for experiment in experiments:
            started = perf_counter()
            name = experiment["dataset"]
            directory = workspace.root / name
            directory.mkdir(exist_ok=False)
            process, results = directory / "course", directory / "result"
            row = deepcopy(experiment["case"])
            row["roles"] = ["train", "validation"]
            manifest = {"dataset": name, "scope": "technical", "data_root": experiment["data_root"], "cases": [row],
                        "plan_scope": experiment.get("plan_scope", "technical")}
            write_json(directory / "manifest.json", manifest)
            config = deepcopy(DEFAULTS)
            config.update({"dataset": name, "scope": "technical", "manifest": str(directory / "manifest.json"),
                           "plan": experiment["plan"], "pretrained": experiment["pretrained"],
                           "work_root": str(process), "run_dir": str(process / "training"),
                           "result_root": str(results)})
            config["training"].update({"patch": [32, 32, 32], "attempts": 2, "batch": 1,
                                       "save_every_attempts": 2, "foreground_fraction": 1.})
            config["technical"] = {"roi_xyz": experiment["roi_xyz"], "allow_role_overlap": True}
            write_json(directory / "config.json", config)
            config = load_config(directory / "config.json")
            record = {"dataset": name, "case_id": row["case_id"], "source_identity": row["sha256"],
                      "roi_xyz": experiment["roi_xyz"], "patch": config["training"]["patch"], "device": "cpu"}
            workspace.results.append(record)
            mark = perf_counter()
            prepared = prepare(config, process / "prepared.json")
            record["prepare_seconds"] = perf_counter() - mark
            dataset = CaseDataset(prepared, config["resources"])
            first = dataset.get(row["case_id"])
            if not (first["label"] > 0).any():
                raise ValueError("所选技术子区域没有真实前景标签")
            second = dataset.get(row["case_id"])
            if first is not second or dataset.hits != 1 or dataset.misses != 1:
                raise ValueError("MONAI 确定性病例缓存未正确复用")
            record["preprocessing"] = {"shape_czyx": list(first["image"].shape), "label_values": list(map(int, set(first["label"].ravel()))),
                                       "unknown_classes": row["unknown_classes"], "timing": first["timing"],
                                       "cache_bytes": dataset.cached_bytes, "cache_hits": dataset.hits}
            dataset.clear()
            del first, second, dataset
            gc.collect()
            record["first_train"] = train(config, process / "prepared.json", stop_after=0 if validate_only else 1)
            record["resume_train"] = train(config, process / "prepared.json", resume=True,
                                           stop_after=0 if validate_only else None)
            expected_updates = 0 if validate_only else 2
            if not record["resume_train"]["restored_exact"] or record["resume_train"]["updates"] != expected_updates:
                raise ValueError("训练恢复状态或更新数未通过")
            if not validate_only and (not record["first_train"]["parameters_changed_this_invocation"] or not record["resume_train"]["parameters_changed_this_invocation"]):
                raise ValueError("训练更新没有改变实际模型参数")
            record["optimizer_updates_in_this_test"] = expected_updates
            predictions = results / "prediction"
            record["prediction"] = predict(config, predictions)
            record["prediction_reuse"] = predict(config, predictions)
            if record["prediction_reuse"]["reused"] != 1 or record["prediction_reuse"]["new"] != 0:
                raise ValueError("重复预测没有复用完成结果")
            record["evaluation"] = evaluate(predictions, prepared, results / "evaluation.json", config)
            events = [json.loads(line) for line in (process / "training" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            if (predictions / "events.jsonl").exists() or not any(row["event"] == "prediction" for row in events):
                raise ValueError("预测事件日志未保存到过程目录")
            record["storage"] = {"work_root": str(process), "result_root": str(results),
                                 "prediction_events_in_course": True, "result_events_absent": True}
            for kind in ("image", "label"):
                if file_digest(row[kind]) != row["sha256"][kind]:
                    raise ValueError("原始文件在短测后发生变化")
            record["seconds"] = perf_counter() - started
            record["source_unchanged"] = True
            write_json(workspace.report / f"{name}.json", record)
            gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STU-Net 两数据集真实有界短测")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--validate-only", action="store_true", help="核验保护逻辑、初始状态恢复与预测；不执行优化器更新")
    args = parser.parse_args()
    run(args.spec, args.work_root, args.report, args.validate_only)
