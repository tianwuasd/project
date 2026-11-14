from copy import deepcopy
import atexit
import gc
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import threading
from time import perf_counter
from uuid import uuid4
from unittest import TestCase

from update import code_guard

if mp.current_process().name == "MainProcess":
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
from chd_segmentation.compatibility import compatible_code
from chd_segmentation.observability.events import Stage, event
from chd_segmentation.prediction.infer import predict
from chd_segmentation.prediction.metrics import evaluate
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.engine import code_identity, train


class SmokeWorkspace:
    def __init__(self, root, report):
        self.root, self.report = Path(root).resolve(), Path(report).resolve()
        self.nonce = uuid4().hex
        self.process = psutil.Process()
        self.birth = self.process.create_time()
        self.started = perf_counter()
        self.peak_rss, self.min_available = 0, psutil.virtual_memory().available
        self.peak_group_rss, self.device = 0, "cpu"
        self.stop = threading.Event()
        self.results = []

    def observe(self):
        while not self.stop.wait(.2):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            processes = {p.info["pid"]: p.info for p in psutil.process_iter(["pid", "ppid", "memory_info"])}
            family, pending = {self.process.pid}, [self.process.pid]
            while pending:
                parent = pending.pop()
                children = {pid for pid, p in processes.items() if p["ppid"] == parent} - family
                family.update(children)
                pending.extend(children)
            group = sum(processes[pid]["memory_info"].rss for pid in family
                        if pid in processes and processes[pid]["memory_info"] is not None)
            self.peak_group_rss = max(self.peak_group_rss, group)

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
                  "peak_process_group_rss_bytes": self.peak_group_rss,
                  "minimum_available_ram_bytes": self.min_available, "peak_vram_bytes": None,
                  "device": self.device, "artifacts_before_cleanup": files,
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


def run(spec_path, root, report, validate_only=False, check_logging=False, startup_receipt=None):
    experiments = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))["experiments"]
    if not 1 <= len(experiments) <= 2 or len({row["case"]["image"] for row in experiments}) != len(experiments):
        raise ValueError("短测只允许总计一至两个不同真实病例")
    with SmokeWorkspace(root, report) as workspace:
        for experiment in experiments:
            started = perf_counter()
            only_validate = validate_only or experiment.get("validate_only", False)
            attempts = experiment.get("attempts", 2)
            if type(attempts) is not int or not 1 <= attempts <= 2:
                raise ValueError("每例短测只允许一至两次训练更新")
            first_stop = 0 if attempts == 1 else 1
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
                           "plan": str(process / "plans" / "preprocessing_plan.json"), "pretrained": experiment["pretrained"],
                           "work_root": str(process), "run_dir": str(process / "training"),
                           "result_root": str(results)})
            config["planning"].update(experiment.get("planning", {}))
            if config["planning"]["mode"] != "auto":
                raise ValueError("短测必须从所选真实病例自动生成技术计划")
            config["training"].update({"patch": [32, 32, 32], "attempts": attempts, "batch": 1,
                                       "save_every_attempts": 2, "foreground_fraction": 1.})
            config["technical"] = {"roi_xyz": experiment["roi_xyz"], "allow_role_overlap": True}
            config["resources"].update(workers=1, max_workers=1, readers=1)
            config["resources"].update(experiment.get("resources", {}))
            if config["resources"]["device"] != "cpu":
                raise ValueError("本地短测只允许 CPU，CUDA 验收需要独立授权")
            write_json(directory / "config.json", config)
            config = load_config(directory / "config.json")
            record = {"dataset": name, "case_id": row["case_id"], "source_identity": row["sha256"],
                      "roi_xyz": experiment["roi_xyz"], "patch": config["training"]["patch"], "device": "cpu"}
            workspace.results.append(record)
            event(config["run_dir"], "smoke_case_started", dataset=name, case_id=row["case_id"],
                  optimizer_update_limit=0 if only_validate else attempts)
            mark = perf_counter()
            prepared = prepare(config, process / "prepared.json")
            record["prepare_seconds"] = perf_counter() - mark
            generation = prepared["plan"]["generation"]
            record["planning"] = {key: generation[key] for key in
                                  ("request_identity", "samples_per_case_applied", "total_samples", "seconds")}
            record["planning"]["fit_ids"] = prepared["plan"]["fit_ids"]
            plan_stamp = Path(config["plan"]).stat().st_mtime_ns
            mark = perf_counter()
            repeated = prepare(config, process / "prepared.json")
            record["planning"]["reuse_seconds"] = perf_counter() - mark
            if repeated != prepared or Path(config["plan"]).stat().st_mtime_ns != plan_stamp:
                raise ValueError("自动计划未按相同身份复用")
            record["planning"]["reused_without_write"] = True
            dataset = CaseDataset(prepared, config["resources"], log_dir=config["run_dir"], logging=config.get("logging"))
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
            record["first_train"] = train(config, process / "prepared.json", stop_after=0 if only_validate else first_stop)
            record["resume_train"] = train(config, process / "prepared.json", resume=True,
                                           stop_after=0 if only_validate else None)
            expected_updates = 0 if only_validate else attempts
            if not record["resume_train"]["restored_exact"] or record["resume_train"]["updates"] != expected_updates:
                raise ValueError("训练恢复状态或更新数未通过")
            if not only_validate and ((first_stop and not record["first_train"]["parameters_changed_this_invocation"]) or not record["resume_train"]["parameters_changed_this_invocation"]):
                raise ValueError("训练更新没有改变实际模型参数")
            record["optimizer_updates_in_this_test"] = expected_updates
            predictions = results / "prediction"
            record["prediction"] = predict(config, predictions)
            record["prediction_reuse"] = predict(config, predictions)
            if record["prediction_reuse"]["reused"] != 1 or record["prediction_reuse"]["new"] != 0:
                raise ValueError("重复预测没有复用完成结果")
            record["evaluation"] = evaluate(predictions, prepared, results / "evaluation.json", config)
            if check_logging:
                check = TestCase()
                repeated_evaluation = evaluate(predictions, prepared, results / "evaluation.json", config)
                check.assertTrue(repeated_evaluation["reused"])
                check.assertEqual(startup_receipt["first_event"], "command_starting")
                check.assertTrue(startup_receipt["flushed_before_exit"])
                with Stage(config["run_dir"], "log_heartbeat_verify", heartbeat_seconds=.001, console=False):
                    for _ in range(4):
                        check.assertEqual(file_digest(row["image"]), row["sha256"]["image"])
                with check.assertRaises(FileNotFoundError), Stage(config["run_dir"], "log_failure_verify"):
                    file_digest(directory / "missing-input-for-log-check.nii.gz")
                logs_path = Path(config["run_dir"]) / "events.jsonl"
                before_bytes = logs_path.stat().st_size
                silent = CaseDataset(prepared, config["resources"], log_dir=config["run_dir"],
                                     logging={"case_details": False})
                silent.get(row["case_id"])
                silent.clear()
                check.assertEqual(logs_path.stat().st_size, before_bytes)
                current = code_identity()
                baseline = "afffd433e388f93fe5d840ba6a899cb2b814c81804e59d4b02c310cb47cf8dde"
                check.assertTrue(compatible_code(baseline, current))
                check.assertFalse(compatible_code("0" * 64, current))
                check.assertFalse(compatible_code(baseline, "0" * 64))
            events = [json.loads(line) for line in (process / "training" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            if (predictions / "events.jsonl").exists() or not any(row["event"] == "prediction" for row in events):
                raise ValueError("预测事件日志未保存到过程目录")
            if check_logging:
                kinds = {r["event"] for r in events}
                stages = {r.get("stage") for r in events}
                required_events = {"plan_case_completed", "plan_reused", "resources_allocated", "checkpoint_committed",
                                   "checkpoint_loaded", "train", "validation", "prediction", "prediction_reused",
                                   "evaluation_completed", "evaluation_reused"}
                required_stages = {"prepare", "source_verify", "read_decompress", "crop_normalize_resample", "batch_supply",
                                   "training_inputs", "model_build", "pretrained_load", "sliding_window", "prediction_restore"}
                if only_validate:
                    required_events -= {"train", "validation"}
                    required_stages -= {"batch_supply"}
                check.assertTrue(required_events <= kinds, required_events - kinds)
                check.assertTrue(required_stages <= stages, required_stages - stages)
                check.assertTrue(any(r["event"] == "heartbeat" and r["stage"] == "log_heartbeat_verify" for r in events))
                failed = next(r for r in events if r["event"] == "stage_failed" and r["stage"] == "log_failure_verify")
                check.assertIn("FileNotFoundError", failed["traceback"])
                check.assertGreaterEqual(len({r["pid"] for r in events}), 2)
                check.assertTrue(all(r["time_utc"].endswith("+00:00") and type(r["pid"]) is int for r in events))
                shutil.copyfile(process / "training" / "events.jsonl", workspace.report / f"{name}-events.jsonl")
                record["logging"] = {"events": len(events), "processes": len({r["pid"] for r in events}),
                                     "bytes": (process / "training" / "events.jsonl").stat().st_size,
                                     "required_events": sorted(required_events), "required_stages": sorted(required_stages),
                                     "heartbeat": True, "failure_traceback": True, "flushed_before_command_exit": True,
                                     "case_details_can_be_disabled": True, "code_compatibility_exact_pair": True}
            record["storage"] = {"work_root": str(process), "result_root": str(results),
                                 "prediction_events_in_course": True, "result_events_absent": True}
            for kind in ("image", "label"):
                if file_digest(row[kind]) != row["sha256"][kind]:
                    raise ValueError("原始文件在短测后发生变化")
            record["seconds"] = perf_counter() - started
            record["source_unchanged"] = True
            write_json(workspace.report / f"{name}.json", record)
            event(config["run_dir"], "smoke_case_completed", dataset=name, case_id=row["case_id"],
                  seconds=record["seconds"], updates=expected_updates)
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
