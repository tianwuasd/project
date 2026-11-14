import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import file_digest, read_json, write_json
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.datasets.splits import apply_split
from chd_segmentation.prediction.infer import predict
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.checkpoint import load_checkpoint
from chd_segmentation.training.engine import train


def run(spec, root, report, worlds, device="cpu", workers=None):
    experiments = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))["experiments"]
    if not 1 <= len(experiments) <= 2 or any(not 1 <= w <= 4 for w in worlds):
        raise ValueError("验证限定一至两个真实病例及 1–4 个进程")
    with SmokeWorkspace(root, report) as workspace:
        workspace.device = device
        for experiment in experiments:
            name = experiment["dataset"]
            directory = workspace.root / name
            directory.mkdir()
            row = {**experiment["case"], "roles": ["train", "validation"]}
            manifest = {"dataset": name, "scope": "technical", "data_root": experiment["data_root"], "cases": [row]}
            write_json(directory / "manifest.json", manifest)
            config = deepcopy(DEFAULTS)
            config.update(dataset=name, scope="technical", manifest=str(directory / "manifest.json"),
                          pretrained=experiment["pretrained"], work_root=str(directory / "course"),
                          run_dir=str(directory / "course" / "rank1"), result_root=str(directory / "result"))
            config["training"].update(patch=[32, 32, 32], batch=4, epochs=1, steps_per_epoch=2,
                                      validation_steps=1, attempts=2, save_every_attempts=2, foreground_fraction=1.)
            config["resources"].update(workers=0, max_workers=1, reserve_ram_bytes=1024 ** 3,
                                       cache_bytes=64 * 1024 ** 2, device=device)
            config["technical"].update(roi_xyz=experiment["roi_xyz"], allow_role_overlap=True)
            write_json(directory / "config.json", config)
            config = load_config(directory / "config.json")
            started = perf_counter()
            prepared_path = directory / "course" / "prepared.json"
            prepared = prepare(config, prepared_path)
            record = {"dataset": name, "case_id": row["case_id"], "source_identity": row["sha256"],
                      "prepare_seconds": perf_counter() - started, "runs": []}
            workspace.results.append(record)
            reference = None
            for world in worlds:
                current = deepcopy(config)
                current["run_dir"] = str(directory / "course" / f"rank{world}")
                current["resources"].update(cpu_processes=world, gpu_count=world,
                                             workers=(1 if world == 1 else 0) if workers is None else workers,
                                             max_workers=2 if workers == "auto" else 1)
                began = perf_counter()
                first = train(current, prepared_path, stop_after=1)
                before = load_checkpoint(current["run_dir"], first["identity"])
                before_model = {k: value.clone() for k, value in before["model"].items()}
                del before
                second = train(current, prepared_path, resume=True)
                state = load_checkpoint(current["run_dir"], second["identity"])
                assert first["attempts"] == 1 and first["updates"] == 1
                assert second["attempts"] == 2 and second["updates"] == 2 and second["epoch"] == 1
                assert second["restored_exact"] and second["world_size"] == world
                assert sum(second["batch_layout"]) == 4
                assert any(not torch.equal(before_model[k], value) for k, value in state["model"].items())
                assert len(list(Path(current["run_dir"]).glob("state-*.pt"))) == 1
                difference = None
                if reference is None:
                    reference = {k: value.clone() for k, value in state["model"].items()}
                else:
                    difference = max(float((reference[k] - value).abs().max()) for k, value in state["model"].items())
                    for key, value in state["model"].items():
                        torch.testing.assert_close(reference[key], value, atol=3e-5, rtol=3e-4)
                record["runs"].append({"world_size": world, "first": first, "resume": second,
                                       "seconds": perf_counter() - began, "max_weight_difference": difference,
                                       "updates_per_second": 2 / (perf_counter() - began)})
                del state, before_model
                if world == worlds[-1]:
                    cpu_prediction = deepcopy(current)
                    cpu_prediction["resources"].update(device="cpu", cpu_processes=1)
                    record["prediction"] = predict(cpu_prediction, directory / "result" / "prediction")
                    record["prediction_reuse"] = predict(cpu_prediction, directory / "result" / "prediction")
                    assert record["prediction_reuse"]["reused"] == 1
            split_path = directory / "selected.json"
            # 只为当前真实病例选择一个角色，不调用任何随机划分。
            write_json(split_path, {"train": [row["case_id"]], "development": [], "test": []})
            split_config = {**config, "split": {"path": str(split_path), "format": "roles", "id_field": "case_id"}}
            selected = apply_split(manifest, split_config)
            assert selected["cases"][0]["roles"] == ["train"] and row["roles"] == ["train", "validation"]
            assert selected["split_source"]["sha256"] == file_digest(split_path)
            record["split_selected_exactly"] = True
            for kind in ("image", "label"):
                assert file_digest(row[kind]) == row["sha256"][kind]
            record["source_unchanged"] = True
            write_json(workspace.report / f"{name}.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实病例多进程有界验证与自动清理")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--worlds", default="1,2,3,4")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", choices=("auto",))
    args = parser.parse_args()
    run(args.spec, args.work_root, args.report, [int(w) for w in args.worlds.split(",")], args.device, args.workers)
