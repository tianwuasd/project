from copy import deepcopy
import os
from pathlib import Path
from time import perf_counter

from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np
import psutil
import torch

from ..artifacts.store import digest, file_digest, operation_lock, read_json, require_output, write_json
from ..data.dataset import CaseDataset
from ..datasets.manifest import load_manifest
from ..models.stunet.adapter import StunetAdapter
from ..observability.events import event
from ..training.checkpoint import load_checkpoint, prediction_snapshot
from ..training.engine import code_identity, device_for
from ..preprocessing.prepare import validate_prepared_config
from .metrics import case_data_identity
from .restore import restore


def predict(config, output, inputs=None, case_ids=None):
    started = perf_counter()
    training = read_json(Path(config["run_dir"]) / "run.json")
    prepared = deepcopy(training["prepared"])
    validate_prepared_config(prepared, config)
    if code_identity() != training["scientific"]["code_identity"]:
        raise ValueError("预测代码与训练记录不一致，需要经过验证的兼容记录")
    if inputs is not None:
        prepared["manifest"] = load_manifest(inputs, prepared["dataset"], prepared["scope"])
    output = require_output(output, config["work_root"], [prepared["manifest"]["data_root"], config["run_dir"]])
    names = {row["case_id"] for row in prepared["manifest"]["cases"]}
    requested = names if case_ids is None else set(case_ids)
    if not requested or not requested <= names:
        raise ValueError("预测病例不存在于输入清单中")
    options = config["prediction"]
    torch.set_num_threads(config["resources"]["cpu_threads"])
    device = device_for(config)
    with operation_lock(output), prediction_snapshot(config["run_dir"]) as pointer:
        completed = read_json(output / "predictions.json") if (output / "predictions.json").exists() else {"cases": {}}
        dataset = CaseDataset(prepared, config["resources"], with_label=False)
        model, state = None, None
        results, reused = [], 0
        for row in prepared["manifest"]["cases"]:
            name = row["case_id"]
            if name not in requested:
                continue
            source_stat = Path(row["image"]).stat()
            identity = digest({"checkpoint": pointer["sha256"], "plan": training["scientific"]["prepared_identity"],
                               "row": row, "source_stamp": [source_stat.st_size, source_stat.st_mtime_ns],
                               "options": options, "technical": prepared["technical"]})
            path = output / f"{name}.nii.gz"
            previous = completed["cases"].get(name)
            if previous and previous["identity"] == identity:
                if not path.is_file() or file_digest(path) != previous["sha256"]:
                    raise ValueError("已有预测损坏，拒绝将其作为完成结果")
                reused += 1
                results.append({**previous, "reused": True})
                continue
            if path.exists() or previous:
                raise FileExistsError("已有预测的身份不同，请使用新的预测目录")
            mark = perf_counter()
            entry = dataset.get(name)
            if model is None:
                state = load_checkpoint(config["run_dir"], training["identity"], pointer=pointer)
                model = StunetAdapter.build(state["spec"]).to(device).eval()
                model.load_state_dict(state["model"], strict=True)
                spec = state["spec"]
                del state
            footprint = int(entry["image"].size * (spec["num_classes"] * 12 + 8))
            if psutil.virtual_memory().available < config["resources"]["reserve_ram_bytes"] + footprint:
                raise MemoryError("整例概率融合内存不足")
            read_seconds = perf_counter() - mark
            mark = perf_counter()

            def predictor(image):
                values = model(image)
                return values[0] if isinstance(values, (tuple, list)) else values

            with torch.inference_mode():
                logits = sliding_window_inference(
                    torch.from_numpy(entry["image"])[None], roi_size=tuple(spec["patch"]),
                    sw_batch_size=options["window_batch"], predictor=predictor, overlap=options["overlap"],
                    mode=options["blend"], sw_device=device, device=torch.device("cpu"), progress=False)
                if not torch.isfinite(logits).all():
                    raise ValueError("滑窗预测出现非有限数值")
                probabilities = logits[0].float().softmax(0).numpy()
            compute_seconds = perf_counter() - mark
            mark = perf_counter()
            native = restore(probabilities, entry, prepared["plan"])
            affine = entry["source_affine"] if options["export_space"] == "source" else entry["working_affine"]
            image = nib.Nifti1Image(native, np.asarray(affine))
            image.header.set_xyzt_units(entry["source_units"] if options["export_space"] == "source" else "mm")
            pending = path.with_name(path.stem + ".pending.nii.gz")
            nib.save(image, pending)
            check = nib.load(pending)
            observed = np.asanyarray(check.dataobj)
            if check.shape != tuple(entry["source_shape_xyz"]) or not np.array_equal(observed, native):
                raise ValueError("预测文件独立读回不一致")
            if not np.allclose(check.affine, affine, atol=1e-5, rtol=0):
                raise ValueError("预测空间头独立读回不一致")
            del observed
            os.replace(pending, path)
            receipt = {"case_id": name, "identity": identity, "file": path.name, "sha256": file_digest(path),
                       "data_identity": case_data_identity(prepared, row), "dataset": prepared["dataset"],
                       "reference_sha256": row.get("sha256", {}).get("label"),
                       "checkpoint_sha256": pointer["sha256"],
                       "shape_xyz": entry["source_shape_xyz"], "affine": affine, "roi_xyz": entry["roi_xyz"],
                       "scope": prepared["scope"], "export_space": options["export_space"],
                       "unknown_classes": row["unknown_classes"], "reused": False,
                       "timing": {"load_seconds": read_seconds, "sliding_window_seconds": compute_seconds,
                                  "restore_write_seconds": perf_counter() - mark, **entry["timing"]}}
            completed["cases"][name] = receipt
            write_json(output / "predictions.json", completed)
            event(output, "prediction", case_id=name, scope=prepared["scope"], timing=receipt["timing"])
            results.append(receipt)
            del native, probabilities, logits, entry
            dataset.clear()
        summary = {"state": "completed", "scope": prepared["scope"], "cases": results,
                   "new": len(results) - reused, "reused": reused, "seconds": perf_counter() - started}
        write_json(output / "summary.json", summary)
        return summary
