from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
from time import perf_counter

from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np
import psutil
import torch

from ..artifacts.store import digest, file_digest, operation_lock, read_json, require_output, require_result_output, write_json
from ..data.dataset import CaseDataset
from ..datasets.manifest import load_manifest
from ..models.stunet.adapter import StunetAdapter
from ..observability.events import event, logged, stage
from ..training.checkpoint import load_checkpoint, prediction_snapshot
from ..training.engine import code_identity, device_for
from ..training.resources import resource_lease
from ..training.session import admitted_training, memory_for
from ..preprocessing.prepare import validate_prepared_config
from ..preprocessing.offline_store import storage_lease
from .metrics import case_data_identity
from .restore import restore
from .log_cursor import start_event_range,finish_event_range
from .blocks import window_probabilities, restore_blocks
from .workspace import PredictionWorkspace


@logged("prediction")
def predict(config, output, inputs=None, case_ids=None, manifest=None):
    if inputs is not None and manifest is not None:
        raise ValueError("预测输入文件与已验证清单只能指定一种")
    if config['resources']['device']=='cuda' and torch.cuda.is_initialized():
        raise RuntimeError("GPU 预测需要新的命令进程，以使用准入后的可见设备")
    runtime = deepcopy(config)
    runtime["resources"].update(gpu_count=1, cpu_processes=1, min_gpus=None, batch_layout=None, share=False,
                                workers=0, cache_bytes=0, prefetch=1, memory_policy='host_fraction')
    runtime['training']['batch'] = 1
    if runtime["resources"]["gpu_ids"] is not None and len(runtime["resources"]["gpu_ids"]) != 1:
        raise ValueError("单次预测仅使用一张卡，请配置一个 gpu_ids 或使用自动选卡")
    training = read_json(Path(config["run_dir"]) / "run.json")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    prepared = deepcopy(training["prepared"])
    if inputs is not None:
        prepared['manifest'] = load_manifest(inputs,prepared['dataset'],prepared['scope'])
    if manifest is not None:
        prepared["manifest"] = manifest
    if config.get('input_source',{}).get('path'):
        from ..data.input_binding import bind_prepared_source
        prepared = bind_prepared_source(prepared,config['input_source']['path'])
    budget = deepcopy(prepared)
    if case_ids is not None:
        budget['manifest']['cases'] = [row for row in prepared['manifest']['cases'] if row['case_id'] in case_ids]
    with admitted_training(runtime, budget, training["scientific"]["spec"], stage="prediction") as (allocation, session):
        runtime['resource_handle'] = allocation['resource_handle']
        runtime['prediction_case_estimates'] = {row['case_id']:row['workspace_bytes']
            for row in allocation['supply']['case_estimates']}
        runtime['prediction_case_plans'] = {row['case_id']:row for row in allocation['supply']['case_estimates']}
        if allocation['gpus']:
            os.environ["CUDA_VISIBLE_DEVICES"] = allocation["gpus"][0]["uuid"]
        try:
            return _predict(runtime, output, inputs, case_ids, manifest, prepared_source=prepared)
        finally:
            if allocation['gpus']:
                torch.cuda.empty_cache()
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _predict(config, output, inputs=None, case_ids=None, manifest=None, prepared_source=None):
    started = perf_counter()
    training = read_json(Path(config["run_dir"]) / "run.json")
    prepared = deepcopy(prepared_source if prepared_source is not None else training["prepared"])
    if config['dataset'] != prepared['dataset']:
        raise ValueError('预测配置与输入数据集不同')
    if inputs is not None:
        prepared["manifest"] = load_manifest(inputs, prepared["dataset"], prepared["scope"])
    if manifest is not None:
        prepared["manifest"] = deepcopy(manifest)
    protected = [prepared["manifest"]["data_root"]]
    if inputs is not None:
        protected.append(inputs)
    output = require_result_output(config, output, protected)
    log_dir = require_output(config["run_dir"], config["work_root"],
                             [prepared["manifest"]["data_root"], Path(__file__).parents[3]])
    event_cursor = start_event_range(log_dir)
    names = {row["case_id"] for row in prepared["manifest"]["cases"]}
    requested = names if case_ids is None else set(case_ids)
    if not requested or not requested <= names:
        raise ValueError("预测病例不存在于输入清单中")
    options = config["prediction"]
    scientific_options = {key:value for key,value in options.items()
                          if key not in ('execution','block_bytes','max_block_bytes')}
    torch.set_num_threads(config["resources"]["cpu_threads"])
    device = device_for(config)
    with ExitStack() as cleanup, operation_lock(output), storage_lease(prepared, config), prediction_snapshot(config["run_dir"], options.get('checkpoint','latest')) as pointer:
        memory = cleanup.enter_context(memory_for(dict(config=config,resource_handle=config['resource_handle'])))
        case_cleanup = cleanup.enter_context(ExitStack())
        task_path = output/'prediction-task.json'
        task = dict(training_run=str(Path(config['run_dir']).resolve()), dataset=prepared['dataset'],
            source=prepared.get('storage'), geometry=prepared['plan'], technical=prepared['technical'],
            available_cases=sorted(names), prediction=scientific_options)
        if task_path.exists():
            recorded = read_json(task_path)
            if recorded['task'] != task:
                raise ValueError('已有预测任务的模型运行、来源或几何参数不同，请使用新的输出目录')
            pointer = recorded['pointer']
        else:
            from ..models.workflow import save_state
            selected = load_checkpoint(config['run_dir'],training['identity'],pointer=pointer)
            save_state(output/'selected-model.pt',{key:selected[key] for key in ('identity','model','spec')})
            del selected
            write_json(task_path, dict(task=task, pointer=pointer, snapshot='selected-model.pt'))
        completed = read_json(output / "predictions.json") if (output / "predictions.json").exists() else {"cases": {}}
        from .commit import CaseCommit
        commits = CaseCommit(output,digest(task))
        commits.recover(completed['cases'],lambda records: write_json(output/'predictions.json',dict(cases=records)))
        dataset = CaseDataset(prepared, config["resources"], with_label=False, log_dir=log_dir,
                              logging=config.get("logging"))
        model, state = None, None
        results, reused = [], 0
        for row in prepared["manifest"]["cases"]:
            name = row["case_id"]
            if name not in requested:
                continue
            event(log_dir, "prediction_case_started", case_id=name, completed=len(results), total=len(requested))
            identity = digest({"checkpoint": pointer.get('sha256', pointer['file']), "plan": training["scientific"]["prepared_identity"],
                               "row": row,
                               "options": scientific_options, "technical": prepared["technical"]})
            path = output / f"{name}.nii.gz"
            previous = completed["cases"].get(name)
            if previous and previous["identity"] == identity:
                if not path.is_file():
                    raise FileNotFoundError('完成记录对应预测文件不存在：'+str(path))
                reused += 1
                results.append({**previous, "reused": True})
                event(log_dir, "prediction_reused", case_id=name, completed=len(results), total=len(requested), file=str(path))
                continue
            if path.exists() or previous:
                raise FileExistsError("已有预测的身份不同，请使用新的预测目录")
            mark = perf_counter()
            amount = config['prediction_case_estimates'][name]
            case_cleanup.enter_context(memory.client.case(amount,purpose='prediction_case:'+name))
            plan = config['prediction_case_plans'][name]
            execution, block_bytes = plan['prediction_execution'], plan['prediction_block_bytes']
            event(log_dir,'prediction_case_admitted',case_id=name,workspace_bytes=amount,
                  execution=execution, block_bytes=block_bytes, whole_workspace_bytes=plan['whole_workspace_bytes'])
            workspace = None
            if execution == 'blocks':
                workspace = case_cleanup.enter_context(PredictionWorkspace(config,
                    dict(output=str(output), case_id=name, identity=identity),block_bytes,memory))
            with stage(config, "prediction_case_load", case_id=name):
                entry = dataset.get(name)
            if model is None:
                with stage(config, "prediction_model_load", device=str(device)):
                    state = torch.load(output/'selected-model.pt',map_location='cpu',weights_only=False)
                    if state['identity'] != training['identity']:
                        raise ValueError('固定预测模型与当前训练运行不同')
                    model = StunetAdapter.build(state["spec"]).to(device).eval()
                    model.load_state_dict(state["model"], strict=True)
                    spec = state["spec"]
                    del state
            read_seconds = perf_counter() - mark
            mark = perf_counter()

            def predictor(image):
                values = model(image)
                return values[0] if isinstance(values, (tuple, list)) else values

            with stage(config, "sliding_window", case_id=name, shape=list(entry["image"].shape),
                       patch=spec["patch"], overlap=options["overlap"], window_batch=options["window_batch"]), torch.inference_mode():
                if execution == 'blocks':
                    probabilities = window_probabilities(entry['image'],predictor,tuple(spec['patch']),
                        spec['num_classes'],options,device,workspace)
                else:
                    logits = sliding_window_inference(
                        torch.from_numpy(entry["image"])[None], roi_size=tuple(spec["patch"]),
                        sw_batch_size=options["window_batch"], predictor=predictor, overlap=options["overlap"],
                        mode=options["blend"], sw_device=device, device=torch.device("cpu"), progress=False)
                    if not torch.isfinite(logits).all():
                        raise ValueError("滑窗预测出现非有限数值")
                    probabilities = logits[0].float().softmax(0).numpy()
                    del logits
            compute_seconds = perf_counter() - mark
            mark = perf_counter()
            with stage(config, "prediction_restore", case_id=name, target_shape_xyz=entry["source_shape_xyz"]):
                native = restore_blocks(probabilities,entry,workspace) if execution == 'blocks' else restore(probabilities, entry, prepared["plan"])
            affine = entry["source_affine"] if options["export_space"] == "source" else entry["working_affine"]
            image = nib.Nifti1Image(native, np.asarray(affine))
            image.header.set_xyzt_units(entry["source_units"] if options["export_space"] == "source" else "mm")
            pending = path.with_name(path.stem + ".pending.nii.gz")
            event(log_dir, "prediction_writing", case_id=name, file=str(path))
            receipt = {"case_id": name, "identity": identity, "file": path.name,
                       "data_identity": case_data_identity(prepared, row), "dataset": prepared["dataset"],
                       "reference_sha256": row.get("sha256", {}).get("label"),
                       "checkpoint_sha256": pointer.get('sha256'), "checkpoint_file": pointer['file'],
                       "shape_xyz": entry["source_shape_xyz"], "affine": affine, "roi_xyz": entry["roi_xyz"],
                       "scope": prepared["scope"], "export_space": options["export_space"],
                       "execution": execution, "block_bytes": block_bytes if execution == 'blocks' else None,
                       "unknown_classes": row["unknown_classes"], "reused": False,
                       "timing": {"load_seconds": read_seconds, "sliding_window_seconds": compute_seconds,
                                  "restore_write_seconds": perf_counter() - mark, **entry["timing"]}}
            commits.begin(name,receipt,path.name,pending.name)
            nib.save(image, pending)
            receipt['timing']['restore_write_seconds'] = perf_counter()-mark
            commits.ready(receipt)
            commits.recover(completed['cases'],lambda records: write_json(output/'predictions.json',dict(cases=records)))
            event(log_dir, "prediction", case_id=name, scope=prepared["scope"], timing=receipt["timing"],
                  completed=len(results) + 1, total=len(requested), bytes=path.stat().st_size)
            results.append(receipt)
            del native, probabilities, entry, image
            dataset.clear()
            case_cleanup.close()
        summary = {"state": "completed", "scope": prepared["scope"], "cases": results,
                   "new": len(results) - reused, "reused": reused, "seconds": perf_counter() - started}
        event(log_dir, "prediction_completed", cases=len(results), new=summary["new"], reused=reused,
              seconds=summary["seconds"], output=str(output))
        summary.update(finish_event_range(event_cursor))
        summary['model_compute_cases'] = len(results)-reused
        write_json(output / "summary.json", summary)
        return summary
