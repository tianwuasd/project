"""服务器可继续的数据准备，逐病例保存完成标记，保留已完成预处理。"""
import argparse
import os
import sys
import importlib.metadata
from pathlib import Path
import shutil
import time
from data_pipeline import ROOT, FOLDS, export_fold, load_json
from prepare_nnunet import configure_paths
from server_data import digest, write_json


def compatible_signature(previous, current):
    return previous == current


def completion_directory(prep, cache, case_ids, plan_hash):
    """完成记录放在训练数据目录之外；旧记录原子迁移，影像不动。

    nnU-Net 2.8.1 会检查数据目录内全部文件的后缀，不能混入 JSON。
    先核查全部待迁移记录，再移动；若中断，下次可以继续。
    """
    destination = prep / "server_case_completion" / cache.name
    moves = []
    for case_id in case_ids:
        source = cache / f"{case_id}.complete.json"
        target = destination / source.name
        if source.exists():
            if load_json(source) != {"plan_sha256": plan_hash}:
                raise ValueError(f"旧完成记录与当前计划不符：{source}")
            if target.exists() and load_json(target) != load_json(source):
                raise ValueError(f"新旧完成记录冲突，保留文件：{source}")
            moves.append((source, target))
    destination.mkdir(parents=True, exist_ok=True)
    for source, target in moves:
        source.replace(target)
    if moves:
        print(f"已迁移 {len(moves)} 份完成记录到 {destination}；预处理影像保持不变。", flush=True)
    return destination


def prepare(fold, gpu_memory):
    from task_resources import begin_cpu_stage
    begin_cpu_stage(ROOT,'prepare')
    configure_paths()
    import torch
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from task_resources import torch_threads
    torch_threads(torch)
    split = load_json(ROOT / "04_data/manifests/splits_v1.json")["folds"][fold]
    phase_started = time.perf_counter()
    progress_path = ROOT / f'00_admin/preparation_progress/{fold}.json'
    def progress(phase, **fields):
        write_json(progress_path, dict(fold=fold, phase=phase, updated=time.time(),
                   elapsed_seconds=time.perf_counter()-phase_started, **fields))
    progress('核对来源与缓存', completed=0, total=len(split['train'])+len(split['val']))
    cases = load_json(ROOT / "04_data/manifests/case_inventory.json")["cases"]
    from server_data import freeze_source_labels
    source_labels = freeze_source_labels(ROOT, fold)
    name = f"Dataset{split['dataset_id']:03d}_{fold}"
    raw = ROOT / "04_data/derived/nnUNet_raw" / name
    prep = ROOT / "04_data/derived/nnUNet_preprocessed" / name
    from preparation_identity import current as preparation_implementation
    signature = {"source_labels": source_labels, "split": split, "gpu_memory_gb": gpu_memory, "inventory_sha256": digest(ROOT / "04_data/manifests/case_inventory.json"), "implementation_sha256": preparation_implementation(), "nnunet_version": importlib.metadata.version("nnunetv2")}
    marker = prep / "server_preparation_signature.json"
    previous = load_json(marker) if marker.exists() else None
    if previous is not None and not compatible_signature(previous, signature):
        raise ValueError("已有预处理输入或代码不同，请使用新的工作区")
    for path in [raw, prep]:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("数据准备路径超出工作区")
    if previous is not None and previous != signature:
        # 保存旧签名供追溯，绝不把任意代码改动当成此次兼容迁移。
        backup = prep / "server_preparation_signature.before_marker_fix.json"
        if backup.exists() and load_json(backup) != previous:
            raise ValueError("旧签名备份冲突，停止迁移")
        write_json(backup, previous)
    complete = ROOT / f"09_reports/{fold}_preparation.json"
    if complete.exists() and marker.exists():
        plan = load_json(prep / "WholeheartBaselinePlans.json")
        prov = load_json(prep / "fingerprint_provenance.json")
        if load_json(complete)["plan_sha256"] != digest(prep / "WholeheartBaselinePlans.json") or prov["sha256"] != digest(prep / "dataset_fingerprint.json") or prov["fit_ids"] != split["train"]:
            raise ValueError("已完成的计划或指纹被改动，停止复用")
        cache = prep / plan["configurations"]["3d_fullres"]["data_identifier"]
        completion_directory(prep, cache, split["train"] + split["val"], digest(prep / "WholeheartBaselinePlans.json"))
        if all((cache / f"{i}{suffix}").is_file() for i in split["train"]+split["val"] for suffix in [".b2nd", "_seg.b2nd", ".pkl"]):
            write_json(marker, signature)
            progress('已完成：复用兼容预处理', completed=len(split['train'])+len(split['val']), total=len(split['train'])+len(split['val']))
            print(f"{fold} 已准备完成，复用同一份来源数据。", flush=True)
            return
    started = time.perf_counter()
    progress('导出来源病例（不读取目标标签）')
    export_fold(ROOT, fold, split, cases, resume=raw.exists())
    write_json(marker, signature)
    fingerprint = prep / "dataset_fingerprint.json"
    provenance = prep / "fingerprint_provenance.json"
    if not fingerprint.exists() or not provenance.exists():
        progress('计算 fit 病例数据指纹')
        from case_execution import fingerprint_workers
        extractor = DatasetFingerprintExtractor(split["dataset_id"], num_processes=fingerprint_workers(ROOT,raw,split["train"]), verbose=False)
        extractor.dataset = {i: extractor.dataset[i] for i in split["train"]}
        extractor.dataset_json = dict(extractor.dataset_json, numTraining=len(split["train"]))
        from case_execution import cpu_scope
        with cpu_scope(1):extractor.run(overwrite_existing=True)
        write_json(provenance, {"fit_ids": split["train"], "excluded_val": split["val"], "excluded_target": split["target"], "sha256": digest(fingerprint)})
    prov = load_json(provenance)
    if prov["fit_ids"] != split["train"] or prov["sha256"] != digest(fingerprint):
        raise ValueError("来源指纹验证失败")
    plans_path = prep / "WholeheartBaselinePlans.json"
    if not plans_path.exists():
        progress('生成 nnU-Net 网络规划（CPU）')
        planner = ExperimentPlanner(split["dataset_id"], gpu_memory_target_in_gb=gpu_memory, plans_name="WholeheartBaselinePlans")
        planner.dataset = {i: planner.dataset[i] for i in split["train"]}
        planner.dataset_json = dict(planner.dataset_json, numTraining=len(split["train"]))
        from case_execution import cpu_scope,planner_threads
        with cpu_scope(planner_threads(ROOT,raw,split['train'])):planner.plan_experiment()
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
    done_dir = completion_directory(prep, cache, split["train"] + split["val"], plan_hash)
    total = len(split['train'])+len(split['val'])
    from task_resources import case_budget
    budget=case_budget(ROOT,[raw/'imagesTr'/f'{i}_0000.nii.gz' for i in split['train']+split['val']],list(reversed(config.spacing)))
    if budget['workers']<1:raise MemoryError('prepare capacity: '+budget['capacity_status'])
    if budget['workers']>=1:
        from cpu_case_queue import run
        commands=[];finished=[]
        for case_id in split['train']+split['val']:
            done=done_dir/f'{case_id}.complete.json'
            valid=done.exists() and load_json(done).get('plan_sha256')==plan_hash and all((cache/f'{case_id}{s}').exists() for s in ['.b2nd','_seg.b2nd','.pkl'])
            if valid:finished.append(case_id)
            else:commands.append((case_id,[sys.executable,'-B',str(Path(__file__).resolve()),fold,'--gpu-memory',str(gpu_memory),'--case',case_id]))
        def completed(case_id):
            finished.append(case_id);progress('逐病例预处理',completed=len(finished),total=total)
            print(f'{fold} 预处理 {len(finished)}/{total}：{case_id}',flush=True)
        progress('逐病例预处理',completed=len(finished),total=total)
        run(commands,budget['workers'],ROOT/f'00_admin/preparation_logs/{fold}',completed,
            refresh=lambda:case_budget(ROOT,[raw/'imagesTr'/f'{i}_0000.nii.gz' for i in split['train']+split['val']],list(reversed(config.spacing))))
        for case_id in split['train']+split['val']:shutil.copyfile(raw/'labelsTr'/f'{case_id}.nii.gz',gt/f'{case_id}.nii.gz')
        write_json(complete,dict(status='passed',seconds=time.perf_counter()-started,source_train=len(split['train']),source_val=len(split['val']),target_used=False,gpu_memory_target_gb=gpu_memory,plan_sha256=plan_hash))
        progress('已完成',completed=total,total=total)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", choices=list(FOLDS))
    parser.add_argument("--gpu-memory", type=float, default=20)
    parser.add_argument('--case',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        from case_execution import prepare_one
        prepare_one(ROOT,args.fold,args.case)
    else:prepare(args.fold, args.gpu_memory)
