from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import multiprocessing as mp
import os
import sys
from pathlib import Path
from time import perf_counter, sleep

import psutil
import torch

from ..artifacts.store import digest, operation_lock, read_json, require_output, write_json
from ..models.stunet.preprocess import StunetTransform
from ..models.stunet.streaming import stream_case
from ..observability.events import event, logged, stage
from ..observability.progress import Progress
from ..training.memory_guard import MemoryGuard
from ..training.budget import case_estimates, private_bytes
from ..training.host import host_capacity
from ..training.resources import refresh_supply, register_member, resource_lease
from .offline_store import (FORMAT, case_directory, case_key, inspect_case, live_references, owned_remove,
                            recipe, save_case, validate_storage, begin_case)
from .prepare import load_prepared, validate_prepared_config
from .scheduling import update_case_slots
from ..acceleration.contracts import CaseEstimate, CasePolicy, ResourceSnapshot
from ..acceleration.cpu import choose_execution


_WORKER_CANCEL = None
_WORKER_HANDLE = None


def worker_cancel(config):
    return config.get('memory_cancel', _WORKER_CANCEL)


def case_guard(config):
    from ..acceleration.legacy import SessionGuard
    handle = config.get('resource_handle', _WORKER_HANDLE)
    if handle is not None:
        return SessionGuard(handle)
    return MemoryGuard(config, 'preprocess_case', owner=config.get('memory_owner'), cancel=worker_cancel(config))


def produce_case(prepared, config, root, name, key, recipe_identity):
    torch.set_num_threads(1)
    began = perf_counter()
    row = next(r for r in prepared["manifest"]["cases"] if r["case_id"] == name)
    event(config["run_dir"], "preprocess_case_started", case_id=name, file=str(Path(root) / key))
    directory = case_directory(root, key)
    pending = begin_case(directory, key, recipe_identity)
    with case_guard(config) as guard, ExitStack() as held:
        estimates = config.get('_case_estimates')
        if estimates is None:
            subset = {**prepared, 'manifest': {**prepared['manifest'], 'cases': [row]}}
            estimate = case_estimates(config, subset, 'preprocess')[0]
        else:
            estimate = estimates[name]
        observation = guard.boundary()
        budget = observation['budget']
        available = max(0, budget['headroom_bytes'] + budget['own_pending_bytes'])
        snapshot = ResourceSnapshot(observation['time_utc'], observation['monotonic'], 0., observation['host'],
            observation['project']['accounted_bytes'], available, observation['paused'], observation['stop_requested'], 0.)
        settings = config['preprocessing']
        explicit = config.get('_explicit_preprocessing', [])
        execution = settings['execution']
        if execution == 'auto' and estimate['legacy_whole_case_bytes'] <= config.get('_whole_capacity', 0):
            execution = 'whole'
        policy = CasePolicy(execution, settings['block_bytes'] if 'block_bytes' in explicit else None,
            settings['max_block_bytes'] if 'max_block_bytes' in explicit else None,
            settings['max_case_threads'] or max(1, int(snapshot.host['cores']*config['resources']['cpu_fraction'])))
        decision = choose_execution(CaseEstimate(name, estimate['legacy_whole_case_bytes'], 0,
            4096, estimate.get('resampled_upper_shape', estimate.get('working_shape'))), snapshot, policy)
        while decision.mode == 'wait':
            guard.checkpoint()
            sleep(config['resources']['poll_seconds'])
            observation = guard.boundary()
            snapshot.headroom_bytes = max(0, observation['budget']['headroom_bytes'] + observation['budget']['own_pending_bytes'])
            snapshot.cancelled = observation['stop_requested']
            if snapshot.cancelled:
                raise InterruptedError('预处理资源等待已取消')
            decision = choose_execution(CaseEstimate(name, estimate['legacy_whole_case_bytes'], 0,
                4096, estimate.get('resampled_upper_shape', estimate.get('working_shape'))), snapshot, policy)
        event(config['run_dir'], 'preprocess_execution_selected', case_id=name, execution=decision.mode,
              reason=decision.reason, whole_peak_bytes=estimate['legacy_whole_case_bytes'],
              workspace_bytes=decision.workspace_bytes, block_bytes=decision.block_bytes)
        workspace = None
        if decision.mode == 'whole':
            ticket = held.enter_context(guard.admit(estimate['legacy_whole_case_bytes'], label='whole_case'))
            if ticket is None:
                raise InterruptedError('整例处理已取消')
            entry = StunetTransform(prepared, config['resources']['reserve_ram_bytes'],
                with_label=row.get('label') is not None, log_dir=config['run_dir'], logging=config['logging'])(row)
        else:
            config = {**config, 'preprocessing': deepcopy(config['preprocessing'])}
            if 'block_bytes' not in explicit:
                config['preprocessing']['block_bytes'] = max(1024, decision.block_bytes)
            if 'max_block_bytes' not in explicit:
                config['preprocessing']['max_block_bytes'] = max(config['preprocessing']['block_bytes'], decision.block_bytes)
            if 'read_block_bytes' not in explicit:
                config['preprocessing']['read_block_bytes'] = min(config['preprocessing']['read_block_bytes'],
                                                                 config['preprocessing']['block_bytes'])
            entry, workspace = stream_case(prepared, row, config, pending, guard)
        try:
            with stage(config, "preprocess_case_save", case_id=name):
                record = save_case(directory, key, recipe_identity, entry, workspace)
        finally:
            if workspace is not None:
                workspace.release_handles()
    pointer = case_pointer(record)
    event(config["run_dir"], "preprocess_case_completed", case_id=name, seconds=perf_counter() - began,
          timing=entry["timing"], bytes=sum(v["stamp"][0] for v in record["files"].values()))
    return name, pointer


def case_pointer(record):
    return {"case_id": record['entry']['case_id'], "key": record["case_key"], "identity": record["identity"], "shape_czyx": record["files"]["image.npy"]["shape"],
            "array_bytes": sum(v["array_bytes"] for n, v in record["files"].items() if n.endswith(".npy")),
            "locations_bytes": record["files"]["locations.npz"]["stamp"][0]}


def initialize_worker(config, owner, cancel, handle=None):
    global _WORKER_CANCEL, _WORKER_HANDLE
    _WORKER_CANCEL = cancel
    _WORKER_HANDLE = handle
    torch.set_num_threads(1)
    if handle is not None:
        from ..acceleration.runtime import attach
        attach(handle)
    else:
        register_member(config, owner)


def process_pending(config, prepared, root, identity, pending, pointers, catalog, producer=produce_case):
    from ..acceleration import ResourceSession, ResourceDemand, RuntimeRequest
    from ..acceleration.memory import system_headroom
    facts, r = host_capacity(), config['resources']
    headroom, _ = system_headroom(facts, min(.8, r['ram_fraction']), r['reserve_ram_bytes'])
    rss = psutil.Process().memory_info().rss
    capacity = max(1, min(r['memory_limit_bytes'], 100_000_000_000, headroom + rss - 256*1024**2))
    cores = max(1, int(facts['cores']*r['cpu_fraction']))
    requested = r['max_workers'] if r['workers']=='auto' else r['workers']
    readers = r['max_readers'] if r['readers']=='auto' else r['readers']
    workers = max(1, min(max(1, requested), readers, len(pending), max(1, cores-1), max(1, capacity//(2*1024**3))))
    estimates = case_estimates(config, prepared, 'preprocess')
    wanted = {name for name, _ in pending}
    whole = sorted((e['legacy_whole_case_bytes'] for e in estimates if e['case_id'] in wanted), reverse=True)
    capacity = min(capacity, (workers+1)*2*1024**3 + sum(whole[:workers]))
    while workers > 1 and (workers+1)*2*1024**3 + sum(whole[:workers]) > capacity:
        workers -= 1
    request = RuntimeRequest(str(Path(config['run_dir']).resolve()), 'preprocess',
        Path(config['work_root'])/'resources', config['run_dir'], r,
        ResourceDemand(workspace_bytes=capacity, cpu_slots=cores))
    before, started = len(pointers), perf_counter()
    runtime = deepcopy(config)
    runtime['_case_estimates'] = {record['case_id']: record for record in estimates}
    runtime['_whole_capacity'] = max(0, capacity-(workers+1)*2*1024**3)
    with ResourceSession(request) as session:
        cores = session.allocation.cpu_threads
        workers = min(workers, max(1, cores-1))
        runtime['memory_owner'] = session.handle.owner
        session.handle.state['case_slots'] = workers
        session.handle.state['cpu_slots'] = cores
        context = mp.get_context('spawn')
        progress = Progress(config, 'preprocess', total=len(prepared['manifest']['cases']), reused=before)
        running = {}
        try:
            pool = ProcessPoolExecutor(workers, mp_context=context, initializer=initialize_worker,
                initargs=(runtime, session.handle.owner, session.handle.cancel, session.handle))
            try:
                while pending or running:
                    if session.handle.cancel.is_set():
                        raise InterruptedError('预处理已取消，完成病例保留')
                    while pending and len(running)<workers:
                        name, key = pending.pop(0)
                        running[pool.submit(producer, prepared, runtime, str(root), name, key, identity)] = name
                    done, _ = wait(running, timeout=.5, return_when=FIRST_COMPLETED)
                    for future in done:
                        name, pointer = future.result()
                        running.pop(future)
                        pointers[name] = pointer
                        catalog['cases'][pointer['key']] = pointer
                        catalog.setdefault('case_index', {})[name] = pointer['key']
                        write_json(root/'catalog.json', catalog)
                    session.handle.state['case_slots'] = max(1, min(workers, len(pending)+len(running)))
                    progress.update(done=len(pointers), new=len(pointers)-before,
                        pending=len(pending), active=len(running), current_case=', '.join(running.values()) or None)
                    progress.snapshot(force=bool(done))
            finally:
                session.handle.cancel.set()
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            session.handle.cancel.set()
            progress.update(done=len(pointers), new=len(pointers)-before, active=0)
            progress.close()
            event(config['run_dir'], 'resource_management_timing', seconds=perf_counter()-started,
                  timing=session.timing.snapshot(), accounting_method='rss_sum')


@logged("preprocess")
def preprocess(config, prepared_path, output, *, producer=produce_case, recipe_builder=recipe, method_label="STUNet"):
    prepared = load_prepared(prepared_path)
    if prepared['dataset'] != config['dataset']:
        raise ValueError('配置与预处理数据集不一致')
    if "storage" in prepared:
        raise ValueError("preprocess 需要 prepare 生成的来源描述，请保留独立的 source.json")
    base = config["preprocessing"]["root"]
    if base is None:
        raise ValueError("请设置 preprocessing.root 或 preprocess --root 指定 data 内的预处理目录")
    base = Path(base).resolve()
    protected = [prepared["manifest"]["data_root"], Path(__file__).parents[3], config["work_root"], config["pretrained"]]
    for item in protected:
        item = Path(item).resolve()
        if base == item or base.is_relative_to(item) or item.is_relative_to(base):
            raise ValueError("离线数据目录与原件、环境过程或源码重叠")
    output = require_output(output, config["work_root"], [base, *protected[:2]])
    content = recipe_builder(prepared, config)
    output = Path(output)
    existing = read_json(output) if output.is_file() else None
    if existing is not None and existing.get('storage'):
        content = read_json(Path(existing['storage']['root']) / 'catalog.json')['recipe']
    identity = digest(content)
    with operation_lock(base):
        index_path = base / "index.json"
        index = read_json(index_path) if index_path.exists() else {}
        if existing is not None and existing.get('storage'):
            root = Path(existing['storage']['root']).resolve()
        elif identity not in index:
            label = ("ImageCHD" if prepared["dataset"] == "imagechd" else "CHD1282") + "-" + method_label + "-" + datetime.now().strftime("%Y%m%d")
            index[identity] = str(Path(label) / f"fold_{prepared['fold']}" / identity[:16])
            write_json(index_path, index)
        if existing is None or not existing.get('storage'):
            root = (base / index[identity]).resolve()
        if not root.is_relative_to(base) or root == base:
            raise ValueError("离线索引路径越界")
    with operation_lock(root):
        catalog_path = root / "catalog.json"
        catalog = read_json(catalog_path) if catalog_path.exists() else {
            "format": FORMAT, "recipe": content, "recipe_identity": identity, "cases": {}}
        if catalog["recipe"] != content or catalog["recipe_identity"] != identity:
            raise ValueError("已有数据版本的完整配方不一致")
        pending, pointers, reused = [], {}, 0
        references, active = live_references(root)
        for row in prepared["manifest"]["cases"]:
            name, key = row["case_id"], case_key(row, identity)
            directory = case_directory(root, key)
            record = inspect_case(directory, key, identity)
            if record is not None:
                previous = catalog["cases"].get(key)
                if previous is None:
                    previous = case_pointer(record)
                pointers[name] = previous
                reused += 1
                event(config["run_dir"], "preprocess_case_reused", case_id=name, completed=reused,
                      total=len(prepared["manifest"]["cases"]))
            else:
                if active:
                    raise RuntimeError("数据版本仍被活动任务引用，禁止修补或替换病例")
                if directory.exists():
                    owner = read_json(directory / "owner.json")
                    if owner != {"case_key": key, "recipe_identity": identity}:
                        raise ValueError("核验失败的病例目录归属不符")
                    owned_remove(directory)
                pending.append((name, key))
        began = perf_counter()
        environment = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        for k in environment:
            os.environ[k] = "1"
        try:
            if pending:
                process_pending(config, prepared, root, identity, pending, pointers, catalog, producer)
        finally:
            for k, value in environment.items():
                if value is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = value
        catalog["cases"].update({p["key"]: p for p in pointers.values()})
        write_json(catalog_path, catalog)
        result = deepcopy(prepared)
        result.pop("identity")
        result["storage"] = {"format": FORMAT, "root": str(root), "base_root": str(base),
                             "recipe_identity": identity, "complete": True, "cases": pointers}
        result["identity"] = digest(result)
        validate_storage(result, expected_transform=content["transform"])
        write_json(output, result)
    event(config["run_dir"], "preprocess_completed", cases=len(pointers), reused=reused,
          new=len(pointers) - reused, output=str(output), data=str(root), seconds=perf_counter() - began)
    final = Progress(config, "preprocess", total=len(pointers), reused=reused, started=began)
    final.update(done=len(pointers), new=len(pointers) - reused, pending=0, active=0,
                 storage_bytes=sum(p["array_bytes"] + p["locations_bytes"] for p in pointers.values()))
    final.close()
    return result


def data_info(prepared_path):
    prepared = load_prepared(prepared_path)
    root = validate_storage(prepared, expected_transform=None)
    with operation_lock(root):
        references, active = live_references(root)
        files = [p for p in root.rglob("*") if p.is_file()]
        result = {"data": str(root), "recipe_identity": prepared["storage"]["recipe_identity"],
                  "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                  "references": references, "active_references": active, "deleted": False}
        result["cleanup_scope"] = "仅此配方及此折的数据版本；原件、其他折和其他模型目录均不属于此范围"
        result["cleanup_blocked"] = bool(active) or len({x["model"] for x in references.values()}) > 1
        result["warning"] = "历史任务恢复或重新预测仍需此数据；请核对全部引用后决定保留，不自动删除正式数据"
    return result
