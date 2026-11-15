from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from datetime import datetime
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter, sleep

import psutil
import torch

from ..artifacts.store import digest, operation_lock, read_json, require_output, write_json
from ..models.stunet.preprocess import StunetTransform
from ..observability.events import event, logged, stage
from ..training.budget import case_estimates, private_bytes
from ..training.host import host_capacity
from ..training.resources import refresh_supply, register_member, resource_lease
from .offline_store import (FORMAT, case_directory, case_key, inspect_case, live_references, owned_remove,
                            recipe, save_case, validate_storage)
from .prepare import load_prepared, validate_prepared_config


def produce_case(prepared, config, root, name, key, recipe_identity):
    torch.set_num_threads(1)
    began = perf_counter()
    row = next(r for r in prepared["manifest"]["cases"] if r["case_id"] == name)
    event(config["run_dir"], "preprocess_case_started", case_id=name, file=str(Path(root) / key))
    transform = StunetTransform(prepared, config["resources"]["reserve_ram_bytes"],
                               log_dir=config["run_dir"], logging=config["logging"])
    transform.console = True
    entry = transform(row)
    with stage(config, "preprocess_case_save", case_id=name):
        record = save_case(case_directory(root, key), key, recipe_identity, entry)
    pointer = case_pointer(record)
    event(config["run_dir"], "preprocess_case_completed", case_id=name, seconds=perf_counter() - began,
          timing=entry["timing"], bytes=sum(v["stamp"][0] for v in record["files"].values()))
    return name, pointer


def case_pointer(record):
    return {"key": record["case_key"], "identity": record["identity"], "shape_czyx": record["files"]["image.npy"]["shape"],
            "array_bytes": sum(v["array_bytes"] for n, v in record["files"].items() if n.endswith(".npy")),
            "locations_bytes": record["files"]["locations.npz"]["stamp"][0]}


def initialize_worker(config, owner):
    torch.set_num_threads(1)
    register_member(config, owner)


def process_pending(config, prepared, root, identity, pending, pointers, catalog):
    runtime = deepcopy(config)
    runtime["resources"].update(device="cpu", cpu_processes=1)
    estimates = {r["case_id"]: r["workspace_bytes"] for r in case_estimates(config, prepared, "preprocess")}
    began, last_log, pool, next_adapt = perf_counter(), 0., None, 0.
    with resource_lease(runtime, prepared, {}, stage="preprocess") as allocation:
        running = {}
        try:
            while pending or running:
                if not running and perf_counter() >= next_adapt:
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                        pool = None
                    plan = refresh_supply(runtime, prepared, allocation, apply=True)
                    if not plan["admitted"]:
                        event(config["run_dir"], "preprocess_waiting", reason=plan["reason"], cpu=plan)
                        sleep(config["resources"]["poll_seconds"])
                        continue
                    allocation["supply"] = plan
                    workers = min(plan["workers"][0], len(pending))
                    if workers:
                        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                            initializer=initialize_worker, initargs=(runtime, allocation["owner"]))
                    next_adapt = perf_counter() + config["resources"]["adapt_seconds"]
                    event(config["run_dir"], "preprocess_supply_applied", workers=workers, cpu=plan)
                facts = host_capacity()
                r = config["resources"]
                reserved = sum(running.values())
                children = [{"pid": p.pid, "birth": p.create_time(), "members": []} for p in psutil.Process().children()]
                credited = private_bytes(children)
                spare = facts["available_bytes"] - r["reserve_ram_bytes"] - max(0, reserved - credited)
                capacity = allocation["supply"]["workspace_capacity_bytes"]
                possible = [item for item in pending if estimates[item[0]] <= capacity - reserved and estimates[item[0]] <= spare]
                if possible and len(running) < max(1, workers):
                    name, key = possible[0]
                    pending.remove((name, key))
                    event(config["run_dir"], "preprocess_admitted", case_id=name, workers=workers,
                          active=len(running) + 1, reserved_bytes=reserved + estimates[name], capacity_bytes=capacity)
                    if pool is None:
                        completed_name, pointer = produce_case(prepared, config, str(root), name, key, identity)
                        pointers[completed_name] = pointer
                        catalog["cases"][pointer["key"]] = pointer
                        write_json(root / "catalog.json", catalog)
                    else:
                        running[pool.submit(produce_case, prepared, config, str(root), name, key, identity)] = estimates[name]
                    continue
                done, _ = wait(running, timeout=.5, return_when=FIRST_COMPLETED) if running else (set(), set())
                for future in done:
                    name, pointer = future.result()
                    running.pop(future)
                    pointers[name] = pointer
                    catalog["cases"][pointer["key"]] = pointer
                    write_json(root / "catalog.json", catalog)
                    event(config["run_dir"], "preprocess_progress", completed=len(pointers),
                          total=len(prepared["manifest"]["cases"]), seconds=perf_counter() - began)
                if not done and perf_counter() - last_log >= config["logging"]["heartbeat_seconds"]:
                    event(config["run_dir"], "preprocess_waiting", pending=len(pending), active=len(running),
                          available_bytes=facts["available_bytes"], reserved_bytes=reserved, capacity_bytes=capacity)
                    last_log = perf_counter()
                if not running and pending:
                    sleep(r["poll_seconds"])
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)


@logged("preprocess")
def preprocess(config, prepared_path, output):
    prepared = load_prepared(prepared_path)
    validate_prepared_config(prepared, config)
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
    content = recipe(prepared, config)
    identity = digest(content)
    with operation_lock(base):
        index_path = base / "index.json"
        index = read_json(index_path) if index_path.exists() else {}
        if identity not in index:
            label = ("ImageCHD" if prepared["dataset"] == "imagechd" else "CHD1282") + "-STUNet-" + datetime.now().strftime("%Y%m%d")
            index[identity] = str(Path(label) / f"fold_{prepared['fold']}" / identity[:16])
            write_json(index_path, index)
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
            record = inspect_case(directory, key, identity, full=True)
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
                process_pending(config, prepared, root, identity, pending, pointers, catalog)
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
        validate_storage(result)
        if output.exists() and read_json(output) != result:
            raise FileExistsError("已有离线描述属于不同输入或配方，请指定新的 output 和 run_dir")
        if not output.exists():
            write_json(output, result)
    event(config["run_dir"], "preprocess_completed", cases=len(pointers), reused=reused,
          new=len(pointers) - reused, output=str(output), data=str(root), seconds=perf_counter() - began)
    return result


def data_info(prepared_path):
    prepared = load_prepared(prepared_path)
    root = validate_storage(prepared)
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
