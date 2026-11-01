"""Explicit V2 registry; queries never discover V1 or mutate selection."""

from contextlib import contextmanager
from pathlib import Path

from ..recovery import output_lock, read_json, write_json
from .identity import PANEL, check, claim, locations, path


def read(values):
    instance = check(values)
    index = locations(values)["runtime"] / "pipeline_v2/index.json"
    if not index.exists():
        return {"panel": PANEL, "instance": instance, "selected": None, "jobs": {}}
    value = read_json(path(index))
    if (
        value.get("panel") != PANEL
        or value.get("instance") != instance
        or not isinstance(value.get("jobs"), dict)
    ):
        raise ValueError("V2任务索引损坏，未覆盖")
    return value


@contextmanager
def transaction(values):
    claim(values)
    folder = locations(values)["runtime"] / "pipeline_v2"
    with output_lock(folder):
        value = read(values)
        yield value
        write_json(folder / "index.json", value)


def resolve_entry(values, key, value=None, *, allow_trash=False):
    value = value or read(values)
    key = key or value.get("selected")
    record = value["jobs"].get(key)
    if not isinstance(record, dict) or record.get("id") != key or (record.get("trashed") and not allow_trash):
        raise ValueError("只能选择本V2索引登记且未回收的任务")
    root = locations(values)["results"]
    directory = path(record["directory"])
    if directory.parent != root or directory.name != key:
        raise ValueError("V2任务路径与登记身份不一致")
    return record


def get(values, key=None):
    record = resolve_entry(values, key)
    file = path(Path(record["directory"]) / "plan.json")
    plan = read_json(file)
    if (
        plan.get("panel") != PANEL
        or plan.get("instance") != check(values)
        or plan.get("v2_id") != record["id"]
    ):
        raise ValueError("任务不属于当前V2实例")
    if {k: str(locations(values)[k]) for k in locations(values)} != plan.get("v2_roots"):
        raise ValueError("任务目录配置与原V2实例不同")
    return record, plan


def list_jobs(values, include_trash=False):
    value = read(values)
    records = []
    for key in value["jobs"]:
        record = resolve_entry(values, key, value, allow_trash=True)
        if record.get("trashed") and not include_trash:
            continue
        state_file = Path(record["directory"]) / "job_state.json"
        state = read_json(path(state_file)) if state_file.exists() and not record.get("trashed") else {}
        records.append({**record, "status": state.get("status", record.get("status", "pending"))})
    return records


def active(folder):
    from ..server.inspection import job_active
    from ..server.resume import live_commands

    return job_active(Path(folder)) or bool(live_commands(read_json(Path(folder) / "plan.json")))


def recycle(values, key, restore=False):
    from ..server.jobs import acquire_lock

    roots = locations(values)
    with acquire_lock(roots["runtime"]), transaction(values) as value:
        record = resolve_entry(values, key, value, allow_trash=True)
        original = path(record["directory"])
        target = path(roots["results"] / ".trash" / key)
        if restore:
            if not record.get("trashed"):
                raise ValueError("任务不在回收区")
            if target.exists():
                if original.exists():
                    raise ValueError("原任务路径已占用")
                target.rename(original)
            elif not original.exists():
                raise ValueError("原任务和回收路径均缺失")
            record["trashed"] = False
        else:
            if active(original):
                raise ValueError("任务或子进程仍活动，不能回收")
            if any(key in r.get("sources", []) for r in value["jobs"].values() if not r.get("trashed")):
                raise ValueError("任务仍被下游使用，不能回收")
            if target.exists():
                raise ValueError("回收目标已存在")
            record["trashed"] = True
            # Durable intent permits restore even if interrupted around rename.
            write_json(roots["runtime"] / "pipeline_v2/index.json", value)
            target.parent.mkdir(exist_ok=True)
            try:
                original.rename(target)
            except BaseException:
                record["trashed"] = False
                raise
            if value.get("selected") == key:
                value["selected"] = None
