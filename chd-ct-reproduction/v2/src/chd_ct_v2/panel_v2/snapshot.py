"""Content-addressed, verified code and configuration for detached tasks."""

import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

from ..recovery import file_hash, write_json
from .identity import path


def files(root):
    result = {}
    for name in ("src", "configs", "start.py", "start_server.py", "pyproject.toml"):
        target = path(root / name)
        paths = sorted(target.rglob("*")) if target.is_dir() else [target]
        for item in paths:
            if "__pycache__" in item.parts or item.suffix == ".pyc":
                continue
            item = path(item)
            if item.is_file():
                result[item.relative_to(root).as_posix()] = file_hash(item)
    return result


def freeze(root, runtime):
    root, runtime = path(root), path(runtime)
    hashes = files(root)
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = runtime / "releases" / digest
    if destination.exists():
        verify({"frozen_code": str(destination), "code_sha256": digest})
        return destination, digest
    temporary = destination.with_name("." + uuid4().hex)
    temporary.mkdir(parents=True)
    for name in hashes:
        target = temporary / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    write_json(temporary / "snapshot.json", {"sha256": digest, "files": hashes})
    verify({"frozen_code": str(temporary), "code_sha256": digest})
    temporary.rename(destination)
    return destination, digest


def verify(plan):
    root = path(plan["frozen_code"])
    record = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    actual = files(root)
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    if record != {"sha256": digest, "files": actual} or digest != plan["code_sha256"]:
        raise ValueError("任务冻结代码已改变，拒绝恢复或运行")
    return root


def freeze_inputs(plan, root, frozen):
    """Copy external configs under runtime, preserving hashes in each task plan."""
    for step in plan["steps"]:
        for key in ("config", "split_file", "rules", "evidence", "diagnosis_source", "spacing_excel"):
            value = step["arguments"].get(key)
            if not value:
                continue
            source = path(value)
            if not source.is_file():
                continue
            if source.is_relative_to(root) and (frozen / source.relative_to(root)).is_file():
                target = frozen / source.relative_to(root)
            else:
                target = Path(plan["runtime"]) / "frozen-inputs" / file_hash(source) / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
                if file_hash(target) != file_hash(source):
                    raise ValueError("冻结输入校验失败")
            step["arguments"][key] = str(target)
    from ..server.workflow import frozen_files

    plan["frozen_files"] = frozen_files(plan["steps"])
    for step in plan["steps"]:
        for key in ("diagnosis_source", "spacing_excel"):
            value = step["arguments"].get(key)
            if value:
                plan["frozen_files"][value] = file_hash(value)
