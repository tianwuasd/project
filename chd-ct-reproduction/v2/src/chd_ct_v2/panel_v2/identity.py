"""Fail closed on foreign state, redirected paths and overlapping roots."""

import json
import os
import stat
import re
from pathlib import Path
from uuid import uuid4

PANEL = "chd-ct-panel-v2"
BRANCH = "main"
MARKER = ".ct-panel-v2.json"


def path(value):
    p = Path(value).expanduser().absolute()
    if ".." in p.parts:
        raise ValueError("目录不能包含上级跳转")
    for part in (*reversed(p.parents), p):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("V2运行目录不能是链接或目录联接")
    return p.resolve()


def locations(values):
    result = {k: path(values[k]) for k in ("runtime", "results", "prepared")}
    source=Path(__file__).resolve().parents[3]
    standalone=source.parent if source.name=='v2' and source.parent.name=='chd-ct-reproduction' else source
    code=next((p for p in (source,*source.parents) if (p/'.git').exists()),standalone)
    managed=(source.parent==result['runtime']/'releases' and re.fullmatch(r'[0-9a-f]{64}',source.name))
    if managed:
        # A worker executes its content-addressed source under its own runtime.
        # Verify that exact release before permitting this ancestor relationship.
        from .snapshot import verify
        verify(dict(frozen_code=str(source),code_sha256=source.name))
    for a, x in result.items():
        if x==code or code in x.parents or (x in code.parents and not (a=='runtime' and managed)):
            raise ValueError('V2运行目录必须与代码仓库分开')
        for b, y in result.items():
            if a != b and (x == y or x in y.parents or y in x.parents):
                raise ValueError("V2环境、结果、缓存根目录必须互相独立")
    return result


def marker(folder):
    f = path(folder) / MARKER
    try:
        path(f)
        return json.loads(f.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def check(values):
    roots = locations(values)
    records = {k: marker(p) for k, p in roots.items()}
    if all(v is None for v in records.values()) and all(
        not p.exists() or not any(p.iterdir()) for p in roots.values()
    ):
        return None
    identities = set()
    for key, value in records.items():
        if (
            not isinstance(value, dict)
            or value.get("panel") != PANEL
            or value.get("role") != key
            or not value.get("instance")
        ):
            raise ValueError("目录不属于本V2；不能连接V1、其他工程或未登记旧目录")
        identities.add(value["instance"])
    if len(identities) != 1:
        raise ValueError("V2根目录来自不同实例，不能混用")
    return identities.pop()


def claim(values):
    roots = locations(values)
    current = [marker(p) for p in roots.values()]
    known = {
        r["instance"]
        for r in current
        if isinstance(r, dict) and r.get("panel") == PANEL and r.get("instance")
    }
    if len(known) > 1:
        raise ValueError("V2实例冲突")
    instance = next(iter(known), uuid4().hex)
    # Validate all destinations before writing any marker.
    for key, folder in roots.items():
        record = marker(folder)
        expected = {"panel": PANEL, "instance": instance, "role": key}
        if record is not None and record != expected:
            raise ValueError("V2目录身份不一致")
        if record is None and folder.exists() and any(folder.iterdir()):
            raise ValueError("拒绝接管非空旧目录；请为V2选择新目录")
    for key, folder in roots.items():
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / MARKER
        expected = {"panel": PANEL, "instance": instance, "role": key}
        try:
            with f.open("x", encoding="utf-8") as stream:
                json.dump(expected, stream)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if marker(folder) != expected:
                raise ValueError("V2目录正在被另一个实例初始化") from None
    return check(values)
