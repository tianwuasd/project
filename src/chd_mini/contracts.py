from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess


TOOL_ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return sha256_file(stream)


def reference_snapshot(sources):
    return {key: file_sha(path) for key, path in sources.items()}


def verify_references(sources, expected):
    current = reference_snapshot(sources)
    changed = sorted(key for key in expected if current[key] != expected[key])
    if changed:
        raise ValueError("冻结输入发生变化：" + ", ".join(changed))
    return current


def sha256_file(stream):
    result = sha256()
    for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
        result.update(block)
    return result.hexdigest()


def validate_settings(value):
    for key in ("epochs", "steps_per_epoch"):
        if type(value[key]) is not int or not 1 <= value[key] <= 2:
            raise ValueError("小样本轮次和每轮尝试次数必须为1或2")
    if type(value["batch"]) is not int or value["batch"] != 4:
        raise ValueError("小样本流程保持全局batch 4")
    if value["epochs"] * value["steps_per_epoch"] < 2:
        raise ValueError("保存后继续更新需要至少两次有界尝试")
    if value["profile"] not in ("technical", "full128") or value["device"] not in ("cpu", "cuda"):
        raise ValueError("需要明确技术范围和设备")
    return value


def select_cases(rows, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("选择种子必须为非负整数")
    patients = {}
    for row in rows:
        if len(row["roles"]) != 1 or row["roles"][0] not in ("train", "validation", "test") or not row.get("patient"):
            raise ValueError("冻结清单需要单独角色和可靠患者身份")
        roles = patients.setdefault(row["patient"], set())
        roles.update(row["roles"])
        if len(roles) != 1:
            raise ValueError("同一患者跨越冻结角色")
    chosen, used = [], set()
    for role, amount in (("train", 2), ("validation", 1), ("test", 1)):
        candidates = [r for r in rows if r["roles"] == [role] and r.get("cohort", "main") == "main"
                      and (role != "test" or r.get("test_set", "internal") == "internal")]
        ordered = sorted(candidates, key=lambda r: (sha256(f"{seed}:{role}:{r['case_id']}".encode()).hexdigest(), r["case_id"]))
        picks = []
        for row in ordered:
            if row["patient"] not in used:
                picks.append(dict(row))
                used.add(row["patient"])
            if len(picks) == amount:
                break
        if len(picks) != amount:
            raise ValueError("冻结角色中独立患者不足：" + role)
        chosen.extend(picks)
    return chosen


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8",
                                   env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip()


def verify_engine(root, expected):
    root = Path(root).resolve(strict=True)
    if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("生产版本必须为完整提交")
    if root == TOOL_ROOT or root.is_relative_to(TOOL_ROOT) or TOOL_ROOT.is_relative_to(root):
        raise ValueError("生产项目与工具必须使用独立目录")
    actual = git(root, "rev-parse", "HEAD")
    if actual != expected:
        raise ValueError("生产项目提交已变化：" + actual)
    paths = ["src", "run.py", "pyproject.toml"]
    if git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths):
        raise ValueError("生产依赖存在未提交修改")
    files = git(root, "ls-files", "--", *paths).splitlines()
    result = {name: file_sha(root / name) for name in files}
    return {"root": str(root), "commit": actual, "files": result,
            "content_sha256": sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}


def confined(path, root):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    absolute = path.absolute()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("工作路径越出专用目录")
    for candidate in (absolute, *absolute.parents):
        if candidate == root:
            break
        if candidate.is_symlink() or (os.name == "nt" and candidate.exists() and
                                      candidate.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("工作路径含链接或目录联接")
    return resolved


def load_settings(path):
    value = validate_settings(read_json(path))
    if value.get("schema") != 1:
        raise ValueError("配置schema必须为1")
    for name in ("engine_root", "python", "manifest", "split", "data_root", "pretrained"):
        value[name] = str(Path(value[name]).resolve(strict=True))
    for name in ("data_root", "engine_root"):
        location = Path(value[name])
        if TOOL_ROOT == location or TOOL_ROOT.is_relative_to(location) or location.is_relative_to(TOOL_ROOT):
            raise ValueError("工具目录与受保护输入重叠")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("选择种子必须是非负整数")
    return value
