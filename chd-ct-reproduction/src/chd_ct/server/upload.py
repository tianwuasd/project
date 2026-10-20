"""Snapshot finished CHD artifacts and upload them with authenticated GitHub CLI.

Only standard-library dependencies are used. The caller must hold the workflow's
runtime lock throughout preparation/upload and obtain consent after the preview.
No public Git download mirror, Git credential helper, or training import is used.
"""

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPOSITORY = "tianwuasd/project"
HOST = "github.com"
CHUNK_BYTES = 1024**3
MAX_ASSETS = 1000
COMMAND_TIMEOUT = 60
UPLOAD_TIMEOUT = 1800
BUFFER_BYTES = 1024**2
TERMINAL_SUCCESS = {"passed", "complete", "completed"}
MODEL_NAMES = {"crop64", "crop128", "all64", "all128", "blood2d", "blood_lstm"}
ROOT_FILES = {"plan.json", "job_state.json", "launcher.log", "receipt.json"}
ARTIFACT_NAMES = {
    "status.json", "server-settings.json", "server.log", "run.log", "report.json", "models.json",
    "training-summary.json", "training-contract.json", "effective-config.json", "folds.json",
    "cpu-tuning.json", "queue.json", "evaluation.json", "metrics.json", "metrics.csv",
    "diagnosis-metrics.json", "trainability.json", "classifier.json", "demo-report.json",
    "prediction-report.json", "prediction-contract.json", "gpu-phase-ready.json",
    "config.json", "config.yaml", "config.yml", "config.toml", "summary.json",
    "resource-plan.json", "resource-probe.json", "cache-reuse-failure.json", "test-report.json",
    "cpu-allocation.json",
}
ARTIFACT_DIRS = {
    "train", "preflight", "test", "predict", "predictions", "evaluation", "evaluate",
    "stage_logs", "logs", "metrics", "configs", "checkpoints", "training", "models",
    "diagnosis-train", "diagnosis-evaluate", "diagnosis-demo", "demo", "environment",
}
CPU_REPORTS = {
    "preprocess.json", "evaluate.json", "gpu_inference.json", "cpu_inference.json",
    "prediction_postprocess.json", "diagnosis_features.json",
}
FORBIDDEN_PARTS = {
    "data", "dataset", "raw", "cache", "prepared", "env", "venv", "site-packages",
    "private", "credentials", "secrets", "tokens", "__pycache__", "uploads", "back",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _display(value):
    return "".join(c if c.isprintable() else " " for c in str(value))


def _json(path):
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("上传清单超过 16 MiB：" + _display(path))
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("上传记录必须是 JSON 对象：" + _display(path))
    return data


def _write_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_path(path):
    """Reject traversal, symlinks and Windows junctions, including parent links."""
    path = Path(path).expanduser()
    if ".." in path.parts:
        raise ValueError("上传路径不能包含 ..：" + _display(path))
    path = path.absolute()
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("上传不接受符号链接或目录联接：" + _display(candidate))
    return path.resolve()


def _fingerprint(path):
    path = _safe_path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("上传只接受普通文件：" + _display(path))
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino


def _hash(path):
    before = _fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(BUFFER_BYTES):
            digest.update(data)
    if before != _fingerprint(path):
        raise ValueError("校验期间文件发生变化，请重新打包：" + _display(path))
    return digest.hexdigest()


def _finished(job, runtime):
    from . import inspection

    job, runtime = _safe_path(job), _safe_path(runtime)
    state = _json(_safe_path(job / "job_state.json"))
    plan = _json(_safe_path(job / "plan.json"))
    if not isinstance(plan.get("runtime"), str) or _safe_path(plan["runtime"]) != runtime:
        raise ValueError("上传必须使用原任务 plan.json 中记录的 runtime")
    if state.get("status") not in TERMINAL_SUCCESS:
        raise ValueError("仅能上传全部所选步骤已完成的任务；失败/中断任务请先续跑完成，再重新打包")
    if inspection.job_active(job):
        raise ValueError("任务仍有存活进程，请等待退出后重新上传")
    steps = state.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("任务步骤清单格式无效")
    for step in steps:
        if not isinstance(step, dict) or step.get("status") not in {*TERMINAL_SUCCESS, "reused", None}:
            raise ValueError("任务仍有未完成步骤，请先完成任务")
    return job, runtime, plan, state


def _blocked(name):
    lower = name.lower()
    return lower.startswith(".") or lower in FORBIDDEN_PARTS or any(
        word in lower for word in ("password", "credential", "secret", "token")
    )


def _artifact_file(path):
    name = path.name.lower()
    if path.parent.name == ".cpu-runtime":
        return name in CPU_REPORTS
    if _blocked(name):
        return False
    if name in ARTIFACT_NAMES:
        return True
    if path.suffix.lower() in {".pt", ".pth"}:
        return True
    if path.parent.name in {"logs", "stage_logs"} and path.suffix.lower() == ".log":
        return True
    return any(
        name == model + suffix for model in MODEL_NAMES
        for suffix in (".history.json", ".timing.json", ".environment.json", ".result.json")
    )


def _walk_artifacts(root):
    """Walk recognized output layout only; never crawl arbitrary sibling folders."""
    root = _safe_path(root)
    if not root.is_dir():
        raise ValueError("所引用的模型/结果目录不存在：" + _display(root))
    _inactive_artifacts(root)
    for child in sorted(root.iterdir()):
        if _blocked(child.name) and child.name != ".cpu-runtime":
            continue
        if child.is_symlink() or getattr(child.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("结果目录包含符号链接，无法安全打包：" + _display(child))
        if child.is_dir():
            if (
                child.name in ARTIFACT_DIRS
                or child.name == ".cpu-runtime"
                or re.fullmatch(r"fold_[0-9]+", child.name)
                or (root.name == "environment" and child.name.isdigit())
            ):
                yield from _walk_artifacts(child)
        elif _artifact_file(child):
            yield child


def _inactive_artifacts(root):
    """Referenced training directories may belong to a different runtime."""
    from .jobs import identity

    for filename in ("status.json", "report.json", "models.json", "queue.json", "active-command.json"):
        path = _safe_path(root / filename)
        if not path.is_file():
            continue
        record = _json(path)
        if record.get("status") in {"running", "queued", "waiting_gpu", "pending_cpu"}:
            raise ValueError("引用的结果/模型目录仍记录为运行中，请先确认该任务结束：" + _display(root))
        candidates = [record]
        stages = record.get("stages", {})
        if isinstance(stages, dict):
            candidates.extend(value for value in stages.values() if isinstance(value, dict))
        for candidate in candidates:
            if candidate.get("identity") and identity(candidate.get("pid")) == candidate["identity"]:
                raise ValueError("引用的结果/模型目录仍有存活进程：" + _display(root))


def _resolve_reference(raw, job, plan):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("任务的模型/结果路径记录无效")
    path = Path(raw).expanduser()
    if ".." in path.parts:
        raise ValueError("上传路径不能包含 ..：" + _display(path))
    if not path.is_absolute():
        path = job / path
    # Moved workflows retain their original absolute paths in frozen metadata.
    original = Path(plan.get("directory", str(job)))
    if original.is_absolute() and path.is_relative_to(original):
        local = job / path.relative_to(original)
        if local.exists():
            path = local
    return _safe_path(path)


def _collect(job, plan, state):
    selected = {}
    inodes = set()

    def add(path, archive_path):
        path = _safe_path(path)
        info = _fingerprint(path)
        inode = info[3:]
        if path in selected or (inode[1] and inode in inodes):
            return
        selected[path] = {"source": str(path), "archive_path": archive_path, "_fingerprint": info}
        inodes.add(inode)

    for name in sorted(ROOT_FILES):
        path = job / name
        if path.exists():
            add(path, "workflow/" + name)
    roots = []
    steps_root = _safe_path(job / "steps")
    if steps_root.is_dir():
        for folder in sorted(steps_root.iterdir()):
            if re.fullmatch(r"[0-9]+_[a-z][a-z-]*", folder.name):
                roots.append((_safe_path(folder), "workflow/steps/" + folder.name))
    external_models = []
    config_files = []
    for record in (plan, state):
        steps = record.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("任务步骤清单格式无效")
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("任务步骤清单格式无效")
            if step.get("output"):
                root = _resolve_reference(step["output"], job, plan)
                if not root.is_relative_to(job):
                    roots.append((root, "external-results/" + hashlib.sha256(str(root).encode()).hexdigest()[:12]))
            reuse = step.get("reuse", {})
            if isinstance(reuse, dict) and reuse.get("output"):
                root = _resolve_reference(reuse["output"], job, plan)
                roots.append((root, "reused-results/" + hashlib.sha256(str(root).encode()).hexdigest()[:12]))
            args = step.get("arguments", {})
            if not isinstance(args, dict):
                raise ValueError("任务参数清单格式无效")
            for key in ("models", "classifier"):
                if args.get(key):
                    external_models.append(_resolve_reference(args[key], job, plan))
            if args.get("config"):
                config_files.append(_resolve_reference(args["config"], job, plan))
    for root in sorted(set(external_models)):
        if root.is_file() and root.name in {"models.json", "classifier.json"}:
            root = root.parent
        roots.append((root, "referenced-models/" + hashlib.sha256(str(root).encode()).hexdigest()[:12]))
    for root, prefix in roots:
        for path in _walk_artifacts(root):
            add(path, prefix + "/" + path.relative_to(root).as_posix())
    for path in sorted(set(config_files)):
        if _blocked(path.name) or path.suffix.lower() not in {".json", ".yaml", ".yml", ".toml"}:
            raise ValueError("配置文件类型不在上传范围：" + _display(path))
        add(path, "referenced-configs/" + hashlib.sha256(str(path).encode()).hexdigest()[:12] + "/" + path.name)
    return list(selected.values())


class _SplitWriter:
    """A tar stream sink with one on-disk copy and incremental part hashes."""

    def __init__(self, directory, limit):
        if not 0 < limit <= 1024**3:
            raise ValueError("分卷大小必须在 1 字节到 1 GiB 之间")
        self.directory, self.limit = directory, limit
        self.stream = None
        self.assets = []
        self.size = 0
        self.digest = None

    def _finish(self):
        if self.stream:
            self.stream.close()
            self.assets.append({
                "name": Path(self.stream.name).name, "size": self.size, "sha256": self.digest.hexdigest(),
            })
            self.stream = None

    def write(self, data):
        original_size = len(data)
        view = memoryview(data)
        while view:
            if self.stream is None:
                if len(self.assets) >= MAX_ASSETS - 1:
                    raise ValueError("上传分卷超过 GitHub 每个 Release 的 1000 个附件限制")
                self.stream = (self.directory / f"artifacts.tar.part{len(self.assets) + 1:04d}").open("xb")
                self.size, self.digest = 0, hashlib.sha256()
            take = min(len(view), self.limit - self.size)
            self.stream.write(view[:take])
            self.digest.update(view[:take])
            self.size += take
            view = view[take:]
            if self.size == self.limit:
                self._finish()
        return original_size

    def close(self):
        self._finish()


class _HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data


def prepare_upload(job: Path, runtime: Path) -> Path:
    """Create an immutable archive snapshot outside the source workflow."""
    job, runtime, plan, state = _finished(job, runtime)
    uploads = _safe_path(runtime / "uploads")
    if uploads == job or uploads.is_relative_to(job):
        raise ValueError("上传暂存目录必须位于原 workflow 之外")
    print("正在收集已完成任务的日志、配置、指标与模型权重……", flush=True)
    files = _collect(job, plan, state)
    total = sum(row["_fingerprint"][0] for row in files)
    uploads.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(uploads).free < total + len(files) * 4096 + BUFFER_BYTES:
        raise ValueError(f"上传暂存空间不足：至少需要约 {total / 1024**3:.2f} GiB 加清单空间")
    package_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    package = uploads / package_id
    package.mkdir()
    writer = _SplitWriter(package, CHUNK_BYTES)
    try:
        with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for index, row in enumerate(files, 1):
                path = Path(row["source"])
                before = row["_fingerprint"]
                if _fingerprint(path) != before:
                    raise ValueError("收集后源文件发生变化，请重新打包：" + _display(path))
                print(f"打包 {index}/{len(files)}：{_display(row['archive_path'])} ({before[0]} 字节)", flush=True)
                info = tarfile.TarInfo(row["archive_path"])
                info.size, info.mtime, info.mode = before[0], before[1] // 10**9, 0o600
                with path.open("rb") as stream:
                    reader = _HashReader(stream)
                    archive.addfile(info, reader)
                if reader.size != before[0] or _fingerprint(path) != before:
                    raise ValueError("打包期间源文件发生变化，请重新打包：" + _display(path))
                row.update(size=before[0], mtime_ns=before[1], sha256=reader.digest.hexdigest())
        writer.close()
        # Recheck every source after the whole stream, not just the most recent file.
        for row in files:
            if _fingerprint(Path(row["source"])) != row.pop("_fingerprint"):
                raise ValueError("打包期间源文件发生变化，请重新打包：" + _display(row["source"]))
        _finished(job, runtime)
        manifest = {
            "schema_version": 1, "package_id": package_id, "repository": REPOSITORY,
            "release_tag": "chd-artifacts-" + package_id, "created_at": _now(),
            "job_directory": str(job), "runtime": str(runtime),
            "name": str(state.get("name") or plan.get("name") or job.name),
            "archive_format": "tar concatenated in asset-name order",
            "files": files, "assets": writer.assets, "source_bytes": total,
        }
        if len(json.dumps(manifest, ensure_ascii=False).encode("utf-8")) > 15 * 1024**2:
            raise ValueError("文件清单过大，请缩小所选任务的上传范围")
        _write_json(package / "manifest.json", manifest)
    except BaseException:
        writer.close()
        (package / "preparation-failed.txt").write_text(
            "打包未完成；此目录不可上传。修复原因后从原任务重新打包。\n", encoding="utf-8"
        )
        raise
    print("上传包已准备：" + _display(package), flush=True)
    return package


def _validate_package(package):
    package = _safe_path(package)
    manifest_file = _safe_path(package / "manifest.json")
    manifest = _json(manifest_file)
    package_id = manifest.get("package_id")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("repository") != REPOSITORY
        or not isinstance(package_id, str)
        or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}", package_id) is None
        or package.name != package_id
        or manifest.get("release_tag") != "chd-artifacts-" + package_id
    ):
        raise ValueError("上传包身份或目标仓库无效，请重新打包")
    for key in ("job_directory", "runtime"):
        if not isinstance(manifest.get(key), str):
            raise ValueError("上传包缺少原任务路径，请重新打包")
    runtime = _safe_path(manifest["runtime"])
    job = _safe_path(manifest["job_directory"])
    if package.parent != _safe_path(runtime / "uploads") or package.is_relative_to(job):
        raise ValueError("上传包不在原 runtime/uploads 中")
    assets = manifest.get("assets")
    files = manifest.get("files")
    if not isinstance(assets, list) or not 1 <= len(assets) < MAX_ASSETS or not isinstance(files, list):
        raise ValueError("上传包文件清单无效")
    for index, asset in enumerate(assets, 1):
        if (
            not isinstance(asset, dict)
            or asset.get("name") != f"artifacts.tar.part{index:04d}"
            or type(asset.get("size")) is not int
            or not 0 < asset["size"] <= 1024**3
            or not isinstance(asset.get("sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", asset["sha256"]) is None
        ):
            raise ValueError("上传分卷清单无效")
        path = _safe_path(package / asset["name"])
        if path.stat().st_size != asset["size"] or _hash(path) != asset["sha256"]:
            raise ValueError("上传分卷校验失败，请重新打包：" + asset["name"])
    for row in files:
        archive_path = row.get("archive_path") if isinstance(row, dict) else None
        if (
            not isinstance(archive_path, str) or not archive_path
            or archive_path.startswith("/") or "\\" in archive_path
            or ".." in archive_path.split("/") or ":" in archive_path
            or type(row.get("size")) is not int or row["size"] < 0
            or not isinstance(row.get("sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None
        ):
            raise ValueError("上传包源文件清单无效")
    extra = {"name": "manifest.json", "size": manifest_file.stat().st_size, "sha256": _hash(manifest_file)}
    return package, manifest, [*assets, extra]


def show_preview(package: Path) -> None:
    package, manifest, assets = _validate_package(package)
    print(f"\n待上传任务：{_display(manifest['name'])}")
    print(f"官方目标：https://github.com/{REPOSITORY}/releases")
    print("将上传日志、配置、指标、最佳和末次模型权重；不包含原始影像、缓存或环境目录。")
    for row in manifest["files"]:
        print(f"  {_display(row['archive_path'])}  {row['size']} 字节")
    print(f"合计 {len(manifest['files'])} 个文件；{len(assets)} 个附件；"
          f"{sum(a['size'] for a in assets) / 1024**3:.3f} GiB")
    print("完整清单与 SHA256：" + _display(package / "manifest.json"))
    print("失败后保留此上传包，可选择同一个包重试；开始上传前请核对清单。")


def _gh(*arguments, timeout=COMMAND_TIMEOUT):
    """Never interpolate shell text, expose stderr tokens, or use a mirror host."""
    executable = shutil.which("gh")
    if not executable:
        raise ValueError("未安装 GitHub CLI (gh)；请先安装 gh，再执行 gh auth login --hostname github.com")
    env = dict(os.environ, GH_HOST=HOST, GH_REPO=REPOSITORY, GH_PROMPT_DISABLED="1", GH_PAGER="cat")
    for key in ("GH_DEBUG", "DEBUG", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        env.pop(key, None)
    try:
        result = subprocess.run(
            [executable, *arguments], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=env, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("GitHub 操作超时；上传包已保留，请使用同一个包重试") from error
    except OSError as error:
        raise ValueError("无法启动 GitHub CLI；请检查 gh 安装后重试") from error
    if result.returncode:
        if arguments[:2] == ("auth", "status"):
            raise ValueError("GitHub 登录不可用；请执行 gh auth login --hostname github.com 后重试")
        raise ValueError("GitHub 操作失败；请检查官方 github.com 连接、登录和仓库写权限，再重试同一个包")
    return result.stdout


def _api(endpoint, *arguments, method=None):
    args = ["api", "--hostname", HOST, endpoint]
    if method:
        args += ["--method", method]
    raw = _gh(*args, *arguments)
    try:
        return json.loads(raw)
    except ValueError as error:
        raise ValueError("GitHub 返回了无效 JSON；保留上传包并重试") from error


def _release(tag):
    # Listing avoids conflating an unauthenticated/network error with a missing tag.
    page = 1
    while page <= 100:
        rows = _api(f"repos/{REPOSITORY}/releases?per_page=100&page={page}")
        if not isinstance(rows, list):
            raise ValueError("GitHub Release 列表格式无效")
        for release in rows:
            if release.get("tag_name") == tag:
                return release
        if len(rows) < 100:
            return None
        page += 1
    raise ValueError("Release 历史超过查询上限；请通过官方 GitHub 确认同名发布后重试")


def _remote_assets(release_id):
    assets = []
    for page in range(1, 12):
        rows = _api(f"repos/{REPOSITORY}/releases/{release_id}/assets?per_page=100&page={page}")
        if not isinstance(rows, list):
            raise ValueError("GitHub 附件列表格式无效")
        assets.extend(rows)
        if len(rows) < 100:
            break
    return assets


def _matches(remote, local):
    digest = remote.get("digest")
    return (
        remote.get("size") == local["size"] and remote.get("state") == "uploaded"
        and (digest is None or digest == "sha256:" + local["sha256"])
    )


def _verify_assets(remote, expected):
    by_name = {row.get("name"): row for row in remote}
    return len(by_name) == len(remote) and set(by_name) == {a["name"] for a in expected} and all(
        _matches(by_name[a["name"]], a) for a in expected
    )


def upload_package(package: Path) -> str:
    """Upload/resume one package, verify every asset, then publish its draft."""
    package, manifest, assets = _validate_package(package)
    _finished(Path(manifest["job_directory"]), Path(manifest["runtime"]))
    _gh("auth", "status", "--hostname", HOST)
    tag = manifest["release_tag"]
    marker = "CHD upload package: " + manifest["package_id"]
    release = _release(tag)
    if release is None:
        print("正在创建 GitHub 草稿发布……", flush=True)
        _gh(
            "release", "create", tag, "--repo", HOST + "/" + REPOSITORY,
            "--target", "main", "--draft", "--latest=false",
            "--title", "CHD artifacts " + manifest["package_id"],
            "--notes", marker + "\n\nSee manifest.json for file sizes and SHA256. "
            "Concatenate artifacts.tar.part* in name order to restore artifacts.tar, then extract it.",
        )
        release = _release(tag)
        if release is None:
            raise ValueError("无法确认草稿是否已创建；请保留同一个上传包重试")
    release_id = release.get("id")
    if (
        type(release_id) is not int or release_id <= 0
        or marker not in (release.get("body") or "").splitlines()
        or release.get("tag_name") != tag
    ):
        raise ValueError("同名 GitHub Release 不属于此上传包，停止上传")
    remote = _remote_assets(release_id)
    if not release.get("draft") and not _verify_assets(remote, assets):
        raise ValueError("已公开的同名 Release 附件不一致；不会覆盖，请核查原发布")
    if release.get("draft"):
        by_name = {row.get("name"): row for row in remote}
        expected = {a["name"] for a in assets}
        if len(by_name) != len(remote) or set(by_name) - expected:
            raise ValueError("草稿包含不属于此上传包的附件，停止上传")
        for index, asset in enumerate(assets, 1):
            old = by_name.get(asset["name"])
            if old and _matches(old, asset):
                print(f"已校验并复用附件 {index}/{len(assets)}：{asset['name']}", flush=True)
                continue
            # Only replace this package's incomplete/corrupt asset on its own draft.
            print(f"正在上传附件 {index}/{len(assets)}：{asset['name']} ({asset['size']} 字节)", flush=True)
            _gh(
                "release", "upload", tag, str(package / asset["name"]),
                "--repo", HOST + "/" + REPOSITORY, "--clobber", timeout=UPLOAD_TIMEOUT,
            )
        if not _verify_assets(_remote_assets(release_id), assets):
            raise ValueError("GitHub 附件大小或 SHA256 校验未通过；草稿已保留，请重试同一个包")
        _api(
            f"repos/{REPOSITORY}/releases/{release_id}", "-F", "draft=false",
            "-f", "make_latest=false", method="PATCH",
        )
    verified = _release(tag)
    if (
        not verified or verified.get("id") != release_id or verified.get("draft")
        or marker not in (verified.get("body") or "").splitlines()
        or not _verify_assets(_remote_assets(release_id), assets)
    ):
        raise ValueError("尚未确认公开发布成功；请保留同一个上传包重试")
    url = f"https://{HOST}/{REPOSITORY}/releases/tag/{tag}"
    _write_json(package / "receipt.json", {
        "schema_version": 1, "package_id": manifest["package_id"], "repository": REPOSITORY,
        "release_tag": tag, "release_id": release_id, "url": url, "verified_at": _now(), "assets": assets,
    })
    return url
