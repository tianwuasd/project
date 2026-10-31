"""Snapshot finished VISTA3D artifacts for publication through SSH Git and Git LFS.

Only standard-library dependencies are used. The caller must hold the workflow's
runtime lock throughout preparation/upload and obtain consent after the preview.
No public Git download mirror, gh login, API token, or training import is used.
"""

import hashlib
import json
import os
import re
import shutil
import stat
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
    if os.environ.get('VISTA_CATALOG_VERSION')=='2':
        from pipeline_catalog import resolve
        job=resolve(runtime,job)
    from task_runtime import load_plan
    plan = load_plan(job)
    state = _json(_safe_path(job / "state.json"))
    if _safe_path(plan["runtime"]) != runtime:
        raise ValueError("上传必须使用原任务 runtime")
    if state.get("status") != "completed" or inspection.job_active(job):
        raise ValueError("仅能上传全部步骤完成且没有活动进程的任务")
    if any(step.get("status") != "completed" for step in state.get("steps",{}).values()):
        raise ValueError("任务仍有未完成步骤")
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
    """Vista adapter: fixed output roots only; never visit raw/cache/predicted images."""
    selected=[]
    def add(path, name):
        path=_safe_path(path)
        if path.is_file():
            selected.append(dict(source=str(path),archive_path=name,_fingerprint=_fingerprint(path)))
    for name in ('plan.json','state.json','code_manifest.json','launcher.log','environment.json','resource-plan.json','software-smoke.json','resume-check.json'):
        add(job/name,'workflow/'+name)
    work=_safe_path(plan['workspace'])
    if not work.is_relative_to(job):
        raise ValueError('任务工作区不在本任务目录内')
    roots=[job/'logs',job/'receipts',work/'06_configs',work/'09_reports',work/'00_admin/preflight',work/'00_admin/preparation_logs',job/'diagnostics',work/'diagnostics',
           work/'00_admin/stage-decisions',work/'00_admin/case-observations',work/'00_admin/preparation_progress',
           work/'07_experiments/tuning',work/'07_experiments/optimization',work/'08_results/evaluation/queue_logs']
    for name in ['server_config.json','model_selection.json','evaluation_protocol.json','resource-plan.json','cpu-case-budget.json','prediction-host-budget.json','assessment-progress.json']:
        add(work/'00_admin'/name,'workflow/workspace/00_admin/'+name)
    # Recovery archives and unfinished smoke attempts are historical failure scenes.
    experiments=work/'07_experiments'
    for root in sorted(experiments.glob('formal/*'))+sorted(experiments.glob('smoke_*')):
        record=root/'run.json'
        if not record.is_file() or _json(record).get('status') not in TERMINAL_SUCCESS:continue
        _inactive_artifacts(root);roots.append(root)
    suffixes={'.json','.jsonl','.csv','.log','.txt','.md','.yaml','.yml','.pt','.pth'}
    for root in roots:
        if not root.exists():continue
        root=_safe_path(root)
        for path in sorted(root.rglob('*')):
            _safe_path(path)
            if path.suffix.lower() in {'.pt','.pth'} and not root.is_relative_to(experiments/'formal'):continue
            if path.is_file() and path.suffix.lower() in suffixes and not any(_blocked(p) for p in path.relative_to(root).parts):
                add(path,'workflow/'+path.relative_to(job).as_posix())
    return selected


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
                    raise ValueError("上传分卷超过本工具的 1000 个文件上限")
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
            "release_tag": "vista3d-artifacts-" + package_id, "created_at": _now(),
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
        or manifest.get("release_tag") != "vista3d-artifacts-" + package_id
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
    print(f"官方目标：https://github.com/{REPOSITORY}/tree/vista3d-artifacts/{manifest['package_id']}")
    print("认证：zmic网络设置中的SSH密钥；分卷用Git LFS（需要Git LFS与可用额度，无需gh）。")
    print("将上传日志、配置、指标、最佳和末次模型权重；不包含原始影像、缓存或环境目录。")
    for row in manifest["files"]:
        print(f"  {_display(row['archive_path'])}  {row['size']} 字节")
    print(f"合计 {len(manifest['files'])} 个文件；{len(assets)} 个上传文件；"
          f"{sum(a['size'] for a in assets) / 1024**3:.3f} GiB")
    print("完整清单与 SHA256：" + _display(package / "manifest.json"))
    print("失败后保留此上传包，可选择同一个包重试；开始上传前请核对清单。")


def upload_package(package: Path) -> str:
    """Publish/retry the same package using its runtime SSH configuration."""
    package, manifest, assets = _validate_package(package)
    from . import network

    settings = network.load(Path(manifest["runtime"]))
    network.show(settings)
    with network.use(settings):
        return _publish(package, manifest, assets)


def _publish(package, manifest, assets):
    from .ssh_upload import publish

    return publish(package, manifest, assets)
