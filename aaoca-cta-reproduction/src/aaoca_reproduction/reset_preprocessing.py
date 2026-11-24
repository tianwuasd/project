from pathlib import Path

from .preprocessing_batch import batch_lock
from .labels import validate_manifest


ALLOWED_CASE_FILES = {
    "normalized.nii.gz",
    "normalized.pending.nii.gz",
    "training_xyz.npy",
    "training_xyz.pending.npy",
    "classification.json",
    "classification.json.pending",
    "localization-skipped.json",
    "localization-skipped.json.pending",
}


def _inspect(root, data_root, manifest):
    if root.is_symlink() or not root.is_relative_to(data_root):
        raise ValueError("分类预处理目录必须是data_root内的普通目录")
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError("分类预处理路径不是目录")
    allowed_root = {".batch.lock"}
    case_ids = {str(case["case_id"]) for case in manifest}
    entries = list(root.iterdir())
    for entry in entries:
        if entry.name in allowed_root:
            if entry.is_symlink() or not entry.is_file():
                raise ValueError("锁路径不是普通文件")
            continue
        if entry.name not in case_ids or entry.is_symlink() or not entry.is_dir():
            raise ValueError("预处理目录含未知内容：" + str(entry))
        for child in entry.iterdir():
            if child.name not in ALLOWED_CASE_FILES or child.is_symlink() or not child.is_file():
                raise ValueError("病例目录含未知内容：" + str(child))
    return entries


def reset_preprocessing(config, manifest, confirm_root=None, check=True):
    manifest = validate_manifest(manifest)
    data_root = Path(config["data_root"]).resolve()
    root = data_root / "aaoca-classification"
    if root.is_symlink() or not root.is_relative_to(data_root):
        raise ValueError("分类预处理目录必须位于data_root内且不含软链接")
    raw_root = Path(config["data_root"])
    if raw_root.is_symlink():
        raise ValueError("data_root不能是软链接")
    sources = [Path(case[key]).resolve() for case in manifest for key in ("image", "label")]
    if any(source.is_relative_to(root) or root.is_relative_to(source.parent) for source in sources):
        raise ValueError("分类预处理目录与原始CT或分割根目录冲突")
    entries = _inspect(root, data_root, manifest)
    files = [child for entry in entries if entry.name != ".batch.lock" for child in entry.iterdir()]
    if check:
        return {"root": str(root), "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                "case_directories": sum(entry.name != ".batch.lock" for entry in entries),
                "lock_preserved": (root / ".batch.lock").exists(), "check_only": True}
    if confirm_root is None or Path(confirm_root).resolve() != root:
        raise ValueError("删除前必须提供与预处理目录完全相同的confirm_root")
    if not root.exists():
        return {"root": str(root), "files": 0, "bytes": 0, "case_directories": 0,
                "lock_preserved": False, "check_only": False}
    deleted = 0
    deleted_bytes = 0
    with batch_lock(root / ".batch.lock"):
        entries = _inspect(root, data_root, manifest)
        for entry in entries:
            if entry.name == ".batch.lock":
                continue
            for child in list(entry.iterdir()):
                deleted_bytes += child.stat().st_size
                child.unlink()
                deleted += 1
            entry.rmdir()
    return {"root": str(root), "files": deleted, "bytes": deleted_bytes,
            "case_directories": sum(entry.name != ".batch.lock" for entry in entries),
            "lock_preserved": (root / ".batch.lock").exists(), "check_only": False}
