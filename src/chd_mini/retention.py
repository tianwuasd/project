from pathlib import Path

from .contracts import confined, file_sha, write_json


def cap_logs(directory, limit=20 * 1024 ** 2, reserve=1024 ** 2):
    directory = Path(directory).resolve(strict=True)
    if not 0 < reserve < limit:
        raise ValueError("日志回执预留必须位于容量上限以内")
    protected = {"summary.json", "ownership.json", "cleanup.json", "processes.json", "retention.json"}
    paths = [confined(p, directory) for p in directory.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in paths)
    records = []
    for path in sorted(paths, key=lambda p: p.stat().st_size, reverse=True):
        if total <= limit - reserve:
            break
        if path.name in protected:
            continue
        original = path.stat().st_size
        keep = max(0, original - (total - (limit - reserve)))
        fingerprint = file_sha(path)
        destination = path.with_name(path.name + ".truncated.log")
        with path.open("rb") as stream:
            first = stream.read(keep // 2)
            stream.seek(original - (keep - len(first)))
            last = stream.read(keep - len(first))
        destination.write_bytes(first + last)
        path.unlink()
        total -= original - keep
        records.append({"path": str(path.relative_to(directory)), "retained": str(destination.relative_to(directory)),
                        "original_bytes": original, "retained_bytes": keep, "original_sha256": fingerprint})
    receipt = {"limit": limit, "removed_bytes": sum(r["original_bytes"] - r["retained_bytes"] for r in records),
               "truncated": records}
    write_json(directory / "retention.json", receipt)
    final = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    if final > limit:
        raise RuntimeError("必要诊断记录超出日志容量，清理状态保持未完成")
    return receipt
