"""Small CPU-only verification acceptance case, without CT data or ML packages."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="校验读取快速验收：合成数据，不训练、不用 GPU")
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chd-ct-reproduction/src"))
    from chd_ct.server.verification import verify_snapshots

    args.work_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="resume-read-", dir=args.work_root))
    cache = run / "cache"
    cache.mkdir()
    expected, before = {}, {}
    for i in range(6):
        path = cache / f"case{i}.bin"
        content = bytes([i]) * (8 * 1024**2)
        path.write_bytes(content)
        expected[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        before[path] = path.stat().st_mtime_ns
    record = dict(inputs=expected, outputs={}, input_roots=[str(cache)], output_roots=[])
    start = time.monotonic()
    result = verify_snapshots(
        [(name, record) for name in ("preprocess", "smoke", "preflight", "train", "test")]
    )
    result["seconds"] = time.monotonic() - start
    assert result["files"] == 6 and result["references"] == 30 and result["bytes"] == 48 * 1024**2
    for path, mtime in before.items():
        assert path.stat().st_mtime_ns == mtime
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[str(path.resolve())]
    path = cache / "case0.bin"
    stamp = path.stat()
    with path.open("r+b") as stream:
        stream.write(b"changed")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    try:
        verify_snapshots([("intentional-change", record)], show_progress=False)
    except ValueError:
        result["changed_content_rejected"] = True
    else:
        raise AssertionError("修改内容后校验未拒绝")
    result["status"] = "passed"
    (run / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS：6 个文件跨 5 步只读取 48 MiB，内容修改能被发现，旧文件校验时保持原样。\n报告：{run}")
    print("耗时仅代表本次合成读取；不代表正式 NAS 任务的加速倍数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
