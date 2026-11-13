from datetime import datetime, timezone
import json
from pathlib import Path


def event(directory, kind, **fields):
    path = Path(directory) / "events.jsonl"
    line = json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields},
                      ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and path.stat().st_size + len(line.encode()) > 16 * 1024 ** 2:
        for index in range(3, 0, -1):
            old = path.with_suffix(f".jsonl.{index}")
            if old.exists():
                if index == 3:
                    old.unlink()
                else:
                    old.replace(path.with_suffix(f".jsonl.{index + 1}"))
        path.replace(path.with_suffix(".jsonl.1"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
    print(json.dumps({"event": kind, **fields}, ensure_ascii=False), flush=True)
