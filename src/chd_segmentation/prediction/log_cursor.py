from pathlib import Path

from filelock import FileLock


def start_event_range(directory):
    path = (Path(directory)/'events.jsonl').resolve()
    with FileLock(str(path.with_suffix('.lock')),timeout=30):
        stat = path.stat() if path.exists() else None
    return path,0 if stat is None else stat.st_size,None if stat is None else (stat.st_dev,stat.st_ino)


def finish_event_range(cursor):
    path,start,identity = cursor
    with FileLock(str(path.with_suffix('.lock')),timeout=30):
        stat = path.stat()
    complete = identity is None or identity==(stat.st_dev,stat.st_ino)
    return dict(event_log=str(path),start_offset=start if complete else None,
        end_offset=stat.st_size,event_log_range_complete=complete)
