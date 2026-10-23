"""Discovery and durable local records; no patient metadata is embedded in code."""
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path, check=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            if check:
                check()
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class RunLock(AbstractContextManager):
    """OS lock adapted from patient_pipeline.runtime; file existence is not a lock."""
    def __init__(self, output):
        self.path = Path(output) / '.preprocess.lock'
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open('a+b')
        try:
            if self.stream.tell() == 0:
                self.stream.write(b'0')
                self.stream.flush()
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError('output is locked by another process or filesystem locking is unavailable') from exc
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.close()  # releases the OS lock, including after a process exits
            self.stream = None


def derived_directory(name):
    lower = name.lower()
    return (name.startswith('.') or name.startswith('研究版')
            or lower in {'preprocessed', 'preprocessing', 'aaoca_preprocessed', 'results',
                         'labels', 'masks', 'segmentations', '__pycache__'})


def image_name(name):
    lower = name.lower()
    return lower.endswith(('.nii', '.nii.gz')) and not re.search(
        r'(^|[_\-.])(label|labels|mask|seg|segmentation)([_\-.]|$)', lower)


def discover(input_root, manifest=None):
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'input directory is missing: {root}')
    if manifest is None and (root / 'CT总清单.json').is_file():
        manifest = root / 'CT总清单.json'
    selected = []
    if manifest is not None:
        rows = read_json(manifest)
        if not isinstance(rows, list):
            raise ValueError('manifest must be a JSON array')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('manifest entries must be objects')
            if row.get('type', 'AAOCA') != 'AAOCA':
                continue
            relative = str(row.get('input', row.get('destination', ''))).replace('\\', '/')
            parts = PurePosixPath(relative)
            if not relative or parts.is_absolute() or '..' in parts.parts or ':' in relative:
                raise ValueError('manifest paths must be relative and remain inside input root')
            digest = row.get('sha256')
            if digest is not None and not re.fullmatch(r'[0-9a-fA-F]{64}', str(digest)):
                raise ValueError('manifest sha256 must be a 64-character hexadecimal hash')
            expected_stat = None
            if 'size' in row or 'mtime_ns' in row:
                if any(isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 0 for key in ['size', 'mtime_ns']):
                    raise ValueError('frozen source size and mtime_ns must be nonnegative integers')
                expected_stat = [row['size'], row['mtime_ns']]
            selected.append((root / relative, str(digest).lower() if digest is not None else None, expected_stat))
    else:
        def discovery_error(error):
            raise error
        for directory, folders, files in os.walk(root, followlinks=False, onerror=discovery_error):
            folders[:] = sorted(name for name in folders
                                if not derived_directory(name) and not (Path(directory) / name).is_symlink())
            selected.extend((Path(directory) / name, None, None) for name in sorted(files) if image_name(name))
    cases, seen = [], set()
    for path, digest, expected_stat in selected:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('source path escapes input root')
        relative = path.relative_to(root).as_posix()
        if not image_name(path.name) or any(derived_directory(p) for p in Path(relative).parts[:-1]):
            raise ValueError('manifest contains a label or derived image; select native HU CT only')
        if resolved in seen:
            raise ValueError('duplicate source in input manifest')
        seen.add(resolved)
        cases.append({'case_id': hashlib.sha256(relative.encode('utf-8')).hexdigest()[:24],
                      'input': relative, 'source': resolved, 'expected_sha256': digest, 'expected_stat': expected_stat})
    if not cases:
        raise ValueError('no native AAOCA NIfTI CT images found')
    return sorted(cases, key=lambda case: case['input'])
