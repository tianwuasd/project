from pathlib import Path, PurePosixPath
from contextlib import ExitStack
import shutil

from .contracts import atomic_json, digest, read_json, sha256_file
from .observability.status import operation_lock, lock_active


def cache_lock_home(root):
    root = Path(root).absolute()
    home = root.parent/'.runtime'/('ac-'+digest(str(root))[:24])
    if home.resolve() != home:
        raise ValueError('Cache lock directory cannot contain links')
    return home


def _home(directory):
    directory = Path(directory).absolute()
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError('Retention requires an existing local directory without links')
    if (directory/'.runtime').resolve() != directory/'.runtime':
        raise ValueError('Retention runtime directory cannot contain links')
    return directory


def _branch_array(name):
    path = PurePosixPath(name)
    return (len(path.parts) == 3 and path.parts[0] == 'branches'
            and all(part not in ('.', '..') for part in path.parts)
            and path.suffix == '.npy' and path.as_posix() == name)


def _scan(directory):
    files = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if relative.parts[0] in ('.runtime', '.input-cache'):
            continue
        if path.resolve() != path:
            raise ValueError('Unexpected link in retained stage output')
        if path.is_file():
            files[relative.as_posix()] = sha256_file(path)
    return files


def validate_retained_files(directory, expected_files, required):
    directory = _home(directory)
    for name in required:
        if not (directory/name).is_file():
            raise ValueError('Missing required output: '+str(directory/name))
    released = {}
    ledger = directory/'.runtime/retention.json'
    if ledger.exists():
        if ledger.resolve() != ledger:
            raise ValueError('Retention ledger cannot be a link')
        record = read_json(ledger)
        if record.get('schema') != 1 or record.get('receipt_digest') != digest(expected_files):
            raise ValueError('Retention ledger does not match original receipt')
        released = record['released']
        if any(not _branch_array(name) or name in required
               or expected_files.get(name) != checksum for name, checksum in released.items()):
            raise ValueError('Invalid released prediction cache entry')
    actual = _scan(directory)
    if any(expected_files.get(name) != checksum for name, checksum in actual.items()):
        raise ValueError('Stage artifacts changed')
    missing = set(expected_files)-set(actual)
    if not missing <= set(released):
        raise ValueError('Stage artifacts changed or missing without retention record')
    return dict(expected_files)


def release_branch_arrays(directory, expected_files, required=()):
    directory = _home(directory)
    validate_retained_files(directory, expected_files, required)
    ledger = directory/'.runtime/retention.json'
    released = dict(read_json(ledger)['released']) if ledger.exists() else {}
    selected = {name: checksum for name, checksum in expected_files.items()
                if _branch_array(name) and name not in required}
    released.update(selected)
    # 先记录已验证文件，再删除；进程终止后可继续完成相同清理。
    atomic_json(directory/'.runtime/retention.json', dict(
        schema=1, receipt_digest=digest(expected_files), released=released))
    removed = 0
    for name in selected:
        path = directory/name
        if path.exists():
            removed += path.stat().st_size
            path.unlink()
    return dict(files=len(selected), removed_bytes=removed)


def release_input_cache(directory):
    directory = _home(directory)
    root = directory/'.input-cache'
    with operation_lock(cache_lock_home(root), 'release-input-cache'):
        leases = cache_lock_home(root)/'leases'
        if leases.exists() and any(lock_active(path) for path in leases.glob('*/.runtime/operation.lock')):
            raise RuntimeError('Input cache has active readers or builders')
        if not root.exists():
            return dict(removed_bytes=0)
        if root.resolve() != root:
            raise ValueError('Input cache cannot contain links')
        paths = list(root.rglob('*'))
        if any(path.resolve() != path for path in paths):
            raise ValueError('Input cache cannot contain links')
        removed = sum(path.stat().st_size for path in paths if path.is_file())
        shutil.rmtree(root)
        return dict(removed_bytes=removed)


def release_shared_prediction_cache(root):
    root = Path(root).absolute()
    if root.name != 'prediction-cache':
        raise ValueError('Expected the workspace prediction-cache directory')
    if not root.exists():
        return dict(entries=0, removed_bytes=0)
    root = _home(root)
    entries = root/'entries'
    if not entries.exists():
        return dict(entries=0, removed_bytes=0)
    if entries.resolve() != entries:
        raise ValueError('Shared prediction entries cannot contain links')
    with ExitStack() as locks:
        locks.enter_context(operation_lock(root, 'release-prediction-cache'))
        planned = []
        for entry in sorted(entries.iterdir()):
            key = entry.name
            if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('Unknown or unfinished shared prediction entry')
            locks.enter_context(operation_lock(root/'locks'/key, 'release-prediction-entry'))
            if entry.resolve() != entry or not entry.is_dir():
                raise ValueError('Shared prediction entry cannot contain links')
            manifest = read_json(entry/'manifest.json')
            record = manifest['record']
            if manifest['digest'] != digest(record) or digest(record['identity']) != key:
                raise ValueError('Shared prediction identity changed')
            files = record['files']
            if read_json(entry/'complete.json') != dict(
                    checkpoint=record['identity']['checkpoint'], files=files):
                raise ValueError('Shared prediction completion marker changed')
            if any(PurePosixPath(name).name != name or '\\' in name or name in ('.', '..')
                   for name in files):
                raise ValueError('Invalid shared prediction case name')
            expected = {name+'.npy': checksum for name, checksum in files.items()}
            actual = _scan(entry)
            if set(actual) != set(expected)|{'manifest.json', 'complete.json'}:
                raise ValueError('Unexpected shared prediction files')
            if any(actual[name] != checksum for name, checksum in expected.items()):
                raise ValueError('Shared prediction array changed')
            planned.append((entry, sum(path.stat().st_size for path in entry.iterdir())))
        # 所有缓存均验证完成后才开始删除，保留工作区外的源文件。
        for entry, size in planned:
            shutil.rmtree(entry)
        return dict(entries=len(planned), removed_bytes=sum(size for _, size in planned))
