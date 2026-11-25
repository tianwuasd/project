import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4


DEFAULT_RUNTIME = Path('/data4/zhangruichen/course/chd-segmentation')


def updater(code):
    spec = importlib.util.spec_from_file_location('_chd_snapshot_updater', code / 'update.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(code, *args):
    return subprocess.run(['git', '-C', str(code), *args], check=True, capture_output=True).stdout


def pin(source):
    code = Path(source).resolve()
    if (code / '.snapshot.json').is_file():
        return code
    module = updater(code)
    runtime = Path(os.environ.get('CHD_RUNTIME_DIR', DEFAULT_RUNTIME)).resolve()
    root = runtime / 'code-snapshots'
    if runtime.is_relative_to(code) or code.is_relative_to(runtime):
        raise ValueError('代码快照过程目录必须独立于维护代码目录')
    inherited = os.environ.get('CHD_CODE_SNAPSHOT')
    if inherited:
        selected = Path(inherited).resolve()
        record = module.read_json(selected / '.snapshot.json')
        if record['source_code'] != str(code) or selected.parent.parent != root:
            raise ValueError('继承的代码快照与启动目录不一致')
        return selected
    with module.code_guard(code / 'run.py', blocking=True) as repo:
        sha = git(code, 'rev-parse', 'HEAD').decode().strip()
        scope = code.relative_to(repo).as_posix()
        names = git(repo, 'ls-files', '-z', '--', scope)
        payload = {}
        for raw in filter(None, names.split(b'\0')):
            path = repo / raw.decode('utf-8')
            name = path.relative_to(code).as_posix()
            if not module.allowed(name) or path.is_symlink() or not path.is_file():
                raise ValueError('快照代码路径无效：' + name)
            payload[name] = path.read_bytes()
        if not {'run.py', 'update.py', 'code_runtime.py', 'src/chd_segmentation/__init__.py'} <= payload.keys():
            raise ValueError('快照代码尚未完整提交')
        files = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(payload.items())}
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        version = sha + '-' + digest[:16]
        selected = root / version / 'code'
        modified = bool(git(repo, 'status', '--porcelain=v1', '--untracked-files=no', '--', scope))
        record = dict(schema=1, sha=sha, digest=digest, modified=modified, source_code=str(code), files=files)
        with module.file_lock(root / '.snapshots.lock', blocking=True):
            if selected.exists():
                if module.read_json(selected / '.snapshot.json') != record or any(
                        hashlib.sha256((selected / name).read_bytes()).hexdigest() != value for name, value in files.items()):
                    raise ValueError('代码快照内容被修改：' + str(selected))
            else:
                pending = root / ('.pending-' + uuid4().hex)
                try:
                    for name, value in payload.items():
                        destination = pending / 'code' / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(value)
                    module.write_json(pending / 'code' / '.snapshot.json', record)
                    pending.rename(selected.parent)
                finally:
                    if pending.exists():
                        shutil.rmtree(pending)
        return selected


def activate_package(namespace):
    code = Path(namespace['__file__']).resolve().parents[2]
    if not (code / 'update.py').is_file():
        return
    selected = pin(code)
    package = selected / 'src' / 'chd_segmentation'
    namespace['__path__'][:] = [str(package)]
    namespace['__file__'] = str(package / '__init__.py')
    spec = importlib.util.spec_from_file_location(namespace['__name__'], package / '__init__.py',
                                                submodule_search_locations=[str(package)])
    namespace['__spec__'], namespace['__loader__'] = spec, spec.loader
    sys.path.insert(0, str(selected / 'src'))
    os.environ['CHD_CODE_SNAPSHOT'] = str(selected)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    if selected != code:
        exec(compile((package / '__init__.py').read_bytes(), str(package / '__init__.py'), 'exec'), namespace)


def info(code):
    record = json.loads((code / '.snapshot.json').read_text(encoding='utf-8'))
    return {key: record[key] for key in ('sha', 'modified', 'source_code', 'digest')} | dict(code=str(code), python=sys.executable)
