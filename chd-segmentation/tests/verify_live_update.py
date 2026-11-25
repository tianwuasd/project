import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from time import monotonic, sleep
from unittest import TestCase


PROJECT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args, data=None):
    return subprocess.run(['git', '-C', str(root), '-c', 'user.name=Codex', '-c', 'user.email=codex@local',
                           *args], input=data, check=True, capture_output=True, timeout=60).stdout


def worker(root):
    import chd_segmentation
    from chd_segmentation.artifacts.store import operation_lock
    code = Path(chd_segmentation.__file__).parents[2]
    runtime = load(code / 'code_runtime.py', 'worker_runtime')
    update = load(code / 'update.py', 'worker_update')
    with operation_lock(root / 'course' / 'runs' / 'active'):
        update.write_json(root / 'ready.json', runtime.info(code))
        deadline = monotonic() + 600
        while not (root / 'continue').exists():
            if monotonic() >= deadline:
                raise TimeoutError('验证控制进程未继续')
            sleep(.1)
        # 更新后才导入实际模块，检验延迟导入与子进程均沿用启动快照。
        from chd_segmentation.models import q_imagechd_setup
        path = Path(q_imagechd_setup.__file__)
        env = dict(os.environ, PYTHONPATH=str(root / 'code' / 'chd-segmentation' / 'src'))
        child = subprocess.run([sys.executable, '-B', str(root / 'code' / 'chd-segmentation' / 'run.py'), '--code-info'],
                               env=env, check=True, capture_output=True, timeout=30)
        update.write_json(root / 'old-result.json', dict(code=str(code), module=str(path),
            module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), child=json.loads(child.stdout)))


def run(root, report):
    update = load(PROJECT / 'update.py', 'update_under_test')
    case, started = TestCase(), monotonic()
    root, report = root.resolve(), report.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / '.owner').write_text('live-update-code-only', encoding='utf-8')
    (root / '.gitignore').write_text('*\n', encoding='utf-8')
    processes, checks, timings = [], [], {}
    result = dict(state='running', scope='two real source revisions; code-only; zero model updates',
                  ram_peak_bytes=None, vram_peak_bytes=None)
    try:
        origin, target, course = root / 'origin', root / 'code', root / 'course'
        origin.mkdir()
        git(origin, 'init', '--bare', '--initial-branch=main')
        paths = ['.gitattributes', '.gitignore', 'README.md', 'pyproject.toml', 'setup_env.py',
                 'update.py', 'run.py', 'code_runtime.py', 'src/chd_segmentation/__init__.py',
                 'src/chd_segmentation/models/__init__.py', 'src/chd_segmentation/models/q_imagechd_setup.py',
                 'src/chd_segmentation/artifacts/__init__.py', 'src/chd_segmentation/artifacts/store.py',
                 'src/chd_segmentation/artifacts/cache.py']
        payload = {name: (PROJECT / name).read_bytes() for name in paths}
        def tree(values):
            def build(prefix):
                direct, directories = {}, set()
                for name, blob in values.items():
                    if name.startswith(prefix):
                        leaf, sep, _ = name[len(prefix):].partition('/')
                        if sep:
                            directories.add(leaf)
                        else:
                            direct[leaf] = '100644 blob ' + blob
                direct.update({name: '040000 tree ' + build(prefix + name + '/') for name in directories})
                data = b''.join((value + '\t' + name).encode() + b'\0' for name, value in sorted(direct.items()))
                return git(origin, 'mktree', '-z', data=data).decode().strip()
            return build('')
        blobs = {name: git(origin, 'hash-object', '-w', '--stdin', data=data).decode().strip() for name, data in payload.items()}
        q_path = 'src/chd_segmentation/models/q_imagechd_setup.py'
        original_q = git(PROJECT, 'show', '66fb59cd194e30e36a1bd829330f037049fb6221:' + q_path)
        old_blobs = dict(blobs, **{q_path: git(origin, 'hash-object', '-w', '--stdin', data=original_q).decode().strip()})
        def commit(values, parent=None):
            code_tree = tree(values)
            repository_tree = git(origin, 'mktree', data=('040000 tree ' + code_tree + '\tchd-segmentation\n').encode()).decode().strip()
            return git(origin, 'commit-tree', repository_tree, *(['-p', parent] if parent else []),
                       data=b'Actual CHD source revision\n').decode().strip()
        first, second = commit(old_blobs), None
        second = commit(blobs, first)
        git(origin, 'update-ref', 'refs/heads/main', first)
        subprocess.run(['git', 'clone', '--no-checkout', '--branch', 'main', str(origin), str(target)],
                       check=True, capture_output=True, timeout=60)
        repo = update.GitRepository(target, expected_urls={str(origin)})
        update.synchronize(repo, course, first, force=True)
        code = target / 'chd-segmentation'
        env = dict(os.environ, CHD_RUNTIME_DIR=str(course), PYTHONPATH=str(code / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
        env.pop('CHD_CODE_SNAPSHOT', None)
        mark = monotonic()
        command = [sys.executable, '-B', str(code / 'run.py'), '--code-info']
        first_info = json.loads(subprocess.run(command, env=env, check=True, capture_output=True, timeout=60).stdout)
        again = json.loads(subprocess.run(command, env=env, check=True, capture_output=True, timeout=60).stdout)
        case.assertEqual(first_info, again)
        case.assertEqual(first_info['sha'], first)
        case.assertFalse(first_info['modified'])
        timings['initial_snapshot_seconds'] = monotonic() - mark
        checks.append('same revision reuses one verified snapshot in course')
        old = subprocess.Popen([sys.executable, '-B', str(__file__), 'worker', str(root)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(old)
        deadline = monotonic() + 30
        while not (root / 'ready.json').exists():
            if old.poll() is not None:
                raise RuntimeError(old.communicate()[1].decode())
            if monotonic() >= deadline:
                raise TimeoutError('旧任务未就绪')
            sleep(.1)
        from filelock import FileLock, Timeout
        with case.assertRaises(Timeout):
            with FileLock(str(course / 'runs' / 'active' / '.run.lock'), timeout=0):
                pass
        checks.append('duplicate task remains excluded by actual run lock')
        git(origin, 'update-ref', 'refs/heads/main', second)
        with update.file_lock(course / '.code.lock'):
            with case.assertRaises(OSError):
                update.synchronize(repo, course, second, force=True)
        checks.append('two simultaneous updaters remain mutually exclusive')
        with update.code_guard(code / 'run.py'):
            with case.assertRaises(OSError):
                update.synchronize(repo, course, second, force=True)
        checks.append('legacy code reader still blocks unsafe initial migration')
        mark = monotonic()
        updated = update.synchronize(repo, course, second, force=True)
        timings['update_during_active_task_seconds'] = monotonic() - mark
        case.assertEqual(updated['sha'], second)
        new_info = json.loads(subprocess.run(command, env=env, check=True, capture_output=True, timeout=60).stdout)
        case.assertEqual(new_info['sha'], second)
        case.assertNotEqual(new_info['code'], first_info['code'])
        (root / 'continue').touch()
        _, error = old.communicate(timeout=45)
        case.assertEqual(old.returncode, 0, error.decode())
        old_result = update.read_json(root / 'old-result.json')
        case.assertEqual(old_result['code'], first_info['code'])
        case.assertEqual(old_result['module_sha256'], hashlib.sha256(original_q).hexdigest())
        case.assertEqual(old_result['child']['sha'], first)
        case.assertEqual(old_result['child']['code'], first_info['code'])
        checks.append('old task delayed imports and inherited child stay on old source after real Git update')
        checks.append('new independent task uses new source and Git HEAD/origin/main match')
        case.assertEqual(git(target, 'rev-parse', 'HEAD'), git(target, 'rev-parse', 'origin/main'))
        case.assertFalse(git(target, 'status', '--porcelain=v1', '--', 'chd-segmentation'))
        case.assertEqual(len(list((course / 'code-snapshots').iterdir())), 3)
        case.assertFalse(any((course / 'code-snapshots').glob('.pending-*')))
        for snapshot in (Path(first_info['code']), Path(new_info['code'])):
            case.assertFalse(list(snapshot.rglob('__pycache__')))
            case.assertFalse((snapshot / '.git').exists())
        checks.append('snapshots contain code only; no bytecode, repository metadata or duplicate per-process copies')
        result.update(state='passed', revisions=[first, second], selected_source_files=paths, first=first_info, second=new_info,
                      old_task=old_result)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)
        if sys.exception() is not None:
            result.update(state='failed', error=str(sys.exception()))
        files = [path for path in root.rglob('*') if path.is_file() and not path.is_symlink()]
        count, size = len(files), sum(path.stat().st_size for path in files)
        result.update(seconds=monotonic()-started, timings=timings, checks=checks,
                      file_count=count, bytes=size, snapshot_count=2,
                      artifact_root=str(root), artifact_files=[str(p.relative_to(root)) for p in files])
        update.write_json(report, result)
        if root != Path(os.environ['CHD_LIVE_TEST_ROOT']).resolve() or root.is_symlink() or (root / '.owner').read_text() != 'live-update-code-only':
            raise ValueError('测试目录归属变化，停止清理')
        for path in files:
            path.chmod(stat.S_IREAD | stat.S_IWRITE)
        shutil.rmtree(root)
        result['cleanup'] = dict(deleted_files=count, released_bytes=size, root_exists=root.exists())
        update.write_json(report, result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if sys.argv[1:2] == ['worker']:
        worker(Path(sys.argv[2]))
    else:
        run(Path(os.environ['CHD_LIVE_TEST_ROOT']), Path(os.environ['CHD_LIVE_TEST_REPORT']))
