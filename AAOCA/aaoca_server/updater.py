"""Ordered public mirrors and isolated releases; never replace a shared checkout."""
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import uuid

from . import mirrors
from .state import dispatch_lock, now, read, separate, write

CODE_ROOT = Path(__file__).resolve().parent.parent
RELEASE_FILES = ('aaoca_preprocessing', 'aaoca_server', 'start_aaoca_server.py',
                 'start_aaoca_server.sh', 'preprocess_aaoca.py', 'setup_aaoca_environment.py',
                 'pyproject.toml', 'requirements-preprocess.txt', 'PREPROCESSING.md',
                 'PREPROCESSING_WORKFLOW.md', 'SERVER_WORKBENCH.md')


def command(args, *, cwd=None, env=None, timeout=45):
    """Reap only this command's process group before releasing caller locks."""
    child = None
    try:
        interrupted = []
        can_defer = threading.current_thread() is threading.main_thread()
        previous_handler = signal.getsignal(signal.SIGINT)
        if can_defer:
            signal.signal(signal.SIGINT, lambda *_: interrupted.append(True))
        try:
            child = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     start_new_session=os.name != 'nt')
        finally:
            if can_defer:
                signal.signal(signal.SIGINT, previous_handler)
        if interrupted:
            raise KeyboardInterrupt
        stdout, stderr = child.communicate(timeout=timeout)
    except BaseException:
        if child is not None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            else:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                child.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    child.kill()
                child.communicate(timeout=3)
            if os.name != 'nt':
                # A helper can close the pipes and ignore TERM: communicate()
                # only reaps the leader, so also finish our remaining group.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        raise
    if child.returncode:
        raise ValueError(stderr.decode('utf-8', 'replace')[-1000:] or f'command failed ({child.returncode})')
    return stdout


def git(root, *args, timeout=30):
    # Network runs ONLY in our fresh private bare repository, which contains no
    # user auth headers, credential helpers, URL rewrites or inherited Git config.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    return command(['git', '-c', 'credential.helper=', '-c', 'http.extraHeader=',
                    '-c', 'http.version=HTTP/1.1', '-C', str(root), *args], env=env, timeout=timeout)


def extract_release(data, destination):
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive.getmembers():
            path = PurePosixPath(item.name)
            if path == PurePosixPath('AAOCA'):
                continue
            if (path.is_absolute() or '..' in path.parts or ':' in item.name or '\\' in item.name
                    or not path.parts or path.parts[0] != 'AAOCA' or len(path.parts) < 2
                    or path.parts[1] not in RELEASE_FILES or not (item.isfile() or item.isdir())):
                raise ValueError('发布归档包含不安全路径或非普通文件')
            target = destination.joinpath(*path.parts[1:])
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    for required in ('start_aaoca_server.py', 'aaoca_server/panel.py', 'aaoca_server/updater.py',
                     'aaoca_preprocessing/jobs.py', 'pyproject.toml'):
        if not (destination / required).is_file():
            raise ValueError('目标版本缺少兼容入口：' + required)
    # Compile only, do not execute downloaded code during validation.
    for source in destination.rglob('*.py'):
        compile(source.read_bytes(), str(source), 'exec')


def managed_release(runtime):
    record = read(runtime / 'release.json', {})
    if not record:
        return None
    commit = record.get('commit', '')
    path = Path(record.get('path', '')).resolve()
    if (not re.fullmatch(r'[0-9a-f]{40}', commit)
            or path != (runtime / 'releases' / commit).resolve()
            or not (path / 'aaoca_server/updater.py').is_file()
            or not (path / 'start_aaoca_server.py').is_file()):
        raise ValueError('发布指针无效，请保留 release.json 并检查发布目录')
    return path


def resume_saved(entry):
    root = Path(entry.get('code_root', CODE_ROOT))
    python = entry.get('python', sys.executable)
    if not (root / 'aaoca_preprocessing/jobs.py').is_file() or not Path(python).is_file():
        raise ValueError('历史任务的代码或 Python 环境已不存在；请恢复原目录')
    script = 'import sys; sys.path.insert(0, sys.argv[1]); from aaoca_preprocessing.jobs import resume; print(resume(sys.argv[2]))'
    return command([python, '-c', script, str(root), entry['output']], timeout=60).decode('utf-8', 'replace')


def ensure_current(runtime):
    release = managed_release(runtime)
    if release is not None and release != CODE_ROOT:
        raise ValueError('其他终端已更新，请退出并重新打开面板后再操作')


def same_release(existing, staging):
    """Recover a publication interrupted between rename and pointer replacement."""
    def files(root):
        result = {}
        for path in root.rglob('*'):
            if path.is_symlink():
                return None
            if path.is_file():
                result[path.relative_to(root)] = path.read_bytes()
        return result
    return not existing.is_symlink() and files(existing) == files(staging)


def update(registry, emit=print):
    settings = mirrors.load(registry.root)
    previous = read(registry.root / 'release.json', {})
    checkout = Path(previous.get('checkout') or os.environ.get('AAOCA_PREPROCESS_CHECKOUT', CODE_ROOT))
    root = Path(git(checkout, 'rev-parse', '--show-toplevel').decode().strip())
    if not separate(root, registry.root):
        raise ValueError('更新需要将 --runtime 放在 Git 仓库之外')
    origin = git(root, 'config', '--get', 'remote.origin.url').decode().strip()
    if not mirrors.known_origin(origin):
        raise ValueError('origin 不是已知 AAOCA 项目地址，已保留原配置')
    if git(root, 'symbolic-ref', '--short', 'HEAD').decode().strip() != 'main':
        raise ValueError('请使用 main 分支更新；不会自动切换分支')
    gitdir = Path(git(root, 'rev-parse', '--absolute-git-dir').decode().strip())
    if any((gitdir / name).exists() for name in ('MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply')):
        raise ValueError('仓库有未完成的 Git 操作，请先处理')
    with dispatch_lock(registry.root):
        ensure_current(registry.root)
        registry.ensure_idle()
        if git(root, 'status', '--porcelain').strip():
            raise ValueError('仓库有本地修改或未跟踪文件，已保留；整理后再更新')
        head = git(root, 'rev-parse', 'HEAD').decode().strip()
        baseline = previous.get('commit') or head
        attempt_root = registry.root / 'downloads' / uuid.uuid4().hex
        cache = attempt_root / 'repository.git'
        cache.mkdir(parents=True)
        git(cache, 'init', '--bare')
        report = {'started_at': now(), 'previous': baseline, 'attempts': [], 'state': 'checking'}
        expected = None
        emit('核对官方提交（最多 12 秒），然后按配置顺序下载。')
        try:
            try:
                refs = git(cache, 'ls-remote', mirrors.OFFICIAL, 'refs/heads/main', timeout=12).decode().split()
                if refs and re.fullmatch(r'[0-9a-f]{40}', refs[0]):
                    expected = refs[0]
            except (ValueError, subprocess.TimeoutExpired):
                pass
            report['official_confirmed'] = bool(expected)
            for name in settings['code']:
                url = mirrors.CODE[name]
                emit('正在尝试更新线路：' + name)
                staging = attempt_root / ('staging-' + uuid.uuid4().hex)
                attempt = {'name': name, 'url': url, 'started_at': now()}
                report['attempts'].append(attempt)
                try:
                    # A private promisor remote also routes archive's lazy blob
                    # fetches through this same candidate, never the checkout origin.
                    git(cache, 'config', 'remote.origin.url', url)
                    git(cache, 'config', 'remote.origin.promisor', 'true')
                    git(cache, 'config', 'remote.origin.partialclonefilter', 'blob:none')
                    git(cache, 'fetch', '--no-tags', '--filter=blob:none', 'origin',
                        '+refs/heads/main:refs/heads/candidate', timeout=45)
                    commit = git(cache, 'rev-parse', 'refs/heads/candidate').decode().strip()
                    if expected and commit != expected:
                        raise ValueError('镜像提交与官方当前提交不一致')
                    git(cache, 'merge-base', '--is-ancestor', baseline, commit)
                    if commit != baseline:
                        paths = ['AAOCA/' + name for name in RELEASE_FILES]
                        listing = git(cache, 'ls-tree', '-r', '-z', commit, '--', *paths)
                        blobs = sorted({item.split(b'\t', 1)[0].split()[2].decode()
                                        for item in listing.split(b'\0') if item})
                        # Batch required blobs instead of archive lazily opening
                        # one network connection for every source file.
                        for offset in range(0, len(blobs), 128):
                            git(cache, 'fetch', '--no-tags', 'origin', *blobs[offset:offset + 128], timeout=45)
                        data = git(cache, 'archive', commit, *paths, timeout=45)
                        extract_release(data, staging)
                    attempt.update(state='verified', commit=commit)
                    break
                except (ValueError, OSError, SyntaxError, tarfile.TarError, subprocess.TimeoutExpired) as exc:
                    attempt.update(state='failed', reason=str(exc))
                    emit('本线路未完成，尝试下一条：' + str(exc))
            else:
                raise ValueError('所有配置线路均失败；当前程序保持不变，详见 update-last.json')
            if commit == baseline:
                report['state'] = 'unchanged'
                return {'changed': False, 'message': '已与官方最新版本一致。' if expected else
                        '此线路与当前版本一致，已停止尝试其他镜像；官方最新状态未确认。'}
            if git(root, 'status', '--porcelain').strip() or git(root, 'rev-parse', 'HEAD').decode().strip() != head:
                raise ValueError('下载期间仓库发生变化，保留当前程序')
            registry.ensure_idle()
            backup = registry.root / 'backups' / uuid.uuid4().hex
            backup.mkdir(parents=True)
            for name in RELEASE_FILES:
                source = CODE_ROOT / name
                if source.is_dir():
                    shutil.copytree(source, backup / name, ignore=shutil.ignore_patterns('__pycache__'))
                elif source.is_file():
                    shutil.copy2(source, backup / name)
            history = registry.entries()
            for entry in history:
                if Path(entry.get('code_root', CODE_ROOT)).resolve() == CODE_ROOT:
                    entry.update(code_root=str(backup), python=entry.get('python', sys.executable))
            release = registry.root / 'releases' / commit
            release.parent.mkdir(parents=True, exist_ok=True)
            if release.exists():
                if not same_release(release, staging):
                    raise ValueError('已有发布目录与本次校验内容不同，保留供核查；不会覆盖')
            else:
                staging.rename(release)
            # Saving history first is safe even if switching the release fails:
            # the backup is byte-identical and its interpreter is unchanged.
            write(registry.root / 'history.json', history)
            record = dict(path=str(release), commit=commit, previous=baseline, checkout=str(root),
                          backup=str(backup), official_confirmed=bool(expected))
            write(registry.root / 'release.json', record)
            report.update(state='updated', **record)
            return {'changed': True, 'message': '更新完成，已停止尝试其他镜像。请重新打开面板载入新版；历史任务保留原代码和环境。'
                    + ('' if expected else ' 官方最新状态未确认。')}
        except BaseException as exc:
            report.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            report['finished_at'] = now()
            write(registry.root / 'update-last.json', report)
