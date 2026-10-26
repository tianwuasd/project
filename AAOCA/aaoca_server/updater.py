"""Authenticated official updates and isolated releases for the private repository."""
import io
from contextlib import contextmanager
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid

from . import download, mirrors, network
from .state import dispatch_lock, now, read, separate, write

CODE_ROOT = Path(__file__).resolve().parent.parent
QUERY_TIMEOUT = 30
FETCH_TIMEOUT = 900
ARCHIVE_TIMEOUT = 180
RELEASE_FILES = ('aaoca_preprocessing', 'aaoca_server', 'start_aaoca_server.py',
                 'start_aaoca_server.sh', 'preprocess_aaoca.py', 'setup_aaoca_environment.py',
                 'upgrade_aaoca_server.py',
                 'pyproject.toml', 'requirements-preprocess.txt', 'PREPROCESSING.md',
                 'PREPROCESSING_WORKFLOW.md', 'SERVER_WORKBENCH.md', 'PRIVATE_UPGRADE.md')


def command(args, *, cwd=None, env=None, timeout=QUERY_TIMEOUT):
    """Reap only this command's process group before releasing caller locks."""
    child = None
    handlers = {}
    interrupted = []
    def interrupt(sig, frame):
        interrupted.append(True)
        if child is not None:
            raise KeyboardInterrupt
    try:
        can_defer = threading.current_thread() is threading.main_thread()
        if can_defer:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupt)
        child = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=os.name != 'nt')
        if interrupted:
            raise KeyboardInterrupt
        stdout, stderr = child.communicate(timeout=timeout)
    except BaseException:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
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
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if child.returncode:
        raise ValueError(stderr.decode('utf-8', 'replace')[-1000:] or f'command failed ({child.returncode})')
    return stdout


def git(root, *args, timeout=30, emit=None, log=None):
    started = time.monotonic()
    settings = network.current()
    # Keep the user's normal credential helpers, but do not inherit repository
    # redirection/config injection from another Git process.
    env = {k: v for k, v in network.child_env(ssh=settings['git_transport'] == 'github-ssh').items()
           if not k.startswith('GIT_') or k in {'GIT_SSH_COMMAND', 'GIT_SSH_VARIANT'}}
    env.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    if emit is not None:
        env['LC_ALL'] = 'C'
    options = ['git', '-c', 'core.askPass=', '-c', 'http.followRedirects=false',
               '-c', 'http.' + network.HTTPS_UPSTREAM + '.followRedirects=false',
               '-c', 'http.version=HTTP/1.1', '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=15']
    if settings['proxy']:
        options += ['-c', 'http.proxy=' + settings['proxy'],
                    '-c', 'http.' + network.HTTPS_UPSTREAM + '.proxy=' + settings['proxy']]
    options += ['-C', str(root)]
    # --get-url only expands Git config; it does not contact the destination.
    # Guard lazy promisor fetching (archive) as well as explicit network calls.
    if args and args[0] in {'fetch', 'ls-remote', 'archive'} and '--get-url' not in args:
        official = network.routes(settings)[0]
        try:
            resolved = command([*options, 'ls-remote', '--get-url', official], env=env,
                               timeout=min(timeout, QUERY_TIMEOUT)).decode().strip()
        except ValueError:
            raise ValueError('无法核验官方线路配置；请检查 Git 配置') from None
        if resolved != official:
            raise ValueError('Git URL 重写改变了官方地址；请检查 insteadOf，未发送认证或访问镜像')
    try:
        if emit is not None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(['git', args[0]], timeout)
            return download.run([*options, *args], env=env, timeout=remaining, emit=emit, log=log)
        return command([*options, *args], env=env, timeout=timeout)
    except download.GitFailure:
        raise
    except ValueError:
        raise ValueError('Git 操作失败；请检查官方线路、代理、主机指纹和仓库认证。不会回退公共镜像。') from None


@contextmanager
def update_stage(registry, report, label, emit):
    started = time.monotonic()
    stage = dict(name=label, started_at=now(), state='running')
    report['stages'].append(stage)
    report['phase'] = label
    write(registry.root / 'update-last.json', report)
    with Path(report['log']).open('a', encoding='utf-8') as log:
        log.write(now() + ' 开始 ' + label + '\n')
    emit(label + '；日志：' + report['log'])
    try:
        yield
        stage['state'] = 'completed'
    except subprocess.TimeoutExpired as exc:
        stage.update(state='failed', error_kind='timeout')
        message = f'{label}超时（上限 {exc.timeout:.1f} 秒）；旧版和下载缓存保留。日志：{report["log"]}'
        stage['error'] = message
        raise ValueError(message) from None
    except KeyboardInterrupt:
        stage.update(state='cancelled', error_kind='cancelled')
        emit(label + '已取消，Git 子进程已回收，旧版及缓存保留。日志：' + report['log'])
        raise
    except BaseException as exc:
        stage.update(state='failed', error_kind='validation' if not isinstance(exc, download.GitFailure) else 'git',
                     error=str(exc))
        raise ValueError(f'{label}失败：{exc}；旧版和缓存保留。日志：{report["log"]}') from None
    finally:
        stage.update(finished_at=now(), seconds=time.monotonic() - started)
        with Path(report['log']).open('a', encoding='utf-8') as log:
            log.write(f'{now()} {label} {stage["state"]}，用时 {stage["seconds"]:.3f} 秒\n')
            if stage.get('error'):
                log.write(stage['error'] + '\n')
        write(registry.root / 'update-last.json', report)


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
    settings = network.load(registry.root)
    with network.use(settings):
        return _update(registry, settings, emit)


def _update(registry, settings, emit):
    previous = read(registry.root / 'release.json', {})
    checkout = Path(previous.get('checkout') or os.environ.get('AAOCA_PREPROCESS_CHECKOUT', CODE_ROOT))
    root = Path(git(checkout, 'rev-parse', '--show-toplevel').decode().strip())
    if not separate(root, registry.root):
        raise ValueError('更新需要将 --runtime 放在 Git 仓库之外')
    origin = git(root, 'config', '--get', 'remote.origin.url').decode().strip()
    if not mirrors.known_origin(origin):
        raise ValueError('origin 不是已知 AAOCA 项目地址，已保留原配置')
    official = network.routes(settings)[0]
    resolved = git(root, 'ls-remote', '--get-url', official).decode().strip()
    if resolved != official:
        raise ValueError('原仓库 Git URL 重写改变了官方地址；请检查 insteadOf，未访问网络')
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
        attempt_root.mkdir(parents=True)
        cache = registry.root / 'download-cache' / 'repository.git'
        if not cache.resolve().is_relative_to(registry.root.resolve()):
            raise ValueError('下载缓存路径超出原 runtime；请保留并检查缓存目录')
        cache.mkdir(parents=True, exist_ok=True)
        git(cache, 'init', '--bare')
        log = attempt_root / 'git-download.log'
        log.touch()
        if os.name == 'posix':
            log.chmod(0o600)
        report = {'started_at': now(), 'previous': baseline, 'attempts': [], 'state': 'checking',
                  'cache': str(cache), 'log': str(log), 'stages': [],
                  'limits': dict(query_seconds=QUERY_TIMEOUT, fetch_seconds=FETCH_TIMEOUT,
                                 archive_seconds=ARCHIVE_TIMEOUT)}
        expected = None
        emit('通过原 runtime 的官方认证线路更新；版本查询最多30秒，对象下载总上限900秒，归档最多180秒。')
        try:
            url = network.routes(settings)[0]
            with update_stage(registry, report, '版本查询', emit):
                refs = git(cache, 'ls-remote', url, 'refs/heads/main', timeout=QUERY_TIMEOUT,
                           emit=emit, log=log).decode().split()
                if not refs or not re.fullmatch(r'[0-9a-f]{40}', refs[0]):
                    raise ValueError('官方 main 提交未确认；请检查仓库访问权限')
                expected = refs[0]
                report['official_confirmed'] = True
                report['remote_commit'] = expected
                emit('已确认官方 main：' + expected)
            for name in [settings['git_transport']]:
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
                    if expected == baseline:
                        report['state'] = 'unchanged'
                        return {'changed': False, 'message': '已与官方最新版本一致，无需下载对象。'}
                    paths = ['AAOCA/' + item for item in RELEASE_FILES]
                    with update_stage(registry, report, '对象下载', emit):
                        deadline = time.monotonic() + FETCH_TIMEOUT

                        def remaining_budget():
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(['git', 'download-stage'], FETCH_TIMEOUT)
                            return remaining

                        def fetch(*arguments):
                            result = git(cache, 'fetch', '--progress', *arguments,
                                         timeout=remaining_budget(), emit=emit, log=log)
                            remaining_budget()
                            return result

                        def inspect(*arguments):
                            result = git(cache, *arguments, timeout=min(QUERY_TIMEOUT, remaining_budget()))
                            remaining_budget()
                            return result

                        fetch('--no-tags', '--filter=blob:none', 'origin',
                              '+refs/heads/main:refs/heads/candidate')
                        commit = inspect('rev-parse', 'refs/heads/candidate').decode().strip()
                        if commit != expected:
                            raise ValueError('版本校验：核对后官方提交已变化，请稍后重试')
                        inspect('merge-base', '--is-ancestor', baseline, commit)
                        listing = inspect('ls-tree', '-r', '-z', commit, '--', *paths)
                        blobs = sorted({item.split(b'\t', 1)[0].split()[2].decode()
                                        for item in listing.split(b'\0') if item})
                        if blobs:
                            existing = inspect('cat-file', '--batch-all-objects',
                                               '--batch-check=%(objectname) %(objecttype)').decode().splitlines()
                            available = {row.split()[0] for row in existing}
                            blobs = [blob for blob in blobs if blob not in available]
                        for offset in range(0, len(blobs), 128):
                            fetch('--no-tags', 'origin', *blobs[offset:offset + 128])
                        remaining_budget()
                    with update_stage(registry, report, '本地归档与校验', emit):
                        data = git(cache, 'archive', commit, *paths, timeout=ARCHIVE_TIMEOUT,
                                   emit=emit, log=log)
                        extract_release(data, staging)
                    attempt.update(state='verified', commit=commit)
                    break
                except (ValueError, OSError, SyntaxError, tarfile.TarError, subprocess.TimeoutExpired) as exc:
                    attempt.update(state='failed', reason=str(exc))
                    emit('官方更新未完成；保留当前版本：' + str(exc))
                    raise
            else:
                raise ValueError('官方线路更新失败；当前程序保持不变，详见 update-last.json；不会回退公共镜像')
            if commit == baseline:
                report['state'] = 'unchanged'
                return {'changed': False, 'message': '已与官方最新版本一致。'}
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
            return {'changed': True, 'message': '官方更新完成。请重新打开面板载入新版；历史任务保留原代码和环境。'}
        except BaseException as exc:
            report.update(state='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            report['finished_at'] = now()
            write(registry.root / 'update-last.json', report)
