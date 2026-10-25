"""Bounded authenticated GitHub download into isolated releases; never update sibling projects."""
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tarfile
import time
from uuid import uuid4
from server_data import digest, read_json, write_json
from task_runtime import CODE, runtime_lock

UPSTREAM = 'https://github.com/tianwuasd/project.git'
SUBDIR = 'vista3d_local_expert_refinement'
ORIGINS = [UPSTREAM, 'git@github.com:tianwuasd/project.git', 'ssh://git@ssh.github.com:443/tianwuasd/project.git',
           'https://gh-proxy.com/https://github.com/tianwuasd/project.git',
           'https://gitclone.com/github.com/tianwuasd/project.git']


def git(root, *args, timeout=20, check=True, binary=False):
    from panel_publication import network
    settings=network.current()
    command = ['git', '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=10']
    command += ['-c','http.followRedirects=false','-c','http.sslVerify=true']
    if settings['proxy']:
        for key in ['http.proxy','http.https://github.com.proxy','http.https://github.com/tianwuasd/project.git.proxy']:
            command += ['-c',key+'='+settings['proxy']]
    command += ['-C', str(root), *args]
    process, cancelled, handlers = None, [False], {}
    def interrupt(*_):
        cancelled[0] = True
        if process is not None:
            raise KeyboardInterrupt
    try:
        for sig in [signal.SIGINT, signal.SIGTERM]:
            handlers[sig] = signal.signal(sig, interrupt)
        env=network.child_env(ssh=settings['git_transport']=='github-ssh')
        env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=env,
                                   start_new_session=os.name == 'posix')
        if cancelled[0]:
            raise KeyboardInterrupt
        out, err = process.communicate(timeout=timeout)
    except BaseException:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None and os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            process.communicate()
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if check and process.returncode:
        raise ValueError('Git操作失败：' + err.decode('utf-8', errors='replace')[-1200:])
    return subprocess.CompletedProcess(command, process.returncode, out if binary else out.decode('utf-8').strip(), err)


def selected_entry(runtime):
    runtime = Path(runtime).resolve()
    pointer = runtime / 'release.json'
    if not pointer.exists():
        return None
    release = read_json(pointer)
    root = Path(release['path']).resolve()
    if not root.is_relative_to(runtime / 'releases'):
        raise ValueError('更新指针不在受管版本目录中')
    for relative, sha in read_json(root / 'release_manifest.json').items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != sha:
            raise ValueError('受管版本文件发生变化，保留原文件，请核查：' + relative)
    entry = root / '05_src/task_panel.py'
    if not entry.is_file():
        raise ValueError('所选版本缺少任务面板入口')
    return entry


def extract_project(data, destination):
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            p = PurePosixPath(member.name)
            prefix = PurePosixPath(SUBDIR)
            if p == prefix or p in prefix.parents:
                continue
            if not p.is_relative_to(prefix) or member.issym() or member.islnk():
                raise ValueError('发布包包含非项目路径或链接')
            relative = p.relative_to(prefix)
            if '..' in relative.parts or relative.is_absolute():
                raise ValueError('发布包路径非法')
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError('发布包包含特殊文件')


def update(runtime):
    from panel_publication import network
    with network.use(network.load(runtime)):
        return _update(runtime)


def _update(runtime):
    from panel_publication import network
    runtime = Path(runtime).resolve()
    with runtime_lock(runtime):
        previous = read_json(runtime / 'release.json') if (runtime / 'release.json').exists() else None
        repository = Path(previous['repository']) if previous else Path(git(CODE, 'rev-parse', '--show-toplevel').stdout)
        origin = git(repository, 'config', '--get', 'remote.origin.url').stdout.rstrip('/')
        if origin.removesuffix('.git') not in {v.rstrip('/').removesuffix('.git') for v in ORIGINS}:
            raise ValueError('origin不属于已核实的私有GitHub仓库；未修改地址')
        if git(repository, 'status', '--porcelain', '--', SUBDIR).stdout:
            raise ValueError('本项目代码存在本地修改，先保留并处理后再更新；未自动stash或覆盖')
        current = previous['commit'] if previous else git(repository, 'rev-parse', 'HEAD').stdout
        selected_route=network.routes(network.current())[0]
        # Do not allow global/local insteadOf rules to redirect credentials.
        resolved=git(repository,'ls-remote','--get-url',selected_route).stdout
        if resolved!=selected_route:
            raise ValueError('Git URL重写改变了官方地址；未发送认证信息，请先核对配置')
        origin=selected_route
        cache = runtime / 'update_cache.git'
        if not cache.exists():
            cache.mkdir()
            git(cache, 'init', '--bare')
        # The original checkout stays unchanged after selecting an isolated release.
        # Validate its selected commit in the managed cache, rebuilding from the
        # authenticated origin if that cache was evicted between updates.
        if git(cache, 'cat-file', '-e', current + '^{commit}', check=False).returncode:
            source = origin if previous else str(repository)
            git(cache, 'fetch', '--no-tags', source, current, timeout=45)
        backup = runtime / 'update_backups' / (str(time.time_ns()))
        backup.mkdir(parents=True)
        write_json(backup / 'previous.json', dict(pointer=previous, commit=current, repository=str(repository)))
        (backup / 'code.tar').write_bytes(git(cache, 'archive', current, SUBDIR, binary=True).stdout)
        attempts, known, chosen = [], None, None
        try:
            known = git(repository, 'ls-remote', origin, 'refs/heads/main', timeout=12).stdout.split()[0]
        except (ValueError, subprocess.SubprocessError, IndexError):
            print('原站查询未完成，将尝试认证获取 main 并核对快进关系。', flush=True)
        for url in [origin]:
            print('检查下载线路：' + url, flush=True)
            try:
                git(cache, 'fetch', '--no-tags', url, 'refs/heads/main', timeout=45)
                commit = git(cache, 'rev-parse', 'FETCH_HEAD').stdout
                if known and commit != known:
                    raise ValueError('线路提交与原站已知版本不符')
                if git(cache, 'merge-base', '--is-ancestor', current, commit, check=False).returncode:
                    raise ValueError('候选历史不能从当前版本快进')
                chosen = commit
                attempts.append(dict(url=url, status='accepted', commit=commit))
                break
            except (ValueError, subprocess.SubprocessError) as error:
                attempts.append(dict(url=url, status='failed', error=str(error)))
        write_json(backup / 'attempts.json', dict(attempts=attempts, official_head=known))
        if chosen is None:
            raise ValueError('所有更新线路失败，保留当前版本；日志：' + str(backup))
        if chosen == current:
            print('所选线路与当前版本相同。' + ('已核对原站。' if known else '尚未确认原站最新状态。'))
            return
        target = runtime / 'releases' / chosen
        temporary = runtime / 'releases' / ('.pending_'+uuid4().hex)
        temporary.mkdir(parents=True)
        extract_project(git(cache, 'archive', chosen, SUBDIR, binary=True).stdout, temporary)
        for required in ['05_src/task_panel.py','05_src/task_worker.py','bootstrap_server.sh']:
            if not (temporary / required).is_file():
                raise ValueError('候选版本尚未包含兼容的新面板：' + required)
        files = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()}
        write_json(temporary / 'release_manifest.json', files)
        if target.exists():
            if read_json(target / 'release_manifest.json') != files or any(digest(target / p) != sha for p,sha in files.items()):
                raise ValueError('同名受管版本内容不同，停止覆盖')
            if temporary.resolve().parent != (runtime / 'releases').resolve() or not temporary.name.startswith('.pending_'):
                raise ValueError('临时发布目录边界不符，拒绝清理')
            shutil.rmtree(temporary)
        else:
            temporary.rename(target)
        write_json(runtime / 'release.json', dict(path=str(target), commit=chosen, repository=str(repository), backup=str(backup)))
        selected_entry(runtime)
        print(f'已选择独立版本 {chosen[:12]}；旧任务仍用原快照。备份：{backup}')
