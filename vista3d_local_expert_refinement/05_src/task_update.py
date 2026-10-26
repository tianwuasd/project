"""Bounded authenticated GitHub download into isolated releases; never update sibling projects."""
import io
import os
import re
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
FETCH_TIMEOUT_SECONDS = 900
QUERY_TIMEOUT_SECONDS = 30
ARCHIVE_TIMEOUT_SECONDS = 180
PROGRESS_POLL_SECONDS = 1
PROGRESS_HEARTBEAT_SECONDS = 5
STALL_TIMEOUT_SECONDS = 120
RUNTIME_PATHS = ('05_src', '06_configs', '04_data/manifests/README.md',
                 'bootstrap_server.sh', 'start_server.sh')


def progress_token(data):
    """Ignore stale transfer rates and repeated counters when detecting stalls."""
    lines=data.decode('utf-8',errors='replace').replace('\r','\n').splitlines()
    for line in reversed(lines):
        match=re.search(r'(?:Enumerating|Counting|Compressing|Receiving|Resolving|Unpacking) (?:objects|deltas):\s*\d[^|]*',line)
        if match:return match.group().strip()
    return ''


def communicate_progress(process, command, timeout, log_path):
    """Repeated communicate calls drain both pipes and retain cumulative output."""
    started=time.monotonic();last_print=last_progress=started
    token='';out_size=0
    offset=0;out=err=b''
    log_path=Path(log_path)
    print(f'代码下载开始，最多等待{timeout:g}秒；Ctrl+C取消并保留当前版本。日志：{log_path}',flush=True)
    with os.fdopen(os.open(log_path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600),'ab') as stream:
        def record(data):
            nonlocal offset
            stream.write(data[offset:]);stream.flush();offset=len(data)
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=timeout:
                raise subprocess.TimeoutExpired(command,timeout,output=out,stderr=err)
            if time.monotonic()-last_progress>=STALL_TIMEOUT_SECONDS:
                raise ValueError(f'下载停滞：{STALL_TIMEOUT_SECONDS:g}秒未见Git计数或字节推进；保留当前版本与缓存；日志：{log_path}')
            try:
                out,err=process.communicate(timeout=min(PROGRESS_POLL_SECONDS,timeout-elapsed))
                record(err)
                status='完成' if process.returncode==0 else '进程退出'
                print(f'代码下载{status}；耗时{time.monotonic()-started:.1f}秒。',flush=True)
                return out,err
            except subprocess.TimeoutExpired as pending:
                out,err=pending.output or b'',pending.stderr or b''
                record(err)
                now=time.monotonic()
                latest=progress_token(err)
                if latest!=token or len(out)>out_size:
                    token=latest;out_size=len(out);last_progress=now
                    if latest:print('Git进度：'+latest[-240:],flush=True)
                    last_print=now
                elif now-last_print>=PROGRESS_HEARTBEAT_SECONDS:
                    print(f'等待Git推进；本次已用{now-started:.0f}秒 / {timeout:g}秒上限；连续{now-last_progress:.0f}秒无新进度；上次计数：{token[-200:] or "尚无"}',flush=True)
                    last_print=now


def git(root, *args, timeout=20, check=True, binary=False, progress_log=None, input_bytes=None):
    from panel_publication import network
    settings=network.current()
    command = ['git', '-c', 'http.lowSpeedLimit=1', '-c', 'http.lowSpeedTime=10']
    command += ['-c','http.followRedirects=false','-c','http.sslVerify=true']
    if args and args[0]=='archive':
        command += ['-c','core.attributesfile='+os.devnull,'-c','core.autocrlf=false']
    if settings['proxy']:
        for key in ['http.proxy','http.https://github.com.proxy','http.https://github.com/tianwuasd/project.git.proxy']:
            command += ['-c',key+'='+settings['proxy']]
    command += ['-C', str(root), *args]
    process, cancelled, handlers = None, [False], {}
    log_start=Path(progress_log).stat().st_size if progress_log and Path(progress_log).exists() else 0
    def interrupt(*_):
        cancelled[0] = True
        if process is not None:
            raise KeyboardInterrupt
    try:
        for sig in [signal.SIGINT, signal.SIGTERM]:
            handlers[sig] = signal.signal(sig, interrupt)
        env={key:value for key,value in network.child_env(ssh=settings['git_transport']=='github-ssh').items()
             if not key.startswith('GIT_') or key in {'GIT_SSH_COMMAND','GIT_SSH_VARIANT'}}
        env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
        if args and args[0] in {'archive','cat-file','ls-tree','merge-base'}:
            env.update(GIT_ALLOW_PROTOCOL='',GIT_NO_LAZY_FETCH='1',GIT_ATTR_NOSYSTEM='1')
        process = subprocess.Popen(command, stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=env,
                                   start_new_session=os.name == 'posix')
        if cancelled[0]:
            raise KeyboardInterrupt
        if progress_log is None:
            out, err = process.communicate(input=input_bytes,timeout=timeout)
        else:
            out, err = communicate_progress(process,command,timeout,progress_log)
    except BaseException as error:
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
            out,err=process.communicate()
            if progress_log is not None and Path(progress_log).exists():
                # communicate returns cumulative bytes; append only the tail
                # received during cancellation/timeout cleanup.
                offset=max(0,Path(progress_log).stat().st_size-log_start)
                with Path(progress_log).open('ab') as stream:stream.write(err[offset:])
            if isinstance(error,subprocess.TimeoutExpired):
                error.output,error.stderr=out,err
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


def backup_loaded_code(destination):
    """Back up loaded runtime files locally; never fetch history for a backup."""
    started=time.monotonic()
    with tarfile.open(destination,'w') as archive:
        for relative in RUNTIME_PATHS:
            source=CODE/relative
            paths=sorted(source.rglob('*')) if source.is_dir() else [source]
            for path in paths:
                if time.monotonic()-started>ARCHIVE_TIMEOUT_SECONDS:
                    raise subprocess.TimeoutExpired(['local','backup'],ARCHIVE_TIMEOUT_SECONDS)
                if '__pycache__' in path.parts or path.suffix=='.pyc' or not path.exists():continue
                if path.is_symlink():raise ValueError('运行代码备份包含链接，停止更新')
                if path.is_file():archive.add(path,arcname=SUBDIR+'/'+path.relative_to(CODE).as_posix(),recursive=False)


def download_runtime(cache, url, current, known, log):
    """Fetch commit/tree metadata, then only missing blobs for runtime paths."""
    git(cache,'config','remote.origin.url',url)
    git(cache,'config','remote.origin.promisor','true')
    git(cache,'config','remote.origin.partialclonefilter','blob:none')
    if git(cache,'ls-remote','--get-url',url).stdout!=url:
        raise ValueError('缓存Git URL重写改变了官方地址，停止更新')
    deadline=time.monotonic()+FETCH_TIMEOUT_SECONDS
    first=[True]
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise subprocess.TimeoutExpired(['git','runtime-download'],FETCH_TIMEOUT_SECONDS)
        return value
    def fetch(*args):
        limit=FETCH_TIMEOUT_SECONDS if first[0] else remaining()
        first[0]=False
        result=git(cache,'fetch','--no-tags','--progress',*args,timeout=limit,progress_log=log)
        remaining()
        return result
    def inspect(*args,**kwargs):
        result=git(cache,*args,timeout=min(QUERY_TIMEOUT_SECONDS,remaining()),**kwargs)
        remaining()
        return result
    print('仅获取版本/目录信息，随后下载VISTA运行代码；不下载其它项目、文献或历史文件内容。',flush=True)
    fetch('--filter=blob:none','origin','+refs/heads/main:refs/heads/vista-candidate')
    commit=inspect('rev-parse','refs/heads/vista-candidate').stdout
    if known and commit!=known:raise ValueError('下载期间官方main变化，请重新检查版本')
    if inspect('merge-base','--is-ancestor',current,commit,check=False).returncode:
        raise ValueError('候选历史不能从当前版本快进')
    roots=[SUBDIR+'/'+name for name in RUNTIME_PATHS]
    listing=inspect('ls-tree','-r','-z',commit,'--',*roots,binary=True).stdout
    paths=[];blobs=set()
    for row in listing.split(b'\0'):
        if not row:continue
        metadata,name=row.split(b'\t',1)
        mode,kind,oid=metadata.decode('ascii').split()
        if kind!='blob' or mode not in {'100644','100755'}:raise ValueError('运行代码清单包含非普通文件')
        paths.append(name.decode('utf-8'));blobs.add(oid)
    if not paths:raise ValueError('目标提交没有VISTA运行文件')
    data=('\n'.join(sorted(blobs))+'\n').encode('ascii')
    available=inspect('cat-file','--batch-check=%(objectname) %(objecttype)',input_bytes=data).stdout
    present={row.split()[0] for row in available.splitlines() if row.endswith(' blob')}
    missing=sorted(blobs-present)
    print(f'运行代码：{len(paths)}个文件；需下载{len(missing)}个文件对象，复用{len(blobs)-len(missing)}个缓存对象。',flush=True)
    for start in range(0,len(missing),128):
        fetch('--no-write-fetch-head','--filter=blob:none','origin',*missing[start:start+128])
    # Detect missing objects here, with implicit networking disabled.
    verified=inspect('cat-file','--batch-check=%(objectname) %(objecttype)',input_bytes=data).stdout
    if any(not row.endswith(' blob') for row in verified.splitlines()) or len(verified.splitlines())!=len(blobs):
        raise ValueError('运行代码对象不完整，未切换版本')
    remaining()
    return commit,paths


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
        backup = runtime / 'update_backups' / (str(time.time_ns()))
        backup.mkdir(parents=True)
        write_json(backup / 'previous.json', dict(pointer=previous, commit=current, repository=str(repository)))
        cache = runtime / 'update_cache.git'
        if not cache.exists():
            cache.mkdir()
            git(cache, 'init', '--bare')
        print('保存当前版本备份。',flush=True)
        backup_loaded_code(backup/'code.tar')
        attempts, known, chosen = [], None, None
        try:
            print('读取GitHub main版本。',flush=True)
            known = git(repository, 'ls-remote', origin, 'refs/heads/main', timeout=QUERY_TIMEOUT_SECONDS).stdout.split()[0]
            print('已读取原站版本：'+known[:12]+'；即将下载并核对代码。',flush=True)
        except (ValueError, subprocess.SubprocessError, IndexError):
            print('原站查询未完成，将尝试认证获取 main 并核对快进关系。', flush=True)
        if known==current:
            print('已与官方main版本一致，无需下载对象。',flush=True)
            return
        for url in [origin]:
            print('检查下载线路：' + url, flush=True)
            try:
                commit,paths=download_runtime(cache,url,current,known,backup/'fetch.log')
                chosen = commit
                attempts.append(dict(url=url, status='accepted', commit=commit))
                break
            except (ValueError, subprocess.SubprocessError) as error:
                reason=(f'下载超时（{error.timeout:g}秒上限）；已经读取到原站版本不代表代码已下载完成'
                        if isinstance(error,subprocess.TimeoutExpired) else str(error))
                print('此线路未完成：'+reason,flush=True)
                attempts.append(dict(url=url, status='failed', error=reason,log=str(backup/'fetch.log')))
        write_json(backup / 'attempts.json', dict(attempts=attempts, official_head=known))
        if chosen is None:
            raise ValueError('更新未完成：'+attempts[-1]['error']+'；保留当前版本；日志：'+str(backup))
        if chosen == current:
            print('所选线路与当前版本相同。' + ('已核对原站。' if known else '尚未确认原站最新状态。'))
            return
        target = runtime / 'releases' / chosen
        temporary = runtime / 'releases' / ('.pending_'+uuid4().hex)
        temporary.mkdir(parents=True)
        print('下载通过，正在校验并生成本项目独立版本。',flush=True)
        extract_project(git(cache, 'archive', '--worktree-attributes', chosen, *paths,
                            binary=True,timeout=ARCHIVE_TIMEOUT_SECONDS).stdout, temporary)
        for required in ['05_src/task_panel.py','05_src/task_worker.py','05_src/task_runtime.py',
                         '06_configs/baseline.json','04_data/manifests/README.md','bootstrap_server.sh','start_server.sh']:
            if not (temporary / required).is_file():
                raise ValueError('候选版本尚未包含兼容的新面板：' + required)
        for source in temporary.rglob('*.py'):
            try:compile(source.read_bytes(),str(source),'exec')
            except SyntaxError as error:
                raise ValueError(f'候选代码语法检查失败：{source.relative_to(temporary)}:{error.lineno}；保留当前版本') from error
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
