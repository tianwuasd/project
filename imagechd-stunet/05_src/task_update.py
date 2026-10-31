# CT 0.22.1 / 88bc9280293e99e7aefea6528f0ddd34ff4f67d1.
# This block is embedded unchanged so old single-file bootstraps remain compatible.
# BEGIN CT UPDATE PROGRESS
"""Drain Git output concurrently, show real progress, and retain bounded safe logs."""

import codecs
import os
import queue
import re
import signal
import subprocess
import threading
import time

HEARTBEAT_SECONDS = 5
TAIL_CHARS = 65536
MAX_LOG_BYTES = 1024 * 1024
PROGRESS = re.compile(
    r"^(?:remote: )?(?:Enumerating|Counting|Compressing|Receiving|Resolving|Unpacking|Updating|Checking out) (?:objects|deltas|files):\s*\d"
)


def safe_text(value):
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    value = "".join(c for c in value if c.isprintable() or c in "\n\t")
    value = re.sub(r"(?i)(authorization|proxy-authorization)\s*[:=].*", r"\1: [redacted]", value)
    value = re.sub(
        r"(?i)\b(password|token|secret|access_token|api[_-]?key)\s*[=:]\s*\S+", r"\1=[redacted]", value
    )
    value = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[redacted]@", value)
    value = re.sub(r"(?i)(https?://[^\s?#]+)[?#][^\s]*", r"\1?[redacted]", value)
    value = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)\b", "[redacted]", value)
    return value


def failure_reason(stderr):
    text = stderr.lower()
    if "would be overwritten" in text:
        return "本地文件将被更新覆盖，已拒绝应用；请先保留并处理这些文件"
    if "no space left" in text:
        return "磁盘空间不足，请先检查代码和 runtime 所在磁盘"
    if any(
        word in text for word in ("rpc failed", "early eof", "index-pack failed", "unexpected disconnect")
    ):
        return "对象传输或解包中断；先检查当前下载线路和磁盘，不据此重建密钥"
    if "permission denied" in text or "authentication failed" in text:
        return "仓库认证或访问权限失败，请检查当前配置的密钥/权限"
    if "host key verification failed" in text:
        return "SSH 主机指纹校验失败，请核对已知主机记录"
    if "could not resolve" in text or "name or service not known" in text:
        return "目标域名解析失败"
    if any(
        word in text
        for word in (
            "ssl",
            "tls",
            "banner exchange",
            "connection reset",
            "connection refused",
            "connection timed out",
            "connect to host",
        )
    ):
        return "代理或远端连接/握手失败，请查看本次目标的代理日志"
    if "not found" in text or "does not appear to be a git repository" in text:
        return "仓库不可访问、地址不存在或权限不足"
    return "Git 进程异常退出；具体输出见日志"


def terminate(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()


class StageTimeout(subprocess.TimeoutExpired):
    def __init__(self, stage, timeout, log):
        super().__init__(stage, timeout)
        self.log = log

    def __str__(self):
        return f"{self.cmd}总超时（上限 {self.timeout:g} 秒）；日志：{self.log}。请查更新记录确认实际状态。"


class StageFailure(ValueError):
    def __init__(self, stage, stderr, log):
        super().__init__(f"{stage}失败：{failure_reason(stderr)}；日志：{log}")
        self.output_tail = stderr[-4000:]


class Journal:
    def __init__(self, path):
        self.stream = path.open("wb")
        self.size = 0
        self.full = False

    def write(self, message):
        if self.full:
            return
        data = (safe_text(message) + "\n").encode("utf-8")
        if self.size + len(data) > MAX_LOG_BYTES - 160:
            data = "日志达到大小上限；仍继续下载并在控制台显示进度。\n".encode("utf-8")
            self.full = True
        self.stream.write(data)
        self.stream.flush()
        self.size += len(data)


def communicate(process, timeout, log, stage):
    """Keep a fixed deadline even when Git keeps writing carriage-return updates."""
    started = time.monotonic()
    deadline, heartbeat = started + timeout, started + HEARTBEAT_SECONDS
    events = queue.Queue(maxsize=128)
    stopped = threading.Event()
    journal = Journal(log)
    tails = {"stdout": "", "stderr": ""}
    handlers = {}

    def emit(item):
        while not stopped.is_set():
            try:
                events.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def reader(name, stream):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""

        def consume(text):
            nonlocal pending
            for char in text:
                if char in "\r\n":
                    if pending:
                        emit((name, pending))
                        pending = ""
                elif len(pending) < 16384:
                    pending += char

        try:
            while not stopped.is_set():
                data = os.read(stream.fileno(), 4096)
                if not data:
                    break
                consume(decoder.decode(data))
            consume(decoder.decode(b"", final=True))
            if pending:
                emit((name, pending))
        except OSError:
            if not stopped.is_set():
                emit(("error", "无法读取 Git 子进程输出"))
        finally:
            emit(("done", name))

    threads = [
        threading.Thread(
            target=reader, args=(name, getattr(process, name)), name="chd-git-" + name, daemon=True
        )
        for name in tails
    ]
    try:
        journal.write(f"阶段：{stage}；总上限 {timeout:g} 秒")
        for thread in threads:
            thread.start()
        done = 0
        while done < 2:
            now = time.monotonic()
            if now >= deadline:
                raise StageTimeout(stage, timeout, log)
            if now >= heartbeat:
                print(
                    f"{stage}仍在进行/等待输出，已用 {now - started:.0f} 秒（上限 {timeout:g} 秒）",
                    flush=True,
                )
                heartbeat = now + HEARTBEAT_SECONDS
            try:
                name, text = events.get(timeout=min(0.1, deadline - now))
            except queue.Empty:
                continue
            if name == "done":
                done += 1
            elif name == "error":
                raise OSError(text)
            else:
                clean = safe_text(text)
                tails[name] = (tails[name] + clean + "\n")[-TAIL_CHARS:]
                journal.write(f"[{name}] {clean}")
                if name == "stderr" and PROGRESS.match(clean):
                    print(clean, flush=True)
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise StageTimeout(stage, timeout, log) from None
        journal.write(f"阶段结束：exit={process.returncode}；耗时 {time.monotonic() - started:.2f} 秒")
        return tails["stdout"], tails["stderr"]
    except BaseException as error:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, signal.SIG_IGN)
        error.output_tail = tails["stderr"][-4000:]
        stopped.set()
        terminate(process)
        process.wait()
        journal.write("阶段停止：" + (str(error) or type(error).__name__))
        raise
    finally:
        stopped.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=2)
        journal.stream.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)

# END CT UPDATE PROGRESS

import io
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from uuid import uuid4
from server_data import digest, read_json, write_json
from task_runtime import CODE, runtime_lock

UPSTREAM = 'https://github.com/tianwuasd/project.git'
SUBDIR = 'imagechd-stunet'
ORIGINS = [UPSTREAM, 'git@github.com:tianwuasd/project.git', 'ssh://git@ssh.github.com:443/tianwuasd/project.git',
           'https://gh-proxy.com/https://github.com/tianwuasd/project.git',
           'https://gitclone.com/github.com/tianwuasd/project.git']
FETCH_TIMEOUT_SECONDS = 900
QUERY_TIMEOUT_SECONDS = 30
ARCHIVE_TIMEOUT_SECONDS = 180
RUNTIME_PATHS = ('05_src', '06_configs', '04_data/manifests/README.md', 'bootstrap_server.sh', 'start_server.sh')


def git(root, *args, timeout=20, check=True, binary=False, progress_log=None, input_bytes=None, stage='Git 操作'):
    """CT authenticated transport/stream handling, with Vista binary/local operations."""
    from panel_publication import network
    settings=network.current()
    command=['git','-c','core.askPass=','-c','http.followRedirects=false',
             '-c','http.'+UPSTREAM+'.followRedirects=false','-c','http.sslVerify=true',
             '-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=15']
    if args and args[0]=='archive':
        command+=['-c','core.attributesfile='+os.devnull,'-c','core.autocrlf=false']
    if settings['proxy']:
        for key in ['http.proxy','http.https://github.com.proxy','http.'+UPSTREAM+'.proxy']:
            command+=['-c',key+'='+settings['proxy']]
    command+=['-C',str(root),*args]
    env={k:v for k,v in network.child_env(ssh=settings['git_transport']=='github-ssh').items()
         if not k.startswith('GIT_') or k in {'GIT_SSH_COMMAND','GIT_SSH_VARIANT'}}
    env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    if args and args[0] in {'archive','cat-file','ls-tree','merge-base'}:
        env.update(GIT_ALLOW_PROTOCOL='',GIT_NO_LAZY_FETCH='1',GIT_ATTR_NOSYSTEM='1')
    process=None;cancelled=False;handlers={}
    def interrupt(*_):
        nonlocal cancelled
        cancelled=True
        if process is not None:raise KeyboardInterrupt
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        process=subprocess.Popen(command,stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=os.name=='posix')
        if cancelled:raise KeyboardInterrupt
        if progress_log is None:
            out,err=process.communicate(input=input_bytes,timeout=timeout)
            if not binary:out=out.decode('utf-8',errors='replace').strip()
        else:
            if input_bytes is not None:raise ValueError('流式Git阶段不接受stdin数据')
            print(f'{stage}：最多 {timeout:g} 秒；日志：{progress_log}',flush=True)
            out,text=communicate(process,timeout,Path(progress_log),stage)
            err=text.encode('utf-8')
            out=out.encode('utf-8') if binary else out.strip()
        result=subprocess.CompletedProcess(command,process.returncode,out,err)
        if check and result.returncode:
            if progress_log is not None:raise StageFailure(stage,err.decode('utf-8',errors='replace'),progress_log)
            raise ValueError('Git操作失败：'+safe_text(err.decode('utf-8',errors='replace'))[-1800:])
        return result
    except BaseException as error:
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        if process is not None:
            terminate(process)
            if progress_log is None:
                out,err=process.communicate()
                if isinstance(error,subprocess.TimeoutExpired):error.output,error.stderr=out,err
            else:process.wait()
        raise
    finally:
        # Stream readers have joined (or communicate has drained them). Explicitly
        # release handles even on streaming success and interrupted downloads.
        if process is not None:
            for name in ['stdin','stdout','stderr']:
                stream=getattr(process,name,None)
                if stream is not None and not stream.closed:
                    try:stream.close()
                    except OSError:pass
        for sig,handler in handlers.items():signal.signal(sig,handler)


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
    first=[True];batch=[0]
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise subprocess.TimeoutExpired(['git','runtime-download'],FETCH_TIMEOUT_SECONDS)
        return value
    def fetch(*args):
        limit=FETCH_TIMEOUT_SECONDS if first[0] else remaining()
        label='提交/目录下载' if first[0] else '运行文件下载'
        first[0]=False;batch[0]+=1
        batch_log=Path(log) if batch[0]==1 else Path(log).with_name(Path(log).stem+'-'+str(batch[0])+Path(log).suffix)
        result=git(cache,'fetch','--no-tags','--progress',*args,timeout=limit,progress_log=batch_log,stage=label)
        remaining()
        return result
    def inspect(*args,**kwargs):
        result=git(cache,*args,timeout=min(QUERY_TIMEOUT_SECONDS,remaining()),**kwargs)
        remaining()
        return result
    print('仅获取版本/目录信息，随后下载VISTA运行代码；不下载其它项目、文献或历史文件内容。',flush=True)
    fetch('--filter=blob:none',url,'+refs/heads/main:refs/heads/vista-candidate')
    commit=inspect('rev-parse','refs/heads/vista-candidate').stdout
    if known and commit!=known:
        print('下载期间main发生变化；重新查询官方引用并核对前后关系。',flush=True)
        verified=git(cache,'ls-remote',url,'refs/heads/main',timeout=min(QUERY_TIMEOUT_SECONDS,remaining()),
                     progress_log=Path(log).with_name('verify-main.log'),stage='官方版本复核').stdout.split()
        remaining()
        if not verified or verified[0]!=commit:
            raise ValueError('官方main仍在变化或与下载版本不符；本次不切换，请稍后重试')
        if inspect('merge-base','--is-ancestor',known,commit,check=False).returncode:
            raise ValueError('下载版本不是原查询版本的正常后继；拒绝回退或分叉')
        print('已确认main正常前移：'+known[:12]+' → '+commit[:12],flush=True)
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
    def stored_blobs():
        # A stdin --batch-check lookup may try to fetch missing promisor objects.
        # Enumerate only stored objects; keep implicit network access disabled.
        available=inspect('cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype)').stdout
        return {row.split()[0] for row in available.splitlines() if row.endswith(' blob')}
    present=stored_blobs()
    missing=sorted(blobs-present)
    print(f'运行代码：{len(paths)}个文件；需下载{len(missing)}个文件对象，复用{len(blobs)-len(missing)}个缓存对象。',flush=True)
    for start in range(0,len(missing),128):
        fetch('--no-write-fetch-head','--filter=blob:none',url,*missing[start:start+128])
    # Detect missing objects here, with implicit networking disabled.
    if not blobs.issubset(stored_blobs()):
        raise ValueError('运行代码对象不完整，未切换版本')
    remaining()
    return commit,paths


def error_detail(error, stage):
    if isinstance(error,subprocess.TimeoutExpired):return 'timeout',f'{stage}总超时（上限 {error.timeout:g} 秒）'
    if isinstance(error,KeyboardInterrupt):return 'cancelled',stage+'被用户取消'
    if isinstance(error,OSError):return 'io',stage+'发生本地I/O错误：'+safe_text(error)
    return 'git',safe_text(error)


def error_tail(error):
    value=getattr(error,'output_tail',None)
    if value is None:value=getattr(error,'stderr','') or ''
    if isinstance(value,bytes):value=value.decode('utf-8',errors='replace')
    return safe_text(value)[-4000:]


def update(runtime):
    from panel_publication import network
    settings=network.load(runtime)
    network.show(settings)
    with network.use(settings):return _update(runtime)


def _update(runtime):
    from panel_publication import network
    runtime=Path(runtime).resolve()
    with runtime_lock(runtime):
        previous=read_json(runtime/'release.json') if (runtime/'release.json').exists() else None
        repository=Path(previous['repository']) if previous else Path(git(CODE,'rev-parse','--show-toplevel').stdout)
        origin=git(repository,'config','--get','remote.origin.url').stdout.rstrip('/')
        if origin.removesuffix('.git') not in {v.rstrip('/').removesuffix('.git') for v in ORIGINS}:
            raise ValueError('origin不属于已核实的私有GitHub仓库；未修改地址')
        if git(repository,'status','--porcelain','--',SUBDIR).stdout:
            raise ValueError('本项目代码存在本地修改，先保留并处理后再更新；未自动stash或覆盖')
        current=previous['commit'] if previous else git(repository,'rev-parse','HEAD').stdout
        origin=network.routes(network.current())[0]
        if git(repository,'ls-remote','--get-url',origin).stdout!=origin:
            raise ValueError('Git URL重写改变了官方地址；未发送认证信息，请先核对配置')
        backup=runtime/'update_backups'/str(time.time_ns());backup.mkdir(parents=True)
        write_json(backup/'previous.json',dict(pointer=previous,commit=current,repository=str(repository)))
        started=time.monotonic()
        report=dict(status='checking',stage='本地备份',before=current,attempts=[],official_head=None,
                    query={},pointer_written=False,limits=dict(query=QUERY_TIMEOUT_SECONDS,fetch=FETCH_TIMEOUT_SECONDS,archive=ARCHIVE_TIMEOUT_SECONDS),
                    network={k:network.current()[k] for k in ['proxy','git_transport']})
        def save(stage=None,**values):
            if stage is not None:report['stage']=stage
            report.update(values);report['elapsed_seconds']=round(time.monotonic()-started,3)
            write_json(backup/'update.json',report)
            write_json(backup/'attempts.json',dict(attempts=report['attempts'],official_head=report['official_head'],query=report['query']))
        save()
        try:
            print('保存当前版本备份。',flush=True)
            journal=Journal(backup/'backup.log');tick=time.monotonic()
            try:
                journal.write(f'阶段：本地备份；总上限 {ARCHIVE_TIMEOUT_SECONDS} 秒')
                backup_loaded_code(backup/'code.tar')
                journal.write('本地备份完成；字节数 '+str((backup/'code.tar').stat().st_size))
            finally:journal.stream.close()
            save('远端版本查询',backup_elapsed_seconds=round(time.monotonic()-tick,3))
            cache=runtime/'update_cache.git'
            if not cache.exists():cache.mkdir();git(cache,'init','--bare')
            known=None;chosen=None;query_log=backup/'query.log';tick=time.monotonic()
            report['query']=dict(status='running',log=str(query_log));save()
            try:
                print(f'读取GitHub main版本（最多 {QUERY_TIMEOUT_SECONDS} 秒）。',flush=True)
                result=git(repository,'ls-remote',origin,'refs/heads/main',timeout=QUERY_TIMEOUT_SECONDS,
                           progress_log=query_log,stage='远端版本查询')
                match=re.fullmatch(r'([0-9a-f]{40,64})\s+refs/heads/main\s*',result.stdout)
                if not match:raise ValueError('上游没有返回有效main提交')
                known=match[1];report['official_head']=known
                report['query'].update(status='passed',head=known)
                print('已读取原站版本：'+known[:12],flush=True)
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                kind,reason=error_detail(error,'远端版本查询');tail=error_tail(error)
                report['query'].update(status='failed',error_kind=kind,error=reason,last_output=tail)
                if not query_log.exists():
                    journal=Journal(query_log)
                    try:journal.write(reason+'\n'+(tail or safe_text(error)))
                    finally:journal.stream.close()
                print('GitHub版本查询未完成：'+reason,flush=True)
                if tail:print('查询最后输出：'+tail[-800:],flush=True)
                print('查询日志：'+str(query_log)+'；继续尝试配置的官方认证下载线路。',flush=True)
            finally:
                report['query']['elapsed_seconds']=round(time.monotonic()-tick,3);save()
            if known==current:
                print('已与官方main版本一致，无需下载对象。',flush=True);save('完成',status='current',after=current);return
            print('检查下载线路：'+origin,flush=True)
            attempt=dict(url=origin,status='running',stage='对象下载',log=str(backup/'fetch.log'))
            report['attempts'].append(attempt);tick=time.monotonic();save('对象下载')
            try:
                chosen,paths=download_runtime(cache,origin,current,known,backup/'fetch.log')
                attempt.update(status='accepted',commit=chosen)
                report['official_head']=chosen
                report['official_head_source']='verified_query_or_official_authenticated_fetch'
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                kind,reason=error_detail(error,'对象下载');tail=error_tail(error)
                attempt.update(status='failed',error_kind=kind,error=reason,last_output=tail)
                print('此线路未完成：'+reason,flush=True)
                if tail:print('下载最后输出：'+tail[-800:],flush=True)
            finally:
                attempt['elapsed_seconds']=round(time.monotonic()-tick,3)
                attempt['logs']=[str(p) for p in sorted(backup.glob('fetch*.log'))];save()
            if chosen is None:raise ValueError('更新未完成：'+attempt['error']+'；保留当前版本；日志：'+str(backup))
            if chosen==current:
                save('完成',status='current',after=current);print('所选线路与当前版本相同。',flush=True);return
            save('离线归档');tick=time.monotonic();journal=Journal(backup/'archive.log')
            try:
                journal.write(f'阶段：离线归档；总上限 {ARCHIVE_TIMEOUT_SECONDS} 秒；二进制stdout不写入日志')
                data=git(cache,'archive','--worktree-attributes',chosen,*paths,binary=True,timeout=ARCHIVE_TIMEOUT_SECONDS).stdout
                journal.write('归档字节数：'+str(len(data)))
            finally:journal.stream.close()
            save('候选校验',archive_elapsed_seconds=round(time.monotonic()-tick,3));tick=time.monotonic()
            target=runtime/'releases'/chosen;temporary=runtime/'releases'/('.pending_'+uuid4().hex)
            temporary.mkdir(parents=True)
            print('下载通过，正在校验并生成本项目独立版本。',flush=True)
            extract_project(data,temporary)
            for required in ['05_src/task_panel.py','05_src/task_worker.py','05_src/task_runtime.py','06_configs/baseline.json',
                             '04_data/manifests/README.md','bootstrap_server.sh','start_server.sh']:
                if not (temporary/required).is_file():raise ValueError('候选版本尚未包含兼容的新面板：'+required)
            for source in temporary.rglob('*.py'):
                try:compile(source.read_bytes(),str(source),'exec')
                except SyntaxError as error:raise ValueError(f'候选代码语法检查失败：{source.relative_to(temporary)}:{error.lineno}；保留当前版本') from error
            files={str(p.relative_to(temporary)):digest(p) for p in temporary.rglob('*') if p.is_file()}
            write_json(temporary/'release_manifest.json',files)
            if target.exists():
                if read_json(target/'release_manifest.json')!=files or any(digest(target/p)!=sha for p,sha in files.items()):
                    raise ValueError('同名受管版本内容不同，停止覆盖')
                if temporary.resolve().parent!=(runtime/'releases').resolve() or not temporary.name.startswith('.pending_'):
                    raise ValueError('临时发布目录边界不符，拒绝清理')
                shutil.rmtree(temporary)
            else:temporary.rename(target)
            save('版本指针切换',validation_elapsed_seconds=round(time.monotonic()-tick,3))
            write_json(runtime/'release.json',dict(path=str(target),commit=chosen,repository=str(repository),backup=str(backup)))
            report['pointer_written']=True
            selected_entry(runtime)
            save('完成',status='completed',after=chosen)
            print(f'已选择独立版本 {chosen[:12]}；旧任务仍用原快照。备份：{backup}',flush=True)
        except BaseException as error:
            kind,reason=error_detail(error,report['stage'])
            save(status='cancelled' if isinstance(error,KeyboardInterrupt) else 'failed',error_kind=kind,error=reason,last_output=error_tail(error))
            raise
