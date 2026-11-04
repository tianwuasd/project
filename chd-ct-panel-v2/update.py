"""Update only this project's managed CODE, preserving local paths and outputs.

Plan is offline. --check stages and validates all changed remote code; --apply
also installs it with per-file backups. There is no automatic environment,
dataset, model, Git branch or sibling-project update.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sys
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
REMOTE='https://api.github.com/repos/tianwuasd/project/contents/chd-ct-panel-v2/'
ROOT_FILES={'run.py','analyze.py','smoke.py','setup_env.py','preprocess_worker.py','update.py',
            'pyproject.toml','README.md','.gitignore','.gitattributes','requirements-analysis.txt',
            'requirements-preprocess.txt','requirements-training.txt','project.local.json.example'}


class UpdateLog:
    """Stdlib-only console/file feedback, including blocked network stages.

    Keep the updater bootstrappable as one file: no installed package or
    newly downloaded helper is needed to display its own update progress.
    """
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('w',encoding='utf-8')
        self.lock=threading.Lock()

    def write(self,message):
        line=f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}'
        with self.lock:
            print(line,flush=True)
            self.stream.write(line+'\n');self.stream.flush()

    @contextmanager
    def waiting(self,label,interval=10):
        began=time.monotonic();stop=threading.Event()
        self.write('开始：'+label)
        def heartbeat():
            while not stop.wait(interval):
                self.write(f'等待：{label}；已运行{time.monotonic()-began:.0f}秒')
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:yield
        finally:stop.set();thread.join()

    def __enter__(self):return self

    def __exit__(self,kind,error,tb):
        if error is not None:
            self.write(f'更新未完成：{type(error).__name__}: {error}')
            with self.lock:
                self.stream.write(''.join(traceback.format_exception(kind,error,tb)));self.stream.flush()
            record=dict(action='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                        error_type=type(error).__name__,error=str(error),log_path=str(self.path))
            self.path.with_name('update-result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        self.stream.close()


def allowed(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        return False
    if name in ROOT_FILES:
        return True
    if not path.parts or path.parts[0] not in {'src','configs','docs','tests','scripts'}:
        return False
    return path.suffix in {'.py','.json','.md','.txt'} and '__pycache__' not in path.parts


def validate_manifest(record):
    files=record.get('files')
    if record.get('project')!='chd-ct-panel-v2' or not isinstance(files,dict) or not files:
        raise ValueError('Unexpected project or empty code manifest')
    for name,sha in files.items():
        if not allowed(name) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Refusing unmanaged/invalid code path: '+name)
    return files


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a repository credential to a redirected hostname."""
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


def github_token():
    """Reuse existing auth without prompting, printing or saving a secret.

    Scope Git's lookup to this official repository. SSH-only installations
    without HTTPS/gh credentials use --source after their normal Git sync;
    this script does not copy keys or change global credential settings.
    """
    for name in ('GH_TOKEN','GITHUB_TOKEN'):
        value=os.environ.get(name,'').strip()
        if value:
            return value
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    git=shutil.which('git')
    if git:
        try:
            result=subprocess.run([git,'-c','credential.interactive=never','credential','fill'],
                input='protocol=https\nhost=github.com\npath=tianwuasd/project.git\n\n',
                capture_output=True,text=True,encoding='utf-8',errors='replace',
                cwd=ROOT,env=env,timeout=20)
            if result.returncode==0:
                fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
                if fields.get('password'):
                    return fields['password']
        except (OSError,subprocess.TimeoutExpired):
            pass
    gh=shutil.which('gh')
    if gh:
        try:
            result=subprocess.run([gh,'auth','token','--hostname','github.com'],
                capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=15)
            if result.returncode==0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError,subprocess.TimeoutExpired):
            pass
    return None


def remote_fetch(name,opener,token,ref='main'):
    """Only the official GitHub API receives auth; errors contain no secret."""
    if name!='code-manifest.json' and not allowed(name):
        raise ValueError('Refusing unmanaged remote path: '+name)
    url=REMOTE+urllib.parse.quote(name,safe='/')+'?'+urllib.parse.urlencode({'ref':ref})
    headers={'Accept':'application/vnd.github.raw+json','User-Agent':'chd-preprocessing-code-updater',
             'X-GitHub-Api-Version':'2022-11-28'}
    if token:
        headers['Authorization']='Bearer '+token
    request=urllib.request.Request(url,headers=headers)
    try:
        with opener.open(request,timeout=30) as response:
            payload=response.read(16*1024**2+1)
    except urllib.error.HTTPError as error:
        if error.code in (401,403,404):
            raise RuntimeError(f'GitHub code request returned {error.code}; check repository access using existing Git HTTPS/gh authentication, or use --source with a synced code folder') from None
        raise
    if len(payload)>16*1024**2:
        raise ValueError('Code file exceeds the updater size limit')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--apply',action='store_true')
    parser.add_argument('--proxy',help='temporary proxy for this updater only')
    parser.add_argument('--source',help='offline source directory with code-manifest.json')
    parser.add_argument('--ref',default='main',help='explicit GitHub branch/tag/commit; default main')
    args=parser.parse_args()
    if not (args.check or args.apply):
        print(json.dumps(dict(target=str(ROOT),source=args.source or REMOTE,ref=args.ref,
            scope='this folder code only; local config, outputs, venv and sibling projects preserved',
            next='--check validates incoming code; --apply installs with backups'),indent=2))
        return 0
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':args.proxy,'http':args.proxy}) if args.proxy else urllib.request.ProxyHandler(),NoRedirect())
    source=Path(args.source).resolve() if args.source else None
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    work=ROOT/'outputs/updates'/stamp
    with UpdateLog(work/'update.log') as feedback:
        feedback.write(f'代码更新：{"安装" if args.apply else "仅检查"}；目标：{ROOT}')
        feedback.write('完整日志：'+str(feedback.path))
        feedback.write('来源：'+(str(source) if source else 'GitHub官方仓库；ref='+args.ref))
        return perform_update(args,source,opener,work,feedback)


def perform_update(args,source,opener,work,feedback):
    """Validate all incoming bytes before replacing files; report each phase."""
    token=None
    if source is None:
        with feedback.waiting('读取已有GitHub访问授权（不显示凭据）'):
            token=github_token()
    def fetch(name):
        if source:
            path=(source/name).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Source path escapes its root')
            return path.read_bytes()
        return remote_fetch(name,opener,token,args.ref)
    with feedback.waiting('读取并验证代码清单'):
        manifest_bytes=fetch('code-manifest.json')
    manifest=json.loads(manifest_bytes)
    files=validate_manifest(manifest)
    incoming=work/'incoming'
    changes=[]
    feedback.write(f'清单版本：{manifest.get("version","未声明")}；需要核对{len(files)}份代码文件')
    for number,(name,expected) in enumerate(files.items(),1):
        target=(ROOT/name).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError('Target symlink/path escapes project: '+name)
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
            feedback.write(f'[{number}/{len(files)}] 已一致：{name}')
            continue
        with feedback.waiting(f'[{number}/{len(files)}] 读取更新：{name}'):
            payload=fetch(name)
        if hashlib.sha256(payload).hexdigest()!=expected:
            raise ValueError('Remote code hash mismatch; no code was changed: '+name)
        staged=incoming/name
        staged.parent.mkdir(parents=True,exist_ok=True)
        staged.write_bytes(payload)
        changes.append(name)
        feedback.write(f'[{number}/{len(files)}] SHA256核验通过，已暂存：{name}')
    # Stage everything before replacing any live file. Old complete files and
    # the previous manifest are retained for manual, reviewable rollback.
    if args.apply:
        feedback.write(f'全部核验通过；备份目录：{work/"backup"}')
        for number,name in enumerate(changes+['code-manifest.json'],1):
            target=ROOT/name
            had_original=target.is_file()
            if had_original:
                backup=work/'backup'/name
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(target,backup)
            if name=='code-manifest.json':
                incoming.mkdir(parents=True,exist_ok=True)
                (incoming/name).write_bytes(manifest_bytes)
            target.parent.mkdir(parents=True,exist_ok=True)
            os.replace(incoming/name,target)
            operation='已备份原有文件并替换' if had_original else '已新建'
            feedback.write(f'[{number}/{len(changes)+1}] {operation}：{name}')
    work.mkdir(parents=True,exist_ok=True)
    record=dict(action='applied' if args.apply else 'checked',changed_files=changes,
                source=str(source) if source else REMOTE,backup=str(work/'backup'),
                ref=args.ref if source is None else None,
                obsolete_files_removed=False,environment_modified=False,log_path=str(feedback.path))
    (work/'update-result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
    feedback.write(f'{"更新完成" if args.apply else "检查完成"}；发现{len(changes)}份代码变化；记录：{work/"update-result.json"}')
    feedback.write('后续命令使用更新后的代码；活动任务仍使用原冻结快照。')
    return 0


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
