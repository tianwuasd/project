"""Update only this project's managed CODE, preserving local paths and outputs.

Plan is offline. --check stages and validates all changed remote code; --apply
also installs it with per-file backups. There is no automatic environment,
dataset, model, Git branch or sibling-project update.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sys
import subprocess
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
REMOTE='https://api.github.com/repos/tianwuasd/project/contents/chd-ct-panel-v2/'
ROOT_FILES={'run.py','analyze.py','smoke.py','setup_env.py','preprocess_worker.py','update.py',
            'pyproject.toml','README.md','.gitignore','.gitattributes','requirements-analysis.txt',
            'requirements-preprocess.txt','project.local.json.example'}


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


def remote_fetch(name,opener,token):
    """Only the official GitHub API receives auth; errors contain no secret."""
    if name!='code-manifest.json' and not allowed(name):
        raise ValueError('Refusing unmanaged remote path: '+name)
    url=REMOTE+urllib.parse.quote(name,safe='/')+'?ref=main'
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
    args=parser.parse_args()
    if not (args.check or args.apply):
        print(json.dumps(dict(target=str(ROOT),source=args.source or REMOTE,
            scope='this folder code only; local config, outputs, venv and sibling projects preserved',
            next='--check validates incoming code; --apply installs with backups'),indent=2))
        return 0
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':args.proxy,'http':args.proxy}) if args.proxy else urllib.request.ProxyHandler(),NoRedirect())
    source=Path(args.source).resolve() if args.source else None
    token=github_token() if source is None else None
    def fetch(name):
        if source:
            path=(source/name).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Source path escapes its root')
            return path.read_bytes()
        return remote_fetch(name,opener,token)
    manifest_bytes=fetch('code-manifest.json')
    manifest=json.loads(manifest_bytes)
    files=validate_manifest(manifest)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    work=ROOT/'outputs/updates'/stamp
    incoming=work/'incoming'
    changes=[]
    for name,expected in files.items():
        target=(ROOT/name).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError('Target symlink/path escapes project: '+name)
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
            continue
        payload=fetch(name)
        if hashlib.sha256(payload).hexdigest()!=expected:
            raise ValueError('Remote code hash mismatch; no code was changed: '+name)
        staged=incoming/name
        staged.parent.mkdir(parents=True,exist_ok=True)
        staged.write_bytes(payload)
        changes.append(name)
    # Stage everything before replacing any live file. Old complete files and
    # the previous manifest are retained for manual, reviewable rollback.
    if args.apply:
        for name in changes+['code-manifest.json']:
            target=ROOT/name
            if target.is_file():
                backup=work/'backup'/name
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(target,backup)
            if name=='code-manifest.json':
                incoming.mkdir(parents=True,exist_ok=True)
                (incoming/name).write_bytes(manifest_bytes)
            target.parent.mkdir(parents=True,exist_ok=True)
            os.replace(incoming/name,target)
    work.mkdir(parents=True,exist_ok=True)
    record=dict(action='applied' if args.apply else 'checked',changed_files=changes,
                source=str(source) if source else REMOTE,backup=str(work/'backup'),
                obsolete_files_removed=False,environment_modified=False)
    (work/'update-result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
    print('Restart commands after an update; active runs continue with their frozen code contract.')
    return 0


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
