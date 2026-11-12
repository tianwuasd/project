"""Scoped code releases and rollback. Standard library + system Git only."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from .contracts import atomic_json,digest,read_json,sha256_file
from .observability.status import operation_lock,lock_active
from .storage_paths import assert_output_location

REMOTE='https://github.com/tianwuasd/project.git'
MIRRORS={'gh-proxy':'https://gh-proxy.com/https://github.com/tianwuasd/project.git',
         'gitclone':'https://gitclone.com/github.com/tianwuasd/project.git'}
BRANCH='chd_1282_monai'
SUBDIRECTORY='dataset1282_analysis'
PROTOCOL=1
REQUIRED={'bootstrap.py','run.py','setup_env.py','update.py','smoke.py','requirements-lock.txt',
          'configs/default.json','src/dataset1282_analysis/__init__.py'}
TOP={'bootstrap.py','run.py','setup_env.py','update.py','smoke.py','start.sh','start.cmd',
     'pyproject.toml','requirements-lock.txt','project.local.json.example','README.md','.gitignore','.gitattributes','release-manifest.json'}
DIRECTORIES={'src','configs','tests','docs','scripts'}


def safe_relative(name):
    if '\\' in name or ':' in name or not name or name.startswith('/'):
        raise ValueError('Unsafe release path')
    parts=PurePosixPath(name).parts
    if any(p in ('..','.') or p.rstrip(' .')!=p or re.match(r'(?i)^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)',p) for p in parts):
        raise ValueError('Unsafe release path')
    if len(parts)==1 and name not in TOP: raise ValueError('File is not in publication allowlist: '+name)
    if len(parts)>1 and parts[0] not in DIRECTORIES: raise ValueError('Directory is not in publication allowlist')
    if '__pycache__' in parts or (len(parts)>1 and PurePosixPath(name).suffix not in ('.py','.json','.md','.toml','.txt','.sh','.cmd','.csv')):
        raise ValueError('Unsupported release file type')
    return name


def source_files(root):
    root=Path(root);result=[]
    for top in sorted(TOP-{'release-manifest.json'}):
        if (root/top).is_file(): result.append(root/top)
    for name in sorted(DIRECTORIES):
        if not (root/name).exists(): continue
        for path in (root/name).rglob('*'):
            if '__pycache__' in path.parts: continue
            if path.is_file(): safe_relative(path.relative_to(root).as_posix());result.append(path)
    return result


def write_manifest(root):
    root=Path(root)
    files={p.relative_to(root).as_posix():sha256_file(p) for p in source_files(root)}
    atomic_json(root/'release-manifest.json',{'schema_version':1,'bootstrap_protocol':PROTOCOL,
                'project':SUBDIRECTORY,'files':files})


def verify_release(root):
    root=Path(root).resolve()
    manifest=read_json(root/'release-manifest.json')
    if manifest.get('schema_version')!=1 or manifest.get('bootstrap_protocol')!=PROTOCOL or manifest.get('project')!=SUBDIRECTORY:
        raise ValueError('Incompatible release schema, project or bootstrap protocol')
    files=manifest.get('files',{})
    if not REQUIRED<=set(files): raise ValueError('Missing required release files')
    seen=set()
    for name,expected in files.items():
        safe_relative(name)
        if name.lower() in seen: raise ValueError('Case-colliding release paths')
        seen.add(name.lower())
        path=root/name
        if path.resolve()!=path or not path.is_file() or path.is_symlink(): raise ValueError('Missing or linked release file')
        if path.stat().st_size>8*1024**2 or sha256_file(path)!=expected: raise ValueError('Release content digest mismatch: '+name)
        if path.suffix=='.py': compile(path.read_bytes(),str(path),'exec')
    actual={p.relative_to(root).as_posix() for p in source_files(root)}
    if actual!=set(files): raise ValueError('Unmanifested managed code files')
    return manifest


def _valid_commit(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Expected full immutable commit identity')
    return commit


def _activate_unlocked(home,commit):
    commit=_valid_commit(commit);home=Path(home)
    release=home/'code_versions'/commit
    verify_release(release)
    old=read_json(home/'current.json') if (home/'current.json').exists() else None
    atomic_json(home/'current.json',{'commit':commit,'bootstrap_protocol':PROTOCOL,
                'activated_at':time.time(),'previous':old.get('commit') if old else None})
    return {'active':commit,'previous':old.get('commit') if old else None,'code_root':str(release)}


def activate_release(home,commit):
    with operation_lock(home,'activate'): return _activate_unlocked(home,commit)


def rollback_release(home,commit):
    with operation_lock(home,'rollback'): return _activate_unlocked(home,commit)


def _git(cache,*args,proxy=None,timeout=120,public_only=False,require_filter=False):
    command=['git','-c','core.hooksPath='+str(Path(cache)/'disabled-hooks'),'-c','protocol.file.allow=never']
    if proxy: command+=['-c','http.proxy='+proxy,'-c','https.proxy='+proxy]
    command+=['--git-dir',str(cache),*args]
    env=None
    if public_only:
        command[1:1]=['-c','credential.helper=','-c','http.extraHeader=',
                      '-c','http.extraHeader=Authorization:','-c','http.extraHeader=Cookie:',
                      '-c','http.followRedirects=false']
        env={k:v for k,v in os.environ.items() if not k.startswith(('GIT_CONFIG','GIT_ASKPASS','SSH_ASKPASS'))}
        env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_TERMINAL_PROMPT='0',GIT_ASKPASS='')
        local_config=Path(cache)/'config'
        if local_config.exists():
            audit=subprocess.run(['git','config','--file',str(local_config),'--name-only','--list'],capture_output=True,check=True,timeout=30)
            allowed={'core.repositoryformatversion','core.filemode','core.bare','core.logallrefupdates','core.symlinks','core.ignorecase','remote.origin.url','remote.origin.promisor','remote.origin.partialclonefilter'}
            if any(key.strip().lower() not in allowed for key in audit.stdout.decode().splitlines()):
                raise ValueError('Mirror cache contains configuration outside the unauthenticated allowlist')
    if require_filter:
        env=dict(env or os.environ);env['GIT_TRACE_PACKET']='1'
        command[1:1]=['-c','protocol.version=2']
    result=subprocess.run(command,capture_output=True,timeout=timeout,env=env)
    if result.returncode:
        # Do not emit credential-bearing URLs or the caller environment.
        raise RuntimeError('Git stage failed: '+args[0]+' (exit '+str(result.returncode)+')')
    if require_filter and not any(re.search(rb'packet:\s+\S+<.*\bfilter\b',line) for line in result.stderr.splitlines()):
        raise ValueError('Remote does not advertise partial-clone filtering; refusing a full repository transfer')
    return result.stdout


def official_public_head(proxy=None):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy} if proxy else {}))
    base='https://api.github.com/repos/tianwuasd/project'
    values=[]
    for url in (base,base+'/commits/'+BRANCH):
        request=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'dataset1282-public-updater'})
        with opener.open(request,timeout=30) as response:
            payload=response.read(1024*1024+1)
        if len(payload)>1024*1024: raise ValueError('Official metadata exceeds size limit')
        values.append(json.loads(payload))
    if values[0].get('private') is not False or values[0].get('visibility')!='public':
        raise ValueError('Mirror updates require an officially verified public repository')
    return _valid_commit(values[1]['sha'])


def fetch_plan(home,proxy=None,mirror=None,expected_commit=None):
    home=assert_output_location(home);cache=home/'.update/cache.git';cache.parent.mkdir(parents=True,exist_ok=True)
    if mirror is not None and mirror not in MIRRORS: raise ValueError('Unknown fixed public mirror')
    if expected_commit and not mirror: raise ValueError('An explicit official commit pin requires mirror mode')
    expected=(_valid_commit(expected_commit) if expected_commit else official_public_head(proxy)) if mirror else None
    if not cache.exists():
        subprocess.run(['git','init','--bare',str(cache)],check=True,capture_output=True,timeout=30)
    # Origin is fixed in code; no credentials are copied to another host.
    _git(cache,'config','remote.origin.url',MIRRORS[mirror] if mirror else REMOTE)
    _git(cache,'config','remote.origin.promisor','true')
    _git(cache,'config','remote.origin.partialclonefilter','blob:none')
    _git(cache,'ls-remote','--heads','origin',BRANCH,proxy=proxy,public_only=bool(mirror),require_filter=True)
    _git(cache,'fetch','--depth=1','--filter=blob:none','origin',BRANCH,proxy=proxy,public_only=bool(mirror))
    commit=_git(cache,'rev-parse','FETCH_HEAD').decode().strip();_valid_commit(commit)
    if expected is not None and commit!=expected: raise ValueError('Mirror HEAD differs from verified official HEAD; activation refused')
    listing=_git(cache,'ls-tree','-rz','--full-tree',commit,'--',SUBDIRECTORY,proxy=proxy,public_only=bool(mirror))
    entries=[];total=0
    for record in listing.split(b'\0'):
        if not record: continue
        info,path=record.split(b'\t',1);mode,kind,oid=info.decode().split()
        name=path.decode('utf-8')
        if not name.startswith(SUBDIRECTORY+'/'): raise ValueError('Git tree escaped fixed subdirectory')
        name=safe_relative(name[len(SUBDIRECTORY)+1:])
        if mode not in ('100644','100755') or kind!='blob': raise ValueError('Linked or special release file')
        entries.append({'name':name,'oid':oid,'mode':mode})
    if not entries or 'release-manifest.json' not in [e['name'] for e in entries]: raise ValueError('Project release manifest not found on official branch')
    if len(entries)>1000: raise ValueError('Release file count exceeds project bound')
    return {'remote':REMOTE,'mirror':mirror,'verified_official_head':expected,'head_evidence':'explicit_official_pin' if expected_commit else 'official_api' if mirror else 'official_git', 'branch':BRANCH,'subdirectory':SUBDIRECTORY,'commit':commit,'cache':str(cache),'entries':entries}


def install_release(home,plan,proxy=None):
    home=assert_output_location(home);commit=_valid_commit(plan['commit']);versions=home/'code_versions';versions.mkdir(exist_ok=True)
    target=versions/commit
    if target.exists(): verify_release(target);return target
    with tempfile.TemporaryDirectory(prefix='candidate-',dir=home/'.update') as temporary:
        candidate=Path(temporary);total=0;seen=set();started=time.monotonic()
        for item in plan['entries']:
            if time.monotonic()-started>600: raise TimeoutError('Release download total time budget exceeded')
            name=safe_relative(item['name'])
            if name.lower() in seen: raise ValueError('Duplicate release path')
            seen.add(name.lower())
            payload=_git(plan['cache'],'cat-file','blob',item['oid'],proxy=proxy,timeout=90,public_only=bool(plan.get('mirror')))
            total+=len(payload)
            if len(payload)>8*1024**2 or total>32*1024**2: raise ValueError('Release size limit exceeded')
            path=candidate/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        verify_release(candidate)
        candidate.rename(target)
    atomic_json(home/'.update'/('receipt-'+commit+'.json'),{k:v for k,v in plan.items() if k not in ('entries','cache')})
    return target


def check_local_changes(home):
    home=Path(home)
    verify_release(home)
    result=subprocess.run(['git','-C',str(home),'status','--porcelain','--untracked-files=no','--','.'],capture_output=True,timeout=30)
    if result.returncode==0 and result.stdout.strip(): raise ValueError('Local tracked code changes: commit or preserve them before activation')


def save_baseline(home):
    home=Path(home);manifest=verify_release(home)
    identity=digest(manifest)[:40]
    target=home/'code_versions'/identity
    if not target.exists():
        target.mkdir(parents=True)
        for name in [*manifest['files'],'release-manifest.json']:
            dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(home/name,dest)
        verify_release(target)
    return identity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_mutually_exclusive_group()
    actions.add_argument('--check',action='store_true');actions.add_argument('--rollback')
    p.add_argument('--proxy',help='Process-scoped proxy, e.g. http://127.0.0.1:7897; no global configuration changes')
    p.add_argument('--mirror',nargs='?',const='gh-proxy',default='gh-proxy',choices=sorted(MIRRORS),help='Fixed public unauthenticated mirror (default: gh-proxy); official API HEAD must match exactly')
    p.add_argument('--expected-commit',help='Full commit independently verified on official GitHub; allows public mirror when official API is unreachable')
    args=p.parse_args(argv)
    home=assert_output_location(os.environ.get('DATASET1282_HOME',Path(__file__).resolve().parents[2]))
    try:
        with operation_lock(home,'update'):
            if args.rollback:
                check_local_changes(home)
                candidate=home/'code_versions'/_valid_commit(args.rollback)
                verify_release(candidate)
                _check_dependencies(home,candidate)
                result=_activate_unlocked(home,args.rollback)
            else:
                plan=fetch_plan(home,args.proxy,args.mirror,args.expected_commit)
                current=read_json(home/'current.json')['commit'] if (home/'current.json').exists() else None
                if args.check:
                    result={k:v for k,v in plan.items() if k not in ('entries','cache')}
                    result.update(active=current,action='check_only',activation_performed=False)
                else:
                    check_local_changes(home)
                    candidate=install_release(home,plan,args.proxy)
                    _check_dependencies(home,candidate)
                    baseline=save_baseline(home)
                    result=_activate_unlocked(home,plan['commit']);result['baseline_snapshot']=baseline
            result['ok']=True
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as e:
        result={'ok':False,'error':str(e),'active_pointer_preserved':True}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['ok'] else 1


def _check_dependencies(home,candidate):
    from .config import resolve_config
    from .environment import check_environment,environment_python
    config=resolve_config({},Path(home)/'project.local.json',candidate)
    report=check_environment(environment_python(config['environment']),candidate/'requirements-lock.txt')
    if not report['ok']: raise ValueError('Candidate dependency check failed; environment installation is a separate operation: '+str(report['errors']))
