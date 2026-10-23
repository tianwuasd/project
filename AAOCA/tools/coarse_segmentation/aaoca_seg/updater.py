"""Bounded public fetch and isolated releases; never rewrite a shared checkout."""
import io
import os
import shutil
import signal
import subprocess
import tarfile
import uuid
from pathlib import Path, PurePosixPath

from .service_state import CODE_ROOT, file_lock, read_json, write_json

OFFICIAL = 'https://github.com/tianwuasd/project.git'
MIRRORS = (OFFICIAL, 'https://gh-proxy.org/' + OFFICIAL, 'https://ghfast.top/' + OFFICIAL,
           'https://gitclone.com/github.com/tianwuasd/project.git')


def git(root, *args, timeout=30):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    child = subprocess.Popen(['git', '-c', 'credential.helper=', '-c', 'http.extraHeader=', '-C', str(root), *args],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                             start_new_session=os.name != 'nt')
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except BaseException:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
        else:
            os.killpg(child.pid, signal.SIGTERM)
        child.communicate()
        raise
    if child.returncode:
        raise ValueError(stderr.decode('utf-8', 'replace')[-1200:])
    return stdout


def extract_release(data, destination):
    prefix = PurePosixPath('AAOCA/tools/coarse_segmentation')
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative == prefix or relative in prefix.parents:
                continue
            try:
                relative = relative.relative_to(prefix)
            except ValueError:
                raise ValueError('发布归档包含意外目录') from None
            if (relative.is_absolute() or '..' in relative.parts or ':' in member.name
                    or '\\' in member.name or member.issym() or member.islnk()):
                raise ValueError('发布归档包含不安全路径')
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError('发布归档包含非普通文件')


def update(work):
    checkout = Path(os.environ.get('AAOCA_SEG_SOURCE_CHECKOUT', CODE_ROOT))
    root = Path(git(checkout, 'rev-parse', '--show-toplevel').decode().strip())
    origin = git(root, 'config', '--get', 'remote.origin.url').decode().strip().rstrip('/')
    accepted = {*MIRRORS, 'https://gh-proxy.com/' + OFFICIAL, 'git@github.com:tianwuasd/project.git'}
    if origin.removesuffix('.git') not in {url.removesuffix('.git') for url in accepted}:
        raise ValueError('仓库 origin 与已知项目不符，停止更新')
    with file_lock(work.root / 'control.lock'):
        work.idle()
        if git(root, 'status', '--porcelain').strip():
            raise ValueError('共享仓库有本地修改，已保留；整理后再更新')
        checkout_head = git(root, 'rev-parse', 'HEAD').decode().strip()
        baseline = read_json(work.root / 'release.json', {}).get('commit') or checkout_head
        expected = None
        try:
            expected = git(root, 'ls-remote', OFFICIAL, 'refs/heads/main', timeout=12).decode().split()[0]
        except (ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        ref = 'refs/aaoca-seg-update/' + uuid.uuid4().hex
        attempts = []
        for url in MIRRORS:
            print('正在检查更新线路：' + url, flush=True)
            try:
                git(root, 'fetch', '--no-tags', url, 'refs/heads/main:' + ref, timeout=45)
                commit = git(root, 'rev-parse', ref).decode().strip()
                if expected and commit != expected:
                    raise ValueError('镜像提交与官方当前提交不一致')
                git(root, 'merge-base', '--is-ancestor', baseline, commit)
                attempts.append({'url': url, 'commit': commit, 'status': 'verified'})
                break
            except (ValueError, subprocess.TimeoutExpired) as error:
                attempts.append({'url': url, 'status': 'failed', 'reason': str(error)})
        else:
            write_json(work.root / 'update-last.json', {'attempts': attempts})
            raise ValueError('更新线路均失败，当前程序保持不变')
        if commit == baseline:
            return '已与官方最新版本一致。' if expected else '此线路与当前版本一致；官方最新状态未确认。'
        if git(root, 'status', '--porcelain').strip() or git(root, 'rev-parse', 'HEAD').decode().strip() != checkout_head:
            raise ValueError('下载期间共享仓库发生变化，保留当前程序')
        release = work.root / 'releases' / commit
        staging = work.root / 'releases' / ('staging-' + uuid.uuid4().hex)
        extract_release(git(root, 'archive', commit, 'AAOCA/tools/coarse_segmentation'), staging)
        if not (staging / 'aaoca_seg' / 'terminal.py').is_file():
            raise ValueError('目标版本没有兼容面板入口')
        backup = work.root / 'backups' / uuid.uuid4().hex
        shutil.copytree(CODE_ROOT, backup, ignore=shutil.ignore_patterns('__pycache__', '.venv'))
        if release.exists():
            raise ValueError('该发布目录已存在；保留供核查，不覆盖')
        staging.rename(release)
        write_json(work.root / 'release.json', {'path': str(release), 'commit': commit,
                   'previous': baseline, 'backup': str(backup), 'official_confirmed': bool(expected), 'attempts': attempts})
        return '新版已安装到独立发布目录。请重新打开面板；旧任务仍使用各自冻结代码。' + ('' if expected else ' 官方最新状态未确认。')
