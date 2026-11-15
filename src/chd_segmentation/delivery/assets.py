from pathlib import Path
import os
import shutil
import stat
import subprocess
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from ..artifacts.store import file_digest


STU_SHA = 'f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6'
MED_SHA = 'c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060'
MED_COMMIT = '332f30d420f1d1b08e2a79b3ae6a602458808383'
MED_REVISION = 'e4a6f35edd7e091619cbc0750f462f1574e23955'
MED_URL = f'https://huggingface.co/wanglab/MedSAM2/resolve/{MED_REVISION}/MedSAM2_latest.pt'


def download_file(url, target, size, sha, proxy=None):
    target = Path(target)
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('官方资产只接受无认证信息的HTTPS来源')
    if target.exists():
        if not target.is_file() or target.stat().st_size != size or file_digest(target) != sha:
            raise ValueError('已有下载目标不同，保留原件：'+str(target))
        return dict(path=str(target), reused=True, bytes=size, sha256=sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name+'.pending')
    if pending.exists():
        raise FileExistsError('下载目标存在未完成文件：'+str(pending))
    handler = ProxyHandler({'http': proxy, 'https': proxy}) if proxy else ProxyHandler()
    opener = build_opener(handler)
    try:
        count = 0
        with opener.open(url, timeout=120) as response, pending.open('xb') as stream:
            for block in iter(lambda: response.read(1024**2), b''):
                count += len(block)
                if count > size:
                    raise ValueError('官方下载容量超过固定清单')
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if count != size or file_digest(pending) != sha:
            raise ValueError('官方下载容量或摘要不匹配')
        if target.exists():
            raise FileExistsError('下载目标被其他进程创建，保留现有文件')
        pending.replace(target)
    finally:
        if pending.exists():
            pending.unlink()
    return dict(path=str(target), reused=False, bytes=size, sha256=sha)


def git_command(path, arguments, proxy=None):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith('GIT_') or name in ('SSH_ASKPASS',):
            env.pop(name)
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0')
    command = ['git', '-c', 'credential.helper=', '-c', 'http.extraHeader=',
               '-c', 'http.sslVerify=true', '-c', 'core.autocrlf=false', '-c', 'core.hooksPath='+os.devnull]
    if proxy:
        command.extend(['-c', 'http.proxy='+proxy])
    return subprocess.run([*command, '-C', str(path), *arguments], env=env,
                          check=True, capture_output=True, text=True)


def verify_official(path):
    head = git_command(path, ['rev-parse', 'HEAD']).stdout.strip()
    changed = git_command(path, ['status', '--porcelain', '--untracked-files=no']).stdout.strip()
    if head != MED_COMMIT or changed:
        raise ValueError('已有MedSAM2源码身份不同，保留原目录：'+str(path))


def clone_official(target, proxy=None):
    target = Path(target)
    if target.exists():
        verify_official(target)
        return dict(path=str(target), commit=MED_COMMIT, reused=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name+'.pending-'+uuid4().hex)
    staging.mkdir()
    try:
        git_command(staging, ['init', '--template='])
        git_command(staging, ['remote', 'add', 'origin', 'https://github.com/bowang-lab/MedSAM2.git'])
        git_command(staging, ['fetch', '--depth=1', 'origin', MED_COMMIT], proxy)
        git_command(staging, ['checkout', '--detach', MED_COMMIT])
        verify_official(staging)
        if target.exists():
            raise FileExistsError('MedSAM2目标被其他进程创建')
        staging.rename(target)
    finally:
        if staging.exists():
            if staging.parent != target.parent or staging.is_symlink() or any(p.is_symlink() for p in staging.rglob('*')):
                raise ValueError('官方源码构建目录归属异常，保留待核对')
            for path in staging.rglob('*'):
                if path.is_file():
                    path.chmod(path.stat().st_mode | stat.S_IWRITE)
            shutil.rmtree(staging)
    return dict(path=str(target), commit=MED_COMMIT, reused=False)


def verify_assets(assets):
    missing = []
    for key, size, sha in [('stunet', 117497661, STU_SHA), ('medsam2', 156040129, MED_SHA)]:
        path = Path(assets[key])
        if not path.is_file():
            missing.append(dict(asset=key, path=str(path), preparation='existing_weight' if key == 'stunet' else 'official_download'))
        elif path.stat().st_size != size or file_digest(path) != sha:
            raise ValueError('已有权重摘要不匹配，保留原文件：'+str(path))
    official = Path(assets['official'])
    if not official.exists():
        missing.append(dict(asset='official', path=str(official), preparation='official_git'))
    else:
        verify_official(official)
    dependencies = assets.get('dependencies')
    if dependencies is not None and not Path(dependencies).is_dir():
        missing.append(dict(asset='dependencies', path=str(dependencies), preparation='linux_environment'))
    return missing
