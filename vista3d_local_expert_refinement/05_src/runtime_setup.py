"""Generic private runtime; read-only panel queries never create this environment."""
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import time
CODE = Path(__file__).resolve().parents[1]
SETTINGS = CODE / '.baseline_settings.json'
DEFAULT_PATHS = {'base': str(Path.home()/'.local/share/vista3d_baseline_runtime'),
                 'work': str(Path.home()/'vista3d_baseline_results'), 'data': ''}
VERSION = '26.7.2-0'
INSTALLER = f'Miniforge3-{VERSION}-Linux-x86_64.sh'
URL = f'https://github.com/conda-forge/miniforge/releases/download/{VERSION}/{INSTALLER}'
SHA256 = '281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05'


def environment(base):
    """只对当前子进程设缓存目录，不改系统配置或 shell 登录文件。"""
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    for key in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(key, None)
    env['PIP_CONFIG_FILE'] = os.devnull
    mapping = {'CONDA_PKGS_DIRS': 'cache/conda_pkgs', 'CONDA_ENVS_PATH': 'envs',
               'PIP_CACHE_DIR': 'cache/pip', 'XDG_CACHE_HOME': 'cache/xdg',
               'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
               'MPLCONFIGDIR': 'cache/matplotlib', 'TMPDIR': 'cache/tmp'}
    for key, relative in mapping.items():
        folder = base / relative
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    env.update(TEMP=env['TMPDIR'], TMP=env['TMPDIR'], PYTHONUNBUFFERED='1',
               PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
               WHOLE_HEART_ENV_PREFIX=str(base / 'envs/vista_baseline_py311'))
    return env


def report_progress(env, phase, message, status='running', **fields):
    """Only a submitted worker writes progress; query imports have no side effects."""
    job = env.get('WHOLE_HEART_JOB')
    print(message, flush=True)
    if not job:
        return
    from server_data import read_json, write_json
    path = Path(job) / 'environment.json'
    try:
        previous = read_json(path)
    except (OSError, ValueError):
        previous = {}
    phase = phase or previous.get('phase', 'installation')
    stamp = datetime.now(timezone.utc).isoformat()
    history = previous.get('history', [])
    if previous.get('phase') != phase:
        if previous.get('phase'):
            history = [*history, {k:previous[k] for k in ('phase','message','status','updated') if k in previous}][-20:]
        previous = {}
    record = dict(previous, phase=phase, message=message, status=status, updated=stamp, history=history, **fields)
    write_json(path, record)


def download_installer(download, env):
    partial = download.with_suffix('.partial')
    report_progress(env, 'download', '下载官方 Miniforge；按已收到字节展示，不估计未知网络等待。')
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as stream:
            total = response.headers.get('Content-Length')
            total = int(total) if total and total.isdigit() else None
            count, last = 0, 0
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                stream.write(chunk)
                count += len(chunk)
                if time.monotonic()-last >= 1:
                    report_progress(env, 'download', '正在下载官方 Miniforge', downloaded_bytes=count, total_bytes=total)
                    last = time.monotonic()
            if total is not None and total != count:
                raise RuntimeError('下载字节数与服务器 Content-Length 不符；保留 partial 文件。')
            report_progress(env, 'download', 'Miniforge 下载完成，下一步校验 SHA256', downloaded_bytes=count, total_bytes=total)
        os.replace(partial, download)
    except BaseException as error:
        report_progress(env, None, 'Miniforge 下载未完成；保留临时文件，续跑时重新下载。', status='failed', error=str(error))
        raise


def ensure_conda(base, env):
    candidate = env.get('WHOLE_HEART_CONDA_EXE') or env.get('CONDA_EXE') or shutil.which('conda', path=env.get('PATH',''))
    if candidate and Path(candidate).is_file():
        report_progress(env, 'conda', '复用已有 Conda 程序；训练环境仍创建在本项目独立 runtime。')
        return Path(candidate)
    prefix = base / 'tools/miniforge3'
    executable = prefix / 'bin/conda'
    if executable.is_file():
        report_progress(env, 'conda', '复用 runtime 内已安装的 Miniforge。')
        return executable
    if prefix.exists():
        raise RuntimeError(f'Conda目录不完整：{prefix}。保留现场，请核查，不自动删除。')
    download = base / 'cache' / INSTALLER
    if not download.exists():
        download_installer(download, env)
    report_progress(env, 'checksum', '校验 Miniforge 官方安装包 SHA256。')
    with download.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != SHA256:
        report_progress(env, None, '安装包校验失败；未执行安装器，原文件保留。', status='failed')
        raise RuntimeError(f'安装包SHA256不符，停止执行：{download}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report_progress(env, 'miniforge', '安装独立 Miniforge；详细输出写入任务总日志。')
    if os.environ.get('WHOLE_HEART_MANAGED') == '1':
        from task_worker import run_child
        run_child(['bash', str(download), '-b', '-p', str(prefix)], env)
    else:
        subprocess.run(['bash', str(download), '-b', '-p', str(prefix)], env=env, check=True)
    if not executable.is_file():
        raise RuntimeError('Miniforge 安装没有生成 conda')
    return executable


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('phase')
    parser.add_argument('message')
    parser.add_argument('--status', default='running')
    args = parser.parse_args()
    report_progress(os.environ, args.phase, args.message, args.status)
