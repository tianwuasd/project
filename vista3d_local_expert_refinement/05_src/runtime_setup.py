"""Generic private runtime; read-only panel queries never create this environment."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
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


def ensure_conda(base, env):
    prefix = base / 'tools/miniforge3'
    executable = prefix / 'bin/conda'
    if executable.is_file():
        return executable
    if prefix.exists():
        raise RuntimeError(f'Conda目录不完整：{prefix}。保留现场，请核查，不自动删除。')
    download = base / 'cache' / INSTALLER
    if not download.exists():
        print('下载官方 Miniforge 安装包；若网络失败，日志会保留具体地址。', flush=True)
        partial = download.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        os.replace(partial, download)
    with download.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != SHA256:
        raise RuntimeError(f'安装包SHA256不符，停止执行：{download}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get('WHOLE_HEART_MANAGED') == '1':
        from task_worker import run_child
        run_child(['bash', str(download), '-b', '-p', str(prefix)], env)
    else:
        subprocess.run(['bash', str(download), '-b', '-p', str(prefix)], env=env, check=True)
    if not executable.is_file():
        raise RuntimeError('Miniforge 安装没有生成 conda')
    return executable
