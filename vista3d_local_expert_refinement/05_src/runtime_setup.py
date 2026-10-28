"""Generic private runtime; read-only panel queries never create this environment."""
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import time
from contextlib import contextmanager
import json
import tempfile
CODE = Path(__file__).resolve().parents[1]
SETTINGS = CODE / '.baseline_settings.json'
DEFAULT_PATHS = {'base': str(Path.home()/'.local/share/vista3d_baseline_runtime'),
                 'work': str(Path.home()/'vista3d_baseline_results'), 'data': ''}
VERSION = '26.7.2-0'
INSTALLER = f'Miniforge3-{VERSION}-Linux-x86_64.sh'
URL = f'https://github.com/conda-forge/miniforge/releases/download/{VERSION}/{INSTALLER}'
SHA256 = '281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05'


def verify_dependencies():
    """Check actual runtime imports and resource APIs before any data/GPU work."""
    import importlib
    import importlib.metadata as metadata
    expected = {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
                'nnunetv2': '2.8.1', 'threadpoolctl': '3.6.0'}
    modules = ('torch', 'torchvision', 'nnunetv2', 'numpy', 'scipy', 'nibabel',
               'blosc2', 'filelock', 'threadpoolctl', 'psutil')
    errors, versions, imported = [], {}, []
    for package, version in expected.items():
        try:
            versions[package] = metadata.version(package)
            if versions[package] != version:
                errors.append(f'{package}: expected {version}, found {versions[package]}')
        except Exception as error:
            errors.append(f'{package}: {type(error).__name__}: {error}')
    for module in modules:
        try:
            importlib.import_module(module)
            imported.append(module)
        except Exception as error:
            errors.append(f'{module}: {type(error).__name__}: {error}')
    if errors:
        raise RuntimeError('计算环境检查失败，尚未开始数据处理；请在 e 环境检查中修复：' + '; '.join(errors))
    from task_resources import process_observation
    observation = process_observation()
    if observation['post_case_rss_bytes'] <= 0:
        raise RuntimeError('计算环境检查失败：资源采集未返回有效进程内存')
    return dict(status='passed', versions=versions, imports=imported,
                process_observation=observation, gpu_requested=False)


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


def _install_routes(env):
    # Explicit user policy: direct first, then one runtime-configured proxy.
    yield 'direct', None
    prefix = env.get('WHOLE_HEART_ENV_PREFIX')
    if prefix:
        from panel_publication.network import load
        proxy = load(Path(prefix).parents[1]).get('proxy')
        if proxy:
            yield 'proxy', proxy
            return
    report_progress(env, None, '直连失败且原runtime未配置代理；请在c网络设置中配置后重试。')


def _route_environment(env, proxy):
    selected = dict(env)
    for name in tuple(selected):
        if name.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy','pip_proxy','conda_proxy_servers'}:
            selected.pop(name)
    selected.update(PIP_CONFIG_FILE=os.devnull, PIP_DEFAULT_TIMEOUT='20', PIP_RETRIES='0',
                    CONDA_REMOTE_CONNECT_TIMEOUT_SECS='15', CONDA_REMOTE_READ_TIMEOUT_SECS='30',
                    CONDA_REMOTE_MAX_RETRIES='0', NO_PROXY='' if proxy else '*', no_proxy='' if proxy else '*')
    if proxy:
        for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','PIP_PROXY'):
            selected[name] = proxy
    return selected


@contextmanager
def _conda_route(command, env):
    """One command-scoped rc file; never alter global/user Conda configuration."""
    if len(command)<2 or command[1] != 'env':
        yield env
        return
    proxy = env.get('HTTPS_PROXY','')
    with tempfile.TemporaryDirectory(prefix='install-route-',dir=env.get('TMPDIR')) as temp:
        path=Path(temp)/'condarc'
        path.write_text(json.dumps({'proxy_servers':dict(http=proxy,https=proxy,all=proxy)}),encoding='utf-8')
        selected=dict(env,CONDARC=str(path))
        # Detect explicit inherited/final host-specific rc rules instead of
        # labelling a request direct while Conda silently uses another proxy.
        result=subprocess.run([command[0],'config','--show','proxy_servers','--json'],
            env=selected,capture_output=True,text=True,check=True,timeout=30)
        actual=json.loads(result.stdout)['proxy_servers']
        if any((value or '') != proxy for value in actual.values()):
            raise RuntimeError('Conda外部配置覆盖本次安装线路；未修改原配置，请核对proxy_servers。')
        yield selected


def _install_child(command, env):
    with _conda_route(command,env) as selected:
        if selected.get('WHOLE_HEART_MANAGED') == '1':
            from task_worker import run_child
            return run_child(command,selected)
        return subprocess.run(command,env=selected,check=True)


def run_install_command(command, env):
    last = None
    retry_command = list(command)
    target = None
    if command[1:3] == ['env','create'] and '--prefix' in command:
        target = Path(command[command.index('--prefix')+1])
        if target.exists():
            target = None
    for attempt,(route,proxy) in enumerate(_install_routes(env),1):
        report_progress(env,None,'安装下载：'+('直连官方源' if route=='direct' else '直连失败，使用c中代理重试一次'),
                        network_route=route,network_attempt=attempt)
        try:
            if attempt>1 and target is not None and target.is_dir():
                retry_command[2]='update'
                retry_command=[value for value in retry_command if value not in {'--yes','-y'}]
                report_progress(env,None,'首次创建留下部分环境目录；代理重试改用env update，保留已下载文件。')
            result=_install_child(retry_command,_route_environment(env,proxy))
            report_progress(env,None,'安装下载线路完成：'+route,network_route=route,network_attempt=attempt)
            return result
        except subprocess.CalledProcessError as error:
            if error.returncode < 0 or 128 <= error.returncode <= 192:
                report_progress(env,None,'安装子进程被信号终止；保留退出码，不启动代理重试。',
                                network_route=route,network_attempt=attempt)
                raise
            last=error
            report_progress(env,None,f'安装下载线路失败：{route}；退出码{error.returncode}，详细原因见上方安装输出。',
                            network_route=route,network_attempt=attempt)
    raise last


class _ExplicitProxyHandler(urllib.request.ProxyHandler):
    def proxy_open(self, request, proxy, protocol):
        # The runtime accepts credential-free local HTTP proxies. Do not let
        # an inherited NO_PROXY=* silently bypass this explicit fallback.
        from urllib.parse import urlsplit
        selected = urlsplit(proxy)
        if selected.scheme != 'http' or selected.username or selected.password:
            raise ValueError('安装回退需要c中配置的无凭据HTTP代理')
        request.set_proxy(selected.netloc,selected.scheme)
        return None


def _open_installer(url, proxies, timeout):
    # An explicit empty handler bypasses inherited environment/system proxies.
    return urllib.request.build_opener(_ExplicitProxyHandler(proxies)).open(url,timeout=timeout)


def download_installer(download, env):
    partial = download.with_suffix('.partial')
    report_progress(env, 'download', '下载官方 Miniforge；按已收到字节展示，不估计未知网络等待。')
    last = None
    for attempt,(route,proxy) in enumerate(_install_routes(env),1):
        report_progress(env,'download','Miniforge下载：'+('直连官方源' if route=='direct' else '直连失败，使用c中代理重试一次'),
                        network_route=route,network_attempt=attempt)
        try:
            proxies = {'http':proxy,'https':proxy} if proxy else {}
            _download_installer_once(download,partial,env,proxies)
            return
        except (OSError,RuntimeError) as error:
            last=error
            report_progress(env,None,f'Miniforge下载线路失败：{route}；{type(error).__name__}: {error}',
                            network_route=route,network_attempt=attempt)
    report_progress(env,None,'Miniforge下载未完成；保留partial文件。',status='failed',error=str(last))
    raise last


def _download_installer_once(download, partial, env, proxies):
    try:
        with _open_installer(URL,proxies,20) as response, partial.open('wb') as stream:
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
        report_progress(env, None, '本次Miniforge下载未完成，保留临时文件。',error=str(error))
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
    import sys
    if len(sys.argv)>2 and sys.argv[1]=='--download':
        run_install_command(sys.argv[2:],os.environ)
        raise SystemExit(0)
    if sys.argv[1:] == ['--check-dependencies']:
        import json
        print(json.dumps(verify_dependencies(), ensure_ascii=False), flush=True)
        raise SystemExit(0)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('phase')
    parser.add_argument('message')
    parser.add_argument('--status', default='running')
    args = parser.parse_args()
    report_progress(os.environ, args.phase, args.message, args.status)
