"""Explicit installation into a NEW venv, with one PyPI mirror per attempt."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from . import mirrors
from . import defaults
from .state import now, write
from .updater import CODE_ROOT, command, managed_release


def install(destination, runtime, emit=print):
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError('环境目录已存在；请选新目录，历史任务的环境保持原样')
    settings = mirrors.load(runtime)
    source = managed_release(runtime) or CODE_ROOT
    destination.mkdir(parents=True)
    report = {'created_at': now(), 'environment': str(destination), 'state': 'creating', 'attempts': []}
    try:
        command([sys.executable, '-m', 'venv', str(destination)], timeout=120)
        python = destination / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('PIP_') and key not in {'PYTHONPATH', 'PYTHONHOME'}}
        # Disable persistent extra-index/trusted-host settings, including user
        # configs. This also applies to pip's build-isolation subprocesses.
        env['PIP_CONFIG_FILE'] = os.devnull
        env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
        env['PIP_NO_INPUT'] = '1'
        for name in settings['pypi']:
            url = mirrors.PYPI[name]
            emit('正在安装预处理环境，镜像：' + name + '（单次最多 300 秒）')
            attempt = {'name': name, 'url': url}
            report['attempts'].append(attempt)
            try:
                log = command([str(python), '-m', 'pip', 'install', '--index-url', url,
                               '--timeout', '20', '--retries', '1', '--only-binary=:all:', str(source)],
                              env=env, timeout=300)
                (destination / ('install-' + name + '.log')).write_bytes(log)
                command([str(python), '-c', 'import numpy, scipy, nibabel, psutil, aaoca_server, aaoca_preprocessing'],
                        env=env, cwd=destination, timeout=30)
                attempt['state'] = 'installed'
                report['state'] = 'installed'
                emit('环境安装完成，已停止尝试其他镜像。Python：' + str(python))
                return python
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                attempt.update(state='failed', reason=str(exc))
                emit('本镜像未完成，尝试下一条。')
        raise ValueError('所有安装镜像均失败，请查看新环境目录中的 install-report.json，并使用新目录重试')
    except BaseException as exc:
        report.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        write(destination / 'install-report.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description='AAOCA 预处理环境安装：有序镜像，首个成功即停止')
    parser.add_argument('--env', default=str(defaults.environment_default()), help='尚不存在的新环境目录，默认原服务器 venv')
    parser.add_argument('--runtime', default=defaults.runtime_default(),
                        help='读取此面板的镜像配置；不存在时使用默认顺序')
    args = parser.parse_args(argv)
    try:
        install(args.env, Path(args.runtime).expanduser().resolve())
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print('未完成：' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
