import json
import os
from pathlib import Path
import subprocess

from .contracts import atomic_json, read_json


CHECK = '''
import importlib.util, json
present = importlib.util.find_spec('cuda') is not None
present = present and importlib.util.find_spec('cuda.bindings') is not None
result = dict(ready=False)
if present:
    from cuda.bindings import driver
    from importlib.metadata import version
    code, = driver.cuInit(0)
    if int(code) != 0: raise RuntimeError(str(code))
    code, count = driver.cuDeviceGetCount()
    if int(code) != 0: raise RuntimeError(str(code))
    code, context = driver.cuCtxGetCurrent()
    if int(code) != 0 or int(context) != 0: raise RuntimeError('查询建立了 CUDA 上下文')
    result = dict(ready=count>0,devices=count,version=version('cuda-bindings'))
print(json.dumps(result))
'''


def probe_binding(python, extra=None):
    from .gpu_environment import child_environment
    env = child_environment()
    if extra:
        env['PYTHONPATH'] = str(extra)
    proc = subprocess.run([str(python),'-B','-c',CHECK],env=env,check=True,
                          capture_output=True,text=True,encoding='utf-8',timeout=30)
    return json.loads(proc.stdout)


def ensure_binding(runtime, selected, install=False, proxy=None):
    from .gpu_environment import child_environment, run_logged
    python = selected['python']
    direct = probe_binding(python)
    if direct['ready']:
        return dict(direct,path=None)
    tag = '.'.join(map(str,selected['python_version'][:2]))
    target = Path(runtime)/('gpu-query-py'+tag)
    marker = target/'.dataset1282-query.json'
    owner = dict(project='dataset1282_analysis',purpose='context-free-gpu-query',python=tag)
    if target.exists():
        if target.is_symlink() or not marker.is_file() or read_json(marker) != owner:
            raise RuntimeError('GPU 查询依赖目录不属于本项目')
        current = probe_binding(python,target)
        if current['ready']:
            return dict(current,path=str(target))
    if not install:
        return dict(ready=False,path=str(target),reason='需要安装 cuda-bindings 查询依赖')
    target.mkdir(parents=True,exist_ok=True)
    atomic_json(marker,owner)
    env = child_environment()
    for key in list(env):
        if key.startswith('PIP_') or key.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):
            env.pop(key,None)
    env['PIP_CONFIG_FILE'] = os.devnull
    if proxy:
        env.update(HTTP_PROXY=proxy,HTTPS_PROXY=proxy)
    command = [python,'-m','pip','--disable-pip-version-check','install','--only-binary=:all:',
               '--target',str(target),'cuda-bindings==12.9.3','--index-url','https://pypi.org/simple']
    with (Path(runtime)/'gpu-query-install.log').open('a',encoding='utf-8') as log:
        run_logged(command,log,env)
    result = probe_binding(python,target)
    if not result['ready']:
        raise RuntimeError('GPU 查询依赖验证失败')
    return dict(result,path=str(target))
