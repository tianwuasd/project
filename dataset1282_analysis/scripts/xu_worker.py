import json
import os
from pathlib import Path
import runpy
import sys


def main():
    script = Path(sys.argv[1]).resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    if root not in script.parents or script.suffix != '.py':
        raise ValueError('工作入口必须属于当前工程')
    grant = json.loads(sys.stdin.readline())
    if grant.get('token') != os.environ.pop('XU_LAUNCH_TOKEN') or not grant.get('members'):
        raise ValueError('启动许可与本次进程组不匹配')
    os.environ['XU_REGISTERED_PIDS'] = json.dumps(grant['members'])
    sys.stdin.close()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(root/'src'))
    from dataset1282_analysis.xu_resources import GpuResourcesChanged
    from dataset1282_analysis.contracts import atomic_json
    try:
        runpy.run_path(str(script), run_name='__main__')
    except GpuResourcesChanged as error:
        atomic_json(Path(os.environ['XU_RESOURCE_FAILURE']), error.evidence())
        raise


if __name__ == '__main__':
    main()
