import json
import os
from pathlib import Path
import sys
import time

from dataset1282_analysis.xu_resources import registered_identities

target = Path(sys.argv[1])
members = json.loads(os.environ['XU_REGISTERED_PIDS'])
verified = registered_identities(members,members)
target.write_text(json.dumps(dict(pid=os.getpid(), parents=json.loads(os.environ['XU_REGISTERED_PIDS']),
                                verified=verified,started_at=time.time())), encoding='utf-8')
if len(sys.argv) > 2:
    if sys.argv[2] == 'wait-peer':
        peer = Path(sys.argv[3])
        deadline = time.monotonic()+15
        while not peer.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('另一个真实工作进程没有写出启动证据')
            time.sleep(.01)
    elif sys.argv[2] == 'resource-change':
        from dataset1282_analysis.xu_resources import GpuResourcesChanged
        raise GpuResourcesChanged('测试注入的资源变化事件', ['cuda:2'])
    elif sys.argv[2] == 'data-error':
        raise ValueError('测试注入的数据错误')
    else:
        time.sleep(float(sys.argv[2]))
