import json
import os
from pathlib import Path
import sys
import time

if len(sys.argv)>3 and sys.argv[3]=='coordinator':
    from dataset1282_analysis.xu_gpu_isolation import configure_coordinator
    configure_coordinator()

if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
    raise ValueError('CPU 隔离失败：拒绝初始化仍可见 GPU 的运行时')

import torch

target = Path(sys.argv[1])
release = Path(sys.argv[2])
result = dict(pid=os.getpid(),visible=os.environ['CUDA_VISIBLE_DEVICES'],count=torch.cuda.device_count(),
              available=torch.cuda.is_available(),initialized=torch.cuda.is_initialized())
target.write_text(json.dumps(result),encoding='utf-8')
deadline = time.monotonic()+30
while not release.exists():
    if time.monotonic()>deadline:
        raise TimeoutError('隔离测试没有收到退出许可')
    time.sleep(.02)
