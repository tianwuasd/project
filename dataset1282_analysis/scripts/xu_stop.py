import argparse
import json
from pathlib import Path
import signal
import socket
import sys
import time

import psutil

# 支持在现有工程目录通过 git show 管道运行，先停止旧任务再更新源码。
root = Path.cwd() if __file__ in ('<stdin>','-') else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from dataset1282_analysis.observability.status import process_birth


def stop_run(output, timeout=30):
    out = Path(output).resolve(strict=True)
    lock = out/'.runtime/operation.lock'
    if not lock.exists():
        status = json.loads((out/'status.json').read_text(encoding='utf-8'))
        if status.get('state') not in ('complete','failed','interrupted','paused','stopped_after_stage'):
            raise ValueError('任务缺少持有者身份，拒绝猜测需要终止的进程')
        print('任务已经结束，无活动持有者')
        return
    row = json.loads(lock.read_text(encoding='utf-8'))
    if (row['hostname'] != socket.gethostname() or row['kind'] != 'xu-model-run'
            or Path(row['run_dir']).resolve() != out):
        raise ValueError('输出目录、主机或任务身份不匹配')
    actual = process_birth(row['pid'])
    if actual != row['birth']:
        raise ValueError('任务持有者进程身份已变化，拒绝发送信号')
    process = psutil.Process(row['pid'])
    command = process.cmdline()
    if '--output' not in command or Path(command[command.index('--output')+1]).resolve() != out:
        raise ValueError('进程启动参数与输出目录身份不匹配')
    identities = {row['pid']:row['birth'],**{r['pid']:r['birth'] for r in row.get('workers',[])}}
    for child in process.children(recursive=True):
        birth = process_birth(child.pid)
        if birth is not None:
            identities[child.pid] = birth
    latest = json.loads(lock.read_text(encoding='utf-8'))
    if latest['nonce'] != row['nonce'] or process_birth(row['pid']) != row['birth']:
        raise ValueError('停止前任务身份已变化')
    process.send_signal(signal.SIGTERM)
    deadline = time.monotonic()+timeout
    while True:
        alive = [pid for pid,birth in identities.items() if process_birth(pid)==birth]
        if not alive:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'任务进程仍未退出，禁止重启: {alive}')
        time.sleep(.2)
    print('本任务已停止，登记进程已退出；训练文件保留')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='按输出目录和进程身份正常停止一项 Xu 训练')
    parser.add_argument('output',type=Path)
    stop_run(parser.parse_args().output)
