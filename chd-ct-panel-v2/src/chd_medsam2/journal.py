from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from chd_stunet.acceleration.preprocess_cpu import interpreter_command
from chd_stunet.observability.locking import try_lock, unlock
from chd_stunet.observability.runlog import CHINA, write_json


@contextmanager
def journal_lock(root):
    with (root/'journal.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0')
            stream.flush()
        deadline = time.monotonic()+5
        while not try_lock(stream):
            if time.monotonic()>=deadline:
                raise TimeoutError('MedSAM2 本地日志写锁等待超时')
            time.sleep(.01)
        try:
            yield
        finally:
            unlock(stream)


def last_sequence(path):
    if not path.is_file():
        return 0
    with path.open('rb') as stream:
        end = stream.seek(0, os.SEEK_END)
        if not end:
            return 0
        position, tail = end, b''
        while position:
            length = min(position, 4096)
            position -= length
            stream.seek(position)
            tail = stream.read(length)+tail
            if not tail.endswith(b'\n'):
                raise ValueError('MedSAM2 本地事件日志最后一行不完整')
            if b'\n' in tail[:-1]:
                break
    return json.loads(tail.splitlines()[-1])['seq']


class TrainingLog:
    def __init__(self, root, task):
        self.root = Path(root)
        self.task = task
        self.started = time.monotonic()
        (self.root/'logs').mkdir(parents=True, exist_ok=True)

    def event(self, event, message, **fields):
        now = datetime.now(CHINA).isoformat(timespec='seconds')
        path = self.root/'logs/events.jsonl'
        line = f'[{now}] {message}'
        with journal_lock(self.root):
            record = dict(seq=last_sequence(path)+1, task=self.task, timestamp=now,
                          elapsed_seconds=time.monotonic()-self.started, event=event, message=message, **fields)
            with path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            with (self.root/'logs/run.log').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(line+'\n')
        print(line, flush=True)

    def status(self, state, **fields):
        with journal_lock(self.root):
            write_json(self.root/'status.json', dict(state=state, task=self.task, pid=os.getpid(),
                       heartbeat=datetime.now(CHINA).isoformat(timespec='seconds'),
                       elapsed_seconds=time.monotonic()-self.started, **fields))


def append_log(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, target.open('a+b') as writer:
        total = reader.seek(0, os.SEEK_END)
        offset = writer.seek(0, os.SEEK_END)
        if offset > total:
            raise ValueError('结果目录日志超过本地记录')
        start = max(0, offset-4096)
        reader.seek(start)
        writer.seek(start)
        if reader.read(offset-start) != writer.read(offset-start):
            raise ValueError('结果目录日志与本地记录前缀冲突')
        writer.write(reader.read(1024*1024))
        return writer.tell()==total


def sync_files(local, output):
    paths = [local/'status.json', local/'launcher.log', local/'logs/events.jsonl', local/'logs/run.log',
             *local.glob('ranks/*/status.json'), *local.glob('ranks/*/logs/events.jsonl'),
             *local.glob('ranks/*/logs/run.log'), *local.glob('control/rank-*.log')]
    blocked, pending = {}, []
    for source in paths:
        if not source.is_file():
            continue
        relative = source.relative_to(local)
        target = output/relative
        try:
            if source.name=='status.json':
                write_json(target, json.loads(source.read_bytes()))
            elif not append_log(source, target):
                pending.append(relative.as_posix())
        except (OSError, ValueError) as error:
            blocked[relative.as_posix()] = str(error)
    return dict(blocked=blocked, pending=pending)


class LogMirror:
    def __init__(self, code, local, output):
        self.local = local
        env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(code/'src'), CUDA_VISIBLE_DEVICES='')
        self.command = [interpreter_command(env), '-m', 'chd_medsam2.journal', str(local), str(output)]
        self.env = env
        self.child = None
        self.kill_requested = False
        self.started = 0.
        self.next_sync = 0.

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.flush()

    def start(self):
        self.child = subprocess.Popen(self.command, env=self.env, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.kill_requested = False
        self.started = time.monotonic()

    def collect(self, timeout):
        child = self.child
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if not self.kill_requested:
                child.kill()
                self.kill_requested = True
            # 内核 I/O 阻塞时进程可能尚未退出，监督进程仍继续运行。
            result = dict(blocked={'probe': '结果目录日志同步超时'}, pending=[])
        else:
            result = (json.loads(stdout) if child.returncode==0 else
                      dict(blocked={'probe': stderr.decode('utf-8', errors='replace')[-2000:]}, pending=[]))
            self.child = None
        self.next_sync = time.monotonic()+(10 if result['blocked'] else 2)
        write_json(self.local/'log-sync.json', dict(result, time=time.time()))
        return result

    def tick(self):
        now = time.monotonic()
        if self.child is not None:
            if self.child.poll() is not None or (not self.kill_requested and now-self.started>=10):
                self.collect(0.1)
        elif now>=self.next_sync:
            self.start()

    def flush(self):
        deadline = time.monotonic()+10
        if self.child is not None:
            self.collect(max(.1, deadline-time.monotonic()))
            if self.child is not None:
                return
        while time.monotonic()<deadline:
            self.start()
            result = self.collect(max(.1, deadline-time.monotonic()))
            if result['blocked'] or not result['pending']:
                break


if __name__=='__main__':
    print(json.dumps(sync_files(*(Path(arg) for arg in sys.argv[1:]))))
