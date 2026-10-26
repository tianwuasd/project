"""Owned CUDA keepers, live visibility intersection and one-second idle discovery.

Program leases coordinate this panel's jobs. They are not administrator-level GPU
reservations. No external process is signalled or treated as ours by user name.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from server_data import read_json,write_json
from task_runtime import identity,alive


def verified_keepers(path):
    try:record=read_json(path)
    except (OSError,ValueError):return {}
    verified={}
    for row in record.get('keepers',[]):
        try:
            registry=read_json(Path(path).parent.parent/f"processes/{row['pid']}.json")
            if (row.get('status')=='ready' and registry.get('role')=='gpu-keeper'
                    and registry.get('identity')==row.get('identity')
                    and registry.get('pid')==row.get('pid') and alive(row) and alive(row['owner'])):
                verified[row['gpu_uuid']]=row
        except (OSError,ValueError,KeyError,TypeError):continue
    return verified


class GPUFleet:
    def __init__(self,job,runtime,gpu_memory=20,python=None):
        self.job=Path(job);self.runtime=Path(runtime);self.memory=gpu_memory
        self.path=self.job/'diagnostics/gpu-leases.json'
        self.processes={};self.stop=threading.Event();self.thread=None
        self.error=None;self.cooldown={};self.lock=threading.RLock()
        self.python=str(python or sys.executable)
        self._release_seen=None
        self.generation=uuid4().hex
        self.discovery_enabled=True

    def start(self):
        self.thread=threading.Thread(target=self._watch,name='gpu-discovery',daemon=True)
        self.thread.start();return self

    def _publish(self):
        rows=[]
        for uuid,(process,stream,ready) in self.processes.items():
            if process.poll() is None and ready.exists():
                row=read_json(ready)
                if row['pid']==process.pid and alive(row):rows.append(row)
        write_json(self.path,dict(schema_version=1,updated=time.time(),keepers=rows,
            generation=self.generation,discovery_enabled=self.discovery_enabled,release_ack=self._release_seen,
            policy='physical ∩ allocated/visible ∩ twice-idle; 1s sampling; program leases only',
            error=str(self.error) if self.error else None))

    def _watch(self):
        from evaluation_gpu import snapshot,allowed_ids,idle_reason
        stable={}
        last_sample=last_cpu=0
        settings=dict(allowed_gpus=None,min_free_mib=6000,
            planner_gpu_memory_gib=self.memory,max_utilization_percent=10)
        try:
            while not self.stop.wait(1):
                with self.lock:
                    request=self.path.with_name('gpu-release-request.json')
                    if request.exists():
                        command=read_json(request)
                        if command.get('generation')==self.generation and command.get('id')!=self._release_seen:
                            if command.get('final'):self.discovery_enabled=False
                            used=set(self.processes)-set(command['gpu_uuids'])
                            if command.get('final'):used=set()
                            self.release_unused(used)
                            self._release_seen=command['id']
                    for uuid,(process,stream,_) in list(self.processes.items()):
                        if process.poll() is not None:
                            stream.close();del self.processes[uuid];self.cooldown[uuid]=time.monotonic()+60
                    if not self.discovery_enabled:
                        stable.clear();self._publish()
                        continue
                    rows=snapshot();permitted=allowed_ids(rows,settings)
                    if not permitted:raise ValueError('服务器实际显卡与本次可见分配范围没有交集')
                    for row in rows:
                        uuid=row['uuid']
                        capacity=row['total_mib']+64>=self.memory*1024
                        eligible=(capacity and uuid in permitted and uuid not in self.processes
                            and time.monotonic()>=self.cooldown.get(uuid,0)
                            and idle_reason(row,settings) is None)
                        stable[uuid]=stable.get(uuid,0)+1 if eligible else 0
                        if stable[uuid]>=2:
                            self._acquire(uuid);stable[uuid]=0
                    if not any(r['uuid'] in permitted and r['total_mib']+64>=self.memory*1024 for r in rows):
                        raise ValueError('分配范围内没有显存容量匹配本次网络规划的GPU；未启动训练')
                    if time.monotonic()-last_sample>=5:
                        from runtime_observations import event
                        event(self.job,'gpu.sample',rows=rows)
                        last_sample=time.monotonic()
                    if time.monotonic()-last_cpu>=30:
                        from task_resources import probe
                        from runtime_observations import event
                        event(self.job,'cpu_memory.sample',snapshot=probe())
                        last_cpu=time.monotonic()
                    self._publish()
        except BaseException as error:
            self.error=error
            with self.lock:self._publish()

    def _acquire(self,uuid):
        from task_worker import child_options,register_child
        directory=self.job/'diagnostics/gpu-keepers';directory.mkdir(parents=True,exist_ok=True)
        ready=directory/f'{uuid}.json';ready.unlink(missing_ok=True)
        stream=(directory/f'{uuid}.log').open('a',encoding='utf-8')
        args=[self.python,'-u','-B',str(Path(__file__).resolve()),'--keeper',uuid,
            '--runtime',str(self.runtime),'--ready',str(ready),
            '--owner',json.dumps(dict(pid=os.getpid(),identity=identity(os.getpid())))]
        try:
            process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,**child_options())
            self.processes[uuid]=(process,stream,ready)
            register_child(process,'gpu-keeper')
        except BaseException:
            stream.close();raise

    def devices(self,wait=False):
        while True:
            if self.error:raise RuntimeError('自动GPU探测失败：'+str(self.error)) from self.error
            rows=verified_keepers(self.path)
            if rows or not wait:return list(rows)
            print('CPU前置已完成；等待当前分配范围内空闲GPU，1秒重查。',flush=True)
            if self.stop.wait(1):raise InterruptedError('GPU保留组已停止')

    def resume_discovery(self):
        """Only an explicitly selected later GPU phase can resume this invocation."""
        with self.lock:
            self.discovery_enabled=True
            self._publish()

    def release_unused(self,used):
        with self.lock:
            for uuid in set(self.processes)-set(used):
                process,stream,_=self.processes.pop(uuid)
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                stream.close();self.cooldown[uuid]=time.monotonic()+600
            self._publish()

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=35)
        if self.thread and self.thread.is_alive():raise RuntimeError('GPU探测线程尚未退出，保留进程记录')
        self.release_unused([])


def keep(uuid,runtime,ready,owner):
    # Inherited workflow FD remains open until OS process exit. Per-GPU flock
    # additionally prevents two cooperating panels racing between idle probes.
    import fcntl
    lockdir=Path(runtime)/'gpu_locks';lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/f'{uuid}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from evaluation_gpu import query
        if any(row[0]==uuid for row in query('gpu_uuid,pid','compute-apps')):
            raise RuntimeError('二次核对已有外部GPU进程，放弃申请')
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        import torch
        torch.set_num_threads(1)
        token=torch.empty(32*1024**2,dtype=torch.uint8,device='cuda')
        torch.cuda.synchronize()
        record=dict(status='ready',gpu_uuid=uuid,pid=os.getpid(),identity=identity(os.getpid()),
            owner=owner,held_bytes=token.numel(),created=time.time())
        write_json(ready,record)
        stopping=threading.Event()
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stopping.set())
        while alive(owner) and not stopping.wait(.5):pass
        Path(ready).unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--keeper',required=True)
    parser.add_argument('--runtime',required=True);parser.add_argument('--ready',required=True)
    parser.add_argument('--owner',required=True);args=parser.parse_args()
    keep(args.keeper,args.runtime,args.ready,json.loads(args.owner))
