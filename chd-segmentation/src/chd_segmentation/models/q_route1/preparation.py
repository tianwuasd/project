from collections import deque
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
import math

from filelock import FileLock

from ...acceleration.blocks import BlockExecution
from ...acceleration.host import host_capacity
from ...acceleration.memory import alive,identity,same_process
from ...acceleration.registry import active_peers
from ...artifacts.store import write_json
from ...observability.events import event


def preparation_options(values):
    defaults = dict(enabled=True,max_threads=None,block_bytes=2*2**30,
                    max_block_bytes=16*2**30,growth_seconds=2.)
    if not isinstance(values,dict) or set(values)-defaults.keys():
        raise ValueError('Q的CPU准备配置字段无效')
    value = {**defaults,**values}
    if type(value['enabled']) is not bool or (value['max_threads'] is not None and
            (type(value['max_threads']) is not int or value['max_threads']<1)):
        raise ValueError('Q准备线程上限必须为空或正整数')
    if any(type(value[k]) is not int or value[k]<1 for k in ('block_bytes','max_block_bytes')) or (
            value['max_block_bytes']<value['block_bytes'] or
            type(value['growth_seconds']) not in (int,float) or not math.isfinite(value['growth_seconds']) or value['growth_seconds']<=0):
        raise ValueError('Q准备块预算及增长间隔无效')
    return value


class ModelCPU:
    def __init__(self,workspace):
        self.workspace = workspace
        self.member = identity()
        self.owner = workspace.guard.owner
        self.root = Path(workspace.config['work_root'])/'resources'
        self.slots,self.fair = 1,1

    def _change(self,wanted):
        with FileLock(self.root/'.resource.lock'):
            path = self.root/'active.json'
            rows = active_peers(path)
            own = next(r for r in rows if same_process(r,self.owner))
            handle = self.workspace.guard.client.handle
            with handle.condition:
                self.slots = max(1,handle.state['world'])
                base_slots = max(self.slots,handle.state['cpu_slots'])
            claims = [r for r in own.get('model_cpu_claims',[]) if alive(r) and not same_process(r,self.member)]
            facts = host_capacity()
            cores = max(0,math.floor(facts['cores']*self.workspace.config['resources']['cpu_fraction'])-
                        sum(r['cpu_slots'] for r in rows if r is not own))
            self.fair = max(0,cores//self.slots)
            spare = max(0,cores-base_slots-sum(max(0,r['threads']-1) for r in claims)+1)
            granted = min(wanted,self.fair,spare)
            if wanted:
                claims.append(dict(self.member,threads=granted))
            own['model_cpu_claims'] = claims
            own['cpu_slots'] = base_slots+sum(max(0,r['threads']-1) for r in claims)
            write_json(path,rows,durable=False)
            return granted

    def request(self,wanted):
        return self._change(wanted)

    def shrink(self,granted):
        with FileLock(self.root/'.resource.lock'):
            path = self.root/'active.json'
            rows = active_peers(path)
            own = next(r for r in rows if same_process(r,self.owner))
            claims = own['model_cpu_claims']
            claim = next(r for r in claims if same_process(r,self.member))
            if not 0<=granted<=claim['threads']:
                raise ValueError('Q准备只能缩减已获准的CPU额度')
            claim['threads'] = granted
            own['model_cpu_claims'] = claims
            handle = self.workspace.guard.client.handle
            with handle.condition:
                base_slots = max(1,handle.state['world'],handle.state['cpu_slots'])
            own['cpu_slots'] = base_slots+sum(max(0,r['threads']-1) for r in claims)
            write_json(path,rows,durable=False)

    def close(self):
        self._change(0)


class Preparation:
    def __init__(self,config,guard,entry,stage=None):
        settings = preparation_options(config.get('cpu_preparation',{}))
        self.guard,self.case_id = guard,entry['case_id']
        self.entry = entry
        self.log_dir = Path(config['run_dir'])/stage if stage else Path(config['run_dir'])
        self.stage = stage or 'cpu_preparation'
        self.last_log,self.started = 0.,perf_counter()
        self.config = dict(guard.config,work_root=config.get('work_root') or config['run_dir'],
            preprocessing=dict(max_case_threads=settings['max_threads'],block_bytes=settings['block_bytes'],
                max_block_bytes=settings['max_block_bytes'],growth_seconds=settings['growth_seconds'],auto_blocks=True))
        self.executor = BlockExecution(self)
        self.executor.cpu = ModelCPU(self)

    def release_pages(self):
        from ...artifacts.cache import release_private_mapping
        import numpy as np
        for value in (self.entry['image'],self.entry.get('label')):
            if isinstance(value,np.memmap):
                release_private_mapping(value)

    def report(self,phase,completed,total):
        now = perf_counter()
        if now-self.last_log>=15 or completed==total:
            event(self.log_dir,'q_cpu_preparation',case_id=self.case_id,phase=phase,
                model='q_route1',stage=self.stage,
                completed=completed,total=total,seconds=now-self.started,**self.executor.metrics())
            self.last_log = now

    def sections(self,shape,halo,operation,consume,per_voxel=512):
        from .training_inputs import grid_plan
        pending = deque([tuple(slice(0,n) for n in shape)])
        runner = self.executor
        runner._select_phase('q_prepare_blocks')
        done,total = 0,math.prod(shape)
        while pending:
            threads,cap = runner._capacity()
            minimum = grid_plan((1,1,1),halo,runner.settings['max_block_bytes'],per_voxel)['minimum_workspace_bytes']
            _,free = runner._available()
            threads = min(threads,max(1,free//max(1,2*minimum)))
            cap = max(cap,minimum)
            if len(pending)==1:
                part = pending[0]
                sizes = tuple(s.stop-s.start for s in part)
                full = grid_plan(sizes,halo,runner.settings['max_block_bytes'],per_voxel)
                if full['extent']==sizes and full['workspace_bytes']<=free:
                    threads,cap = 1,full['workspace_bytes']
            jobs = []
            for _ in range(threads):
                if not pending:
                    break
                part = pending.popleft()
                sizes = tuple(s.stop-s.start for s in part)
                plan = grid_plan(sizes,halo,cap,per_voxel)
                section = tuple(slice(s.start,s.start+n) for s,n in zip(part,plan['extent']))
                for axis in reversed(range(3)):
                    rest = tuple(slice(s.start,section[i].stop) if i<axis else
                        slice(section[i].stop,s.stop) if i==axis else s for i,s in enumerate(part))
                    if all(s.stop>s.start for s in rest):
                        pending.appendleft(rest)
                jobs.append((section,plan['workspace_bytes'],math.prod(plan['extent'])))
            if not runner._apply(jobs,operation,'q_prepare_blocks',consume):
                pending.extendleft(job[0] for job in reversed(jobs))
                runner.wanted = max(1,len(jobs)//2)
                runner.bytes = max(minimum,min(runner.bytes,max(job[1] for job in jobs)//2))
                if len(jobs)==1 and jobs[0][1]<=minimum:
                    with self.guard.admit(minimum,label='q_prepare_minimum') as ready:
                        if ready is None:
                            raise InterruptedError('Q准备等待已取消')
                continue
            done += sum(job[2] for job in jobs)
            self.release_pages()
            self.report('blocks',done,total)

    def map(self,items,amount,operation,consume):
        pending = deque(items)
        runner = self.executor
        runner._select_phase('q_prepare_items')
        total,done = len(pending),0
        if amount*total<=64*2**20:
            began = perf_counter()
            with self.guard.admit(amount,label='q_prepare_small_items') as ready:
                if ready is None:
                    raise InterruptedError('Q小任务准备已取消')
                for item in pending:
                    consume(item,operation(item))
            runner.current_threads = 1
            runner.peak_threads = max(runner.peak_threads,1)
            runner.completed += total
            runner.compute_seconds += perf_counter()-began
            runner.applied_bytes = amount
            runner.peak_job_bytes = max(runner.peak_job_bytes,amount)
            runner.peak_bytes = max(runner.peak_bytes,amount)
            self.report('small_items',total,total)
            return
        while pending:
            threads,cap = runner._capacity()
            threads = min(threads,max(1,cap*threads//amount))
            jobs = [(pending.popleft(),amount,1) for _ in range(min(threads,len(pending)))]
            if not runner._apply(jobs,operation,'q_prepare_items',consume):
                pending.extendleft(job[0] for job in reversed(jobs))
                runner.wanted = max(1,len(jobs)//2)
                if len(jobs)==1:
                    with self.guard.admit(amount,label='q_prepare_item') as ready:
                        if ready is None:
                            raise InterruptedError('Q局部准备等待已取消')
                continue
            done += len(jobs)
            self.report('items',done,total)


@contextmanager
def preparation(config,guard,entry,stage=None):
    settings = preparation_options(config.get('cpu_preparation',{}))
    if guard is None or not settings['enabled']:
        yield None
        return
    runtime = Preparation(config,guard,entry,stage)
    try:
        yield runtime
    finally:
        runtime.executor.close()
