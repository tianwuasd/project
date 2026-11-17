from contextlib import contextmanager
import hashlib
from time import perf_counter

import torch

from ..observability.events import event
from .memory import alive


class CpuReady:
    def __init__(self,context,world,root,owner=None,cancel=None):
        self.world,self.root = world,root
        self.owner,self.cancel = owner,cancel
        self.condition = context.Condition()
        self.generation = context.Value('q',0)
        self.arrived = context.Array('q',[-1]*world)
        self.phases = context.Array('q',[0]*world)

    def require_active(self):
        if self.owner is not None and not alive(self.owner):
            raise RuntimeError('模型协调进程已经退出，停止CPU等候')
        if self.cancel is not None and self.cancel.is_set():
            raise RuntimeError('模型任务已取消，停止CPU等候')

    def wait(self,rank,phase,device=None):
        started = perf_counter()
        self.require_active()
        if device is not None and device.type=='cuda':
            completed_event = torch.cuda.Event()
            completed_event.record(torch.cuda.current_stream(device))
            completed_event.synchronize()
        completed = perf_counter()
        token = int.from_bytes(hashlib.sha256(phase.encode('utf-8')).digest()[:8],'big',signed=True)
        with self.condition:
            generation = self.generation.value
            if self.arrived[rank]==generation:
                raise RuntimeError('同一rank重复进入CPU就绪阶段')
            self.arrived[rank],self.phases[rank] = generation,token
            if all(value==generation for value in self.arrived):
                if len(set(self.phases))!=1:
                    raise ValueError('各rank的CPU就绪阶段或批次不同')
                self.generation.value += 1
                self.condition.notify_all()
            while self.generation.value==generation:
                self.require_active()
                self.condition.wait(timeout=5)
                if self.generation.value==generation:
                    event(self.root,'model_cpu_wait',rank=rank,phase=phase,
                          waiting_seconds=perf_counter()-completed,device_collective_submitted=False)
        event(self.root,'model_cpu_ready',rank=rank,phase=phase,
              device_completion_seconds=completed-started,cpu_wait_seconds=perf_counter()-completed,
              device_collective_submitted=False)


class OrderedSupply:
    def __init__(self,context,shares,slots,ready,root,capacity_bytes=0,estimates=(),unit='item'):
        if slots<1 or not shares or any(value<1 for value in shares):
            raise ValueError('供数名额和训练份额必须为正')
        self.ready,self.root,self.world = ready,root,len(shares)
        self.condition = context.Condition()
        self.slots = slots
        self.batch = context.Value('q',0)
        self.active = context.Value('q',0)
        self.built = context.Array('q',[0]*self.world)
        self.quotas = context.Array('q',shares)
        self.waiting = context.Array('q',[0]*self.world)
        self.reserved = context.Array('q',[0]*self.world)
        self.queued = context.Array('q',[0]*self.world)
        self.consumer_active = context.Array('q',[0]*self.world)
        self.unit = unit
        self.capacity = capacity_bytes
        self.estimates = {row['case_id']:row['workspace_bytes'] for row in estimates}
        self.phase = context.Array('c',256)

    def snapshot(self):
        with self.condition:
            return dict(phase=self.phase.value.decode('utf-8'),batch=self.batch.value,active=self.active.value,reader_slots=self.slots,
                built=list(self.built),quotas=list(self.quotas),waiting=list(self.waiting),
                active_workspace_bytes=sum(self.reserved),workspace_capacity_bytes=self.capacity,
                samples_built_complete=[int(q>0 and b==q) for b,q in zip(self.built,self.quotas)],
                ready_depth=list(self.queued),consumer_active=list(self.consumer_active),queue_unit=self.unit,
                ready_depth_min=0,ready_depth_max=1)

    def start(self,rank,phase,quotas,device=None):
        if len(phase.encode('utf-8'))>=256 or len(quotas)!=self.world or any(value<0 for value in quotas):
            raise ValueError('供数份额需要覆盖全部rank且不能为负')
        self.ready.wait(rank,f'supply-start:{phase}:{quotas}',device)
        if rank==0:
            with self.condition:
                if self.active.value or any(self.waiting) or any(self.queued) or any(self.consumer_active):
                    raise RuntimeError('前一阶段供数尚未结束')
                self.batch.value = 0
                self.phase.value = phase.encode('utf-8')
                self.quotas[:] = quotas
                self.built[:] = [0]*self.world
                self.condition.notify_all()
        self.ready.wait(rank,f'supply-started:{phase}')

    def mark_ready(self,rank):
        with self.condition:
            if self.queued[rank] or not self.built[rank]:
                raise RuntimeError('数据尚未构建或发生重复就绪登记')
            self.queued[rank] = 1
        event(self.root,'model_data_ready',rank=rank,**self.snapshot())

    def deliver(self,rank):
        with self.condition:
            if self.queued[rank]!=1:
                raise RuntimeError('交付前没有已就绪的数据')
            self.queued[rank] = 0
            self.consumer_active[rank] = 1
        event(self.root,'model_data_delivered',rank=rank,**self.snapshot())

    def reserve(self,rank,name):
        self.ready.require_active()
        amount = self.estimates[name]
        started = perf_counter()
        with self.condition:
            if self.reserved[rank] or amount<1 or amount>self.capacity:
                raise MemoryError('病例工作空间预留无效或超过已准入预算')
            while sum(self.reserved)+amount>self.capacity:
                self.ready.require_active()
                self.condition.wait(timeout=1)
            self.ready.require_active()
            self.reserved[rank] = amount
        event(self.root,'model_case_workspace',rank=rank,case_id=name,case_estimate_bytes=amount,
              workspace_wait_seconds=perf_counter()-started,**self.snapshot())

    @contextmanager
    def build(self,rank,batch,position):
        started = perf_counter()
        self.ready.require_active()
        with self.condition:
            if not 0<=position<self.quotas[rank] or batch<self.batch.value:
                raise ValueError('病例构建序号超出当前供数计划')
            if batch==self.batch.value and position<self.built[rank]:
                raise ValueError('病例构建序号已经完成')
            self.waiting[rank] += 1
            while not (batch==self.batch.value and position==self.built[rank] and
                       all(b>=min(position,q) for b,q in zip(self.built,self.quotas)) and
                       self.active.value<self.slots):
                self.ready.require_active()
                self.condition.wait(timeout=5)
            self.waiting[rank] -= 1
            self.active.value += 1
        admitted = perf_counter()
        event(self.root,'model_build_started',rank=rank,batch=batch,position=position,
              admission_wait_seconds=admitted-started,**{k:v for k,v in self.snapshot().items() if k!='batch'})
        finished = False
        try:
            yield
            finished = True
        finally:
            with self.condition:
                self.active.value -= 1
                self.reserved[rank] = 0
                if finished:
                    self.built[rank] += 1
                self.condition.notify_all()
            event(self.root,'model_build_completed',rank=rank,batch=batch,position=position,
                  seconds=perf_counter()-admitted,success=finished,
                  **{k:v for k,v in self.snapshot().items() if k!='batch'})

    def finish(self,rank,phase,device=None):
        batch = self.batch.value
        self.ready.wait(rank,f'supply-consumed:{phase}:{batch}',device)
        if rank==0:
            with self.condition:
                if self.active.value or any(self.queued) or list(self.built)!=list(self.quotas):
                    raise RuntimeError('当前批次未完成构建或消费')
                event(self.root,'model_batch_credit_returned',rank=rank,**self.snapshot())
                self.batch.value += 1
                self.built[:] = [0]*self.world
                self.consumer_active[:] = [0]*self.world
                self.condition.notify_all()
        self.ready.wait(rank,f'supply-released:{phase}:{batch}')
        event(self.root,'model_batch_released',rank=rank,consumed_batch=batch,**self.snapshot())
