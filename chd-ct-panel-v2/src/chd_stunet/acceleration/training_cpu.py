"""V2 bounded CPU/RAM supply, deterministic consumption, and ready-only prefetch.

Transport is bounded process-pool pipes (NumPy payloads), with no global SHM
cleanup. All work is CPU-only; only processes created by this executor may be
terminated. Resource knobs never change batch, patch, sampling or updates.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import multiprocessing as mp
import os
import random
import signal
import time
import sys
import numpy as np
from .preprocess_cpu import snapshot,policy,GIB
from ..observability.runlog import write_json

_LOADERS=None


@contextmanager
def deterministic_cpu_rng(seed):
    import torch
    from threadpoolctl import threadpool_limits
    python_state,numpy_state,torch_state=random.getstate(),np.random.get_state(),torch.get_rng_state()
    previous_threads=torch.get_num_threads()
    try:
        if previous_threads!=1:torch.set_num_threads(1)
        random.seed(seed);np.random.seed(seed)
        # torch.manual_seed also seeds CUDA. Dataset sampling must NEVER change
        # the training device's RNG, including in serial CPU fallback mode.
        torch.random.default_generator.manual_seed(seed)
        with torch.no_grad(),threadpool_limits(limits=1):yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)
        if previous_threads!=1:torch.set_num_threads(previous_threads)


def sequence_seed(seed,phase,epoch,sequence):
    key=f'{seed}:{phase}:{epoch}:{sequence}'.encode()
    return int.from_bytes(hashlib.blake2b(key,digest_size=4).digest(),'little')


def _initialize(loaders,read_gate,versions=None):
    global _LOADERS
    if os.environ.get('CHD_TRAIN_OWNER_DIR'):
        import psutil
        from pathlib import Path
        write_json(Path(os.environ['CHD_TRAIN_OWNER_DIR'])/f'worker-{os.getpid()}.json',dict(
            pid=os.getpid(),born=psutil.Process().create_time(),parent_pid=os.getppid(),
            nonce=os.environ['CHD_TRAIN_NONCE']))
    # The console sends Ctrl+C to the process group. Only the training parent
    # decides the safe checkpoint boundary; workers must finish/cancel on demand.
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    if versions:
        import importlib.metadata
        if any(importlib.metadata.version(name)!=version for name,version in versions.items()):
            raise RuntimeError('Training environment changed before worker startup')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    import torch,blosc2
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    blosc2.set_nthreads(1)
    threadpool_limits(limits=1)
    _LOADERS=loaders
    for loader in _LOADERS.values():loader.read_gate=read_gate


def _produce(phase,seed):
    began=time.monotonic()
    result=_LOADERS[phase](seed)
    result['supply']=dict(pid=os.getpid(),seconds=time.monotonic()-began,seed=seed,source_file=__file__)
    return result


def worker_estimate(settings,loaders):
    """Reserve import/metadata, augmentation temporaries and queued output bytes."""
    voxels=int(np.prod(settings['patch_size']))*settings['batch_size']
    initial=int(np.prod(np.ceil(np.asarray(settings['patch_size'])*1.9)))*settings['batch_size']
    metadata=sum(values.nbytes for loader in loaders.values() for row in loader.rows.values()
                 for values in row['properties']['class_locations'].values() if isinstance(values,np.ndarray))
    return int(.8*GIB)+metadata+initial*32+voxels*64


def serial_estimate(settings):
    voxels=int(np.prod(settings['patch_size']))*settings['batch_size']
    initial=int(np.prod(np.ceil(np.asarray(settings['patch_size'])*1.9)))*settings['batch_size']
    return initial*32+voxels*64


class BatchPool:
    def __init__(self,loaders,settings,output,log,workers=None,bootstrap=None,versions=None,stage=None,read_gate=None):
        self.loaders,self.settings,self.output,self.log=loaders,settings,output,log
        self.requested_workers=workers
        self.bootstrap=bootstrap;self.versions=versions
        self.executor=None;self.recoveries=0
        self.estimate=worker_estimate(settings,loaders)
        self.serial_estimate=serial_estimate(settings)
        admission_estimate=self.serial_estimate if workers==0 else self.estimate
        self.stage=stage if stage is not None else policy(snapshot(),[dict(estimate=admission_estimate)]*settings['train_steps_per_epoch'],
                          workers if workers not in (None,0) else None)
        self.workers=0 if workers==0 else self.stage['workers']
        if self.workers==0 and (self.stage['cpu_budget']<1 or self.stage['memory_budget_bytes']<self.serial_estimate):
            raise MemoryError('Serial supply also requires an admitted CPU slot and real working-set RAM')
        if self.workers<1 and workers!=0:
            raise MemoryError('No CPU/RAM capacity for one training supply worker; inspect resources or explicitly use --workers 0')
        self.shared_read_gate=read_gate is not None
        self.read_gate=str(read_gate or output/'training_read_gate.json')
        self._open()

    def _open(self):
        self.prefetch=max(1,2*self.workers) if self.workers else 1
        if not self.shared_read_gate:write_json(self.read_gate,dict(slots=max(1,min(2,self.workers))))
        for loader in self.loaders.values():loader.read_gate=self.read_gate
        if self.workers:
            self.executor=ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),
                initializer=_initialize,initargs=(self.loaders,self.read_gate,self.versions))
        import json
        from pathlib import Path
        read_limit=json.loads(Path(self.read_gate).read_text(encoding='utf-8'))['slots']
        scope='全组' if self.shared_read_gate else ''
        self.log.event('supply.policy',f'CPU供数{self.workers}进程，预取上限{self.prefetch}，{scope}读取上限{read_limit}',
                       workers=self.workers,prefetch=self.prefetch,read_slots=read_limit,shared_read_gate=self.shared_read_gate,
                       estimated_worker_bytes=self.estimate,policy=self.stage)

    def produce_serial(self,phase,seed):
        began=time.monotonic();result=self.loaders[phase](seed)
        result['supply']=dict(pid=os.getpid(),seconds=time.monotonic()-began,seed=seed)
        return result

    def submit(self,phase,seed):
        # spawn executes this frozen entry before unpickling the loader. It
        # prepends frozen src to sys.path, so imports cannot use updated live
        # files. Normal unit-test loaders keep their existing bootstrap.
        if self.bootstrap is None:return self.executor.submit(_produce,phase,seed)
        main=sys.modules['__main__'];previous=getattr(main,'__file__',None)
        try:
            main.__file__=str(self.bootstrap)
            return self.executor.submit(_produce,phase,seed)
        finally:
            if previous is None:del main.__file__
            else:main.__file__=previous

    def close(self,force=False):
        if self.executor is None:return
        executor=self.executor;self.executor=None
        if force:
            # CPython 3.11 has no public terminate_workers API. Retain only the
            # Process handles owned by this executor, never a PID search/kill.
            processes=list(getattr(executor,'_processes',{}).values())
            executor.shutdown(wait=False,cancel_futures=True)
            for process in processes:
                if process.is_alive():process.terminate()
            deadline=time.monotonic()+5
            for process in processes:process.join(max(0,deadline-time.monotonic()))
        else:
            executor.shutdown(wait=True,cancel_futures=True)

    def shrink(self):
        if self.recoveries>=3:raise MemoryError('Three CPU supply pressure recoveries exhausted')
        self.close(force=True);self.recoveries+=1
        self.workers=max(0,self.workers//2)
        fresh=policy(snapshot(),[dict(estimate=self.estimate)]*self.settings['train_steps_per_epoch'],
                     max(1,self.workers))
        self.workers=min(self.workers,fresh['workers'])
        if self.workers==0 and (fresh['cpu_budget']<1 or fresh['memory_budget_bytes']<self.serial_estimate):
            raise MemoryError('Serial recovery has no CPU/RAM working-set headroom')
        self._open()


class BatchQueue:
    """Peek is not consumption; commit only after a completed train/val attempt."""
    def __init__(self,pool,phase,epoch,total,start=0):
        self.pool,self.phase,self.epoch,self.total=pool,phase,epoch,total
        self.cursor=start;self.pending={};self._fill()

    def _fill(self):
        if not self.pool.workers:return
        for seq in range(self.cursor,min(self.total,self.cursor+self.pool.prefetch)):
            if seq not in self.pending:
                seed=sequence_seed(self.pool.settings['seed'],self.phase,self.epoch,seq)
                self.pending[seq]=self.pool.submit(self.phase,seed)

    @property
    def ready(self):
        return bool(self.pool.workers and self.cursor in self.pending and self.pending[self.cursor].done())

    def peek(self):
        if self.cursor>=self.total:raise StopIteration
        if not self.pool.workers:
            if self.cursor not in self.pending:
                seed=sequence_seed(self.pool.settings['seed'],self.phase,self.epoch,self.cursor)
                self.pending[self.cursor]=self.pool.produce_serial(self.phase,seed)
            return self.pending[self.cursor]
        result=self.pending[self.cursor].result(timeout=300)
        owned=set(getattr(self.pool.executor,'_processes',{}))
        if result['supply']['pid'] not in owned:
            raise RuntimeError('Batch came from an unowned/redirected compute PID')
        if self.pool.bootstrap is not None:
            from pathlib import Path
            if not Path(result['supply']['source_file']).resolve().is_relative_to(Path(self.pool.bootstrap).parent.resolve()):
                raise RuntimeError('Supply worker imported live code instead of the frozen snapshot')
        return result

    def commit(self):
        if self.cursor not in self.pending:raise RuntimeError('Cannot consume a batch that was not requested')
        del self.pending[self.cursor];self.cursor+=1;self._fill()
