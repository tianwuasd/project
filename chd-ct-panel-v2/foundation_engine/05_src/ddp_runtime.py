"""Fresh, supervised distributed ranks; importing this controller is stdlib-only.

Targets are importable ``module:function`` callables returning JSON-safe dicts.
This layer owns processes and RNG transport, never model/checkpoint semantics.
Launch from the main thread under the usual ``if __name__ == '__main__'`` guard.
"""
from datetime import timedelta
import importlib
import json
import math
import multiprocessing as mp
from multiprocessing.context import get_spawning_popen
from multiprocessing import reduction
import os
from pathlib import Path
import pickle
import signal
import sys
import threading
import time
import traceback
import uuid


def classify_failure(error):
    """Typed causes only: message text is never evidence of GPU capacity."""
    chain=[];current=error
    while current is not None and id(current) not in {id(e) for e in chain}:
        chain.append(current);current=current.__cause__ or current.__context__
    torch=sys.modules.get('torch')
    oom=getattr(getattr(torch,'cuda',None),'OutOfMemoryError',())
    if oom and any(isinstance(e,oom) for e in chain): category='gpu_oom'
    elif any(isinstance(e,MemoryError) for e in chain): category='cpu_memory'
    elif (type(error).__name__=='EagerRetryRequired' and error.__cause__ is not None
          and any(type(e).__name__ in {'BackendCompilerFailed','InductorError','Unsupported','InvalidBackend'}
                  or type(e).__module__.startswith(('torch._dynamo','torch._inductor')) for e in chain[1:])):
        category='compile_retry'
    else: category='noncapacity'
    return dict(classification=category,error_type=type(error).__name__,message=str(error),
                causes=[dict(type=type(e).__name__,module=type(e).__module__,message=str(e)) for e in chain])


class RankGroupFailure(RuntimeError):
    def __init__(self, message, failures):
        super().__init__(message)
        self.failures=failures
        classes={f.get('classification') for f in failures}
        self.classification=('gpu_oom' if 'gpu_oom' in classes else
                             'compile_retry' if classes and classes <= {'compile_retry'} else 'noncapacity')


class _WindowsSpawnProcess(mp.context.SpawnProcess):
    @staticmethod
    def _Popen(process_obj):
        # Windows creates the OS child before serializing process arguments.
        # Preserve its handle if serialization fails so the normal supervisor
        # still reaps it, instead of losing it before Process.start assigns it.
        from multiprocessing.popen_spawn_win32 import Popen
        popen = Popen.__new__(Popen)
        try:
            Popen.__init__(popen, process_obj)
        except BaseException:
            if getattr(popen, 'pid', None) is not None:
                process_obj._popen = popen
                process_obj._sentinel = popen.sentinel
            raise
        return popen


class _SpawnLock:
    """Reduce only inside Process.start, avoiding resource_sharer DupFd tokens.

    On POSIX the descriptor enters spawn's explicit pass-fd set, so even failed
    serialization cannot leave an extra flock reference in resource_sharer.
    On Windows the OS handle is duplicated directly into the spawning child.
    """
    def __init__(self, fd):
        self.fd = fd

    def __reduce__(self):
        popen = get_spawning_popen()
        if popen is None:
            raise RuntimeError('lock transfer requires active spawn')
        if os.name == 'posix':
            return _receive_posix_lock, (reduction.DupFd(self.fd),)
        import msvcrt
        return _receive_windows_lock, (popen.duplicate_for_child(msvcrt.get_osfhandle(self.fd)),)


def _receive_posix_lock(wrapped):
    return wrapped.detach()


def _receive_windows_lock(handle):
    import msvcrt
    return msvcrt.open_osfhandle(handle, os.O_RDONLY)


def _write_record(path, value):
    """Publish only a complete strict-JSON record in the invocation directory."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _initialize_group(dist, invocation, backend, rank, world_size, timeout_seconds):
    # Pass native filesystem text directly. Torch's POSIX file:// handler does
    # not URL-decode its path, so as_uri() corrupts spaces, Unicode and '%'.
    timeout = timedelta(seconds=timeout_seconds)
    store = dist.FileStore(str(invocation / 'rendezvous'), world_size)
    store.set_timeout(timeout)
    dist.init_process_group(backend=backend, store=store, world_size=world_size,
                            rank=rank, timeout=timeout)


def _rank_main(target, arguments, selected, invocation, rank, backend,
               init_timeout_seconds, transferred_lock):
    invocation = Path(invocation)
    log_dir = invocation / f'rank-{rank}'
    world_size = len(selected)
    # Redirect descriptors as well as Python streams (native backend diagnostics).
    stdout = (log_dir / 'stdout.log').open('w', encoding='utf-8', buffering=1)
    stderr = (log_dir / 'stderr.log').open('w', encoding='utf-8', buffering=1)
    os.dup2(stdout.fileno(), 1)
    os.dup2(stderr.fileno(), 2)
    sys.stdout, sys.stderr = stdout, stderr
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                      CUDA_VISIBLE_DEVICES=','.join(selected) if backend == 'nccl' else '',
                      WHOLE_HEART_DDP_INVOCATION=str(invocation),
                      WHOLE_HEART_DDP_LOG_DIR=str(log_dir))
    from task_runtime import identity
    os.environ['VISTA_DDP_RANK_OWNER']=json.dumps(dict(pid=os.getpid(),identity=identity(os.getpid())))
    if transferred_lock is not None:
        os.environ['WHOLE_HEART_LOCK_FD'] = str(transferred_lock)
    else:
        # A parent-only numeric descriptor is invalid in a fresh interpreter.
        os.environ.pop('WHOLE_HEART_LOCK_FD', None)
    dist = None
    try:
        if os.name=='posix':
            # A private group also owns children killed between start/register.
            os.setsid()
            _write_record(log_dir/'process-group.json',dict(pid=os.getpid(),identity=identity(os.getpid()),
                invocation_id=invocation.name,pgid=os.getpgrp(),sid=os.getsid(0)))
        import torch
        import torch.distributed as dist
        if backend == 'nccl':
            torch.cuda.set_device(rank)
        # A unique FileStore avoids parent CUDA and TCP close/rebind races.
        _initialize_group(dist, invocation, backend, rank, world_size, init_timeout_seconds)
        module, name = target.split(':')
        function = getattr(importlib.import_module(module), name)
        result = function(rank, world_size, arguments)
        if not isinstance(result, dict):
            raise TypeError('distributed target must return a JSON-safe dict')
        # Group shutdown is part of success and remains under parent's timeout.
        dist.destroy_process_group()
        dist = None
        _write_record(log_dir / 'result.json', dict(
            invocation_id=invocation.name, rank=rank, world_size=world_size,
            gpu_uuid=selected[rank], pid=os.getpid(), log_dir=str(log_dir), result=result))
    except BaseException as error:
        diagnostic = traceback.format_exc()
        (log_dir / 'traceback.txt').write_text(diagnostic, encoding='utf-8')
        _write_record(log_dir / 'failure.json', dict(classify_failure(error), rank=rank, gpu_uuid=selected[rank], invocation_id=invocation.name))
        print(diagnostic, file=sys.stderr, flush=True)
        # Do not enter another distributed collective after a local failure.
        raise
    finally:
        # Keep the inherited descriptor open through interpreter/native-thread
        # teardown. The OS closes it at actual process exit, even if the parent
        # is killed while multiprocessing is running its exit finalizers.
        stdout.flush()
        stderr.flush()


def _registered_workers(invocation,owners):
    """Only live children registered by this exact invocation and rank identity."""
    from task_runtime import alive
    job=os.environ.get('WHOLE_HEART_JOB');result=[]
    if not job or invocation is None:return result
    for path in (Path(job)/'processes').glob('*.json'):
        try:
            row=json.loads(path.read_text(encoding='utf-8'));owner=row.get('rank_owner') or {}
            if (str(row.get('role','')).startswith('batch:') and row.get('ddp_invocation')==str(invocation)
                    and owner.get('identity') and owners.get(owner.get('pid'))==owner.get('identity') and alive(row)):
                result.append(row)
        except (OSError,ValueError,TypeError):continue
    return result


def _ownership_identity(pid):
    """Capture birth identity before reaping, including an already-zombie child."""
    from task_runtime import identity
    live=identity(pid)
    if live is not None:return live
    if os.name=='posix':
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            return dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start=fields[19])
        except (OSError,ValueError,IndexError):pass
    return None


def _owned_groups(invocation,owners):
    if os.name!='posix' or invocation is None:return []
    groups=[]
    for path in Path(invocation).glob('rank-*/process-group.json'):
        if path.is_symlink():continue
        try:
            row=json.loads(path.read_text(encoding='utf-8'));pid=row['pid']
            if (row.get('invocation_id')==Path(invocation).name and row.get('pgid')==pid and row.get('sid')==pid
                    and row.get('identity') and owners.get(pid)==row['identity']):groups.append(pid)
        except (OSError,ValueError,TypeError,KeyError):continue
    return groups


def _group_members(groups):
    result=[]
    if not groups:return result
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            if int(fields[2]) in groups and fields[0] not in {'Z','X'}:result.append(int(path.parent.name))
        except (OSError,ValueError,IndexError):continue
    return result


def _signal_group(group,signum,owners):
    current=_ownership_identity(group)
    if current is not None and current!=owners.get(group):
        raise RuntimeError('rank group leader PID identity changed; refusing to signal another process')
    try:os.killpg(group,signum)
    except ProcessLookupError:pass


def _stop_ranks(processes,invocation=None,owners=None):
    """Reap every owned rank before releasing the controller's borrowed lock."""
    from task_runtime import alive
    workers=_registered_workers(invocation,owners or {})
    groups=_owned_groups(invocation,owners or {})
    for group in groups:
        _signal_group(group,signal.SIGTERM,owners or {})
    for row in workers:
        if alive(row):
            try:os.kill(row['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 1
    for process in processes:
        if process.pid is not None:
            process.join(max(0, deadline - time.monotonic()))
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.kill()
    for group in groups:
        _signal_group(group,signal.SIGKILL,owners or {})
    for process in processes:
        if process.pid is not None:
            # Actual exit, not a kill request, is the lock release boundary.
            process.join()
    # No rank can publish new groups/children after this point. Include a
    # handshake that raced with the pre-termination ownership snapshot.
    late_groups=set(_owned_groups(invocation,owners or {}))-set(groups)
    for group in late_groups:_signal_group(group,signal.SIGKILL,owners or {})
    groups=list(set(groups)|late_groups)
    workers=list({r['pid']:r for r in [*workers,*_registered_workers(invocation,owners or {})]}.values())
    deadline=time.monotonic()+1
    while any(alive(r) for r in workers) and time.monotonic()<deadline:time.sleep(.05)
    for row in workers:
        if alive(row):
            try:os.kill(row['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    deadline=time.monotonic()+2
    while any(alive(r) for r in workers) and time.monotonic()<deadline:time.sleep(.05)
    if any(alive(r) for r in workers):raise RuntimeError('owned batch workers have not exited; inherited workflow locks remain held')
    deadline=time.monotonic()+2
    while _group_members(groups) and time.monotonic()<deadline:time.sleep(.05)
    if _group_members(groups):raise RuntimeError('owned rank process-group members still alive; memory cleanup/retry refused')
    if invocation is not None:
        from memory_guard import cleanup_owned_segments
        cleanup_owned_segments(invocation,owners or {})


def _read_result(process, invocation, rank, selected):
    path = invocation / f'rank-{rank}' / 'result.json'
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        expected = dict(invocation_id=invocation.name, rank=rank,
                        world_size=len(selected), gpu_uuid=selected[rank],
                        pid=process.pid, log_dir=str(path.parent))
        if (not isinstance(record, dict)
                or any(record.get(key) != value for key, value in expected.items())
                or not isinstance(record.get('result'), dict)):
            raise ValueError('rank/invocation identity mismatch')
        return record
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f'invalid or missing rank {rank} result: {path}: {error}') from error


def launch_ranks(target, arguments, gpu_uuids, log_dir, *, backend='nccl',
                 timeout_seconds=600, init_timeout_seconds=60, lock_fd=None, memory_guard=None) -> list[dict]:
    """Run a fresh rank per selected UUID and return validated rank envelopes.

    ``gloo`` explicitly hides GPUs; its UUID strings are rank identities only.
    NCCL makes the entire ordered UUID group visible and selects local rank.
    SIGINT/SIGTERM cancel the whole invocation and raise KeyboardInterrupt.
    Caller retains ownership of lock_fd; a private duplicate is held until all
    children actually exit. If omitted, inherit WHOLE_HEART_LOCK_FD when set.
    Failure never reuses stale results from a previous invocation.
    """
    if (not isinstance(target, str) or target.count(':') != 1
            or not all(part and all(item.isidentifier() for item in part.split('.'))
                       for part in target.split(':'))):
        raise ValueError('target must be an importable module:function')
    if backend not in ('nccl', 'gloo'):
        raise ValueError('backend must be nccl or gloo')
    if isinstance(gpu_uuids, (str, bytes)):
        raise TypeError('gpu_uuids must be an ordered sequence')
    selected = list(gpu_uuids)
    if (not selected or any(not isinstance(value, str) or not value or ',' in value for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('gpu_uuids must be nonempty and unique')
    for value in (timeout_seconds, init_timeout_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('timeouts must be finite positive seconds')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('launch_ranks requires the main thread for cancellation handlers')
    # Fail before any child starts on ordinary unpicklable target arguments.
    pickle.dumps(arguments)
    if lock_fd is None and os.environ.get('WHOLE_HEART_LOCK_FD'):
        lock_fd = int(os.environ['WHOLE_HEART_LOCK_FD'])
    invocation = Path(log_dir).resolve() / uuid.uuid4().hex
    held_lock = os.dup(lock_fd) if lock_fd is not None else None
    processes, handlers = [], {}
    owners={}
    cancelled = []
    started = time.monotonic()
    def request_cancel(signum, _frame):
        cancelled.append(signum)
    try:
        invocation.mkdir(parents=True, exist_ok=False)
        _write_record(invocation / 'invocation.json', dict(
            invocation_id=invocation.name, backend=backend, target=target,
            gpu_uuids=selected, timeout_seconds=timeout_seconds,
            init_timeout_seconds=init_timeout_seconds))
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_cancel)
        context = mp.get_context('spawn')
        process_type = _WindowsSpawnProcess if os.name == 'nt' else context.Process
        from task_worker import register_child
        for rank in range(len(selected)):
            if memory_guard:memory_guard.check()
            if cancelled:
                raise KeyboardInterrupt('distributed invocation cancelled')
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(f'distributed invocation timed out; logs: {invocation}')
            (invocation / f'rank-{rank}').mkdir()
            process = process_type(target=_rank_main, name=f'ddp-{invocation.name}-{rank}', args=(
                target, arguments, selected, str(invocation), rank, backend,
                init_timeout_seconds, _SpawnLock(held_lock) if held_lock is not None else None))
            processes.append(process)
            try:
                process.start()
                owners[process.pid]=_ownership_identity(process.pid)
            except BaseException:
                if process.pid is not None:
                    owners[process.pid]=_ownership_identity(process.pid)
                    register_child(process, 'ddp-rank')
                raise
            register_child(process, 'ddp-rank')
        while True:
            if cancelled:
                raise KeyboardInterrupt(f'distributed invocation cancelled: {invocation}')
            if memory_guard:memory_guard.check()
            failures = [(rank, p.exitcode) for rank, p in enumerate(processes)
                        if p.exitcode not in (None, 0)]
            published=list(invocation.glob('rank-*/failure.json'))
            if failures or published:
                records=[]
                for r in range(len(processes)):
                    path=invocation / f'rank-{r}' / 'failure.json'
                    if path.exists():records.append(json.loads(path.read_text(encoding='utf-8')))
                if not records:records=[dict(classification='noncapacity',reason='no structured rank failure')]
                _write_record(invocation/'supervisor-failure.json',dict(invocation_id=invocation.name,
                    exits=[dict(rank=r,pid=processes[r].pid,exit_code=code) for r,code in failures],
                    rank_failures=records,latest_memory=memory_guard.latest if memory_guard is not None else None,
                    scope='exit code and observed memory; SIGKILL alone does not establish OOM'))
                raise RankGroupFailure(f'distributed rank failure {failures}; logs: {invocation}',records)
            if all(p.exitcode == 0 for p in processes):
                return [_read_result(p, invocation, rank, selected) for rank, p in enumerate(processes)]
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(f'distributed invocation timed out; logs: {invocation}')
            time.sleep(.05)
    finally:
        try:
            _stop_ranks(processes,invocation,owners)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if held_lock is not None:
                os.close(held_lock)
            for process in processes:
                if process.pid is not None and not process.is_alive():
                    process.close()


def capture_rank_state(device_rank=None):
    """Capture Python/NumPy/CPU RNG, plus only the requested current CUDA rank."""
    import random
    import numpy as np
    import torch
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch_cpu=torch.get_rng_state().clone())
    if device_rank is not None:
        if torch.cuda.current_device() != device_rank:
            raise ValueError('CUDA RNG capture must use the current rank device')
        state['cuda'] = torch.cuda.get_rng_state(device_rank).clone()
    return state


def restore_rank_state(state, device_rank=None):
    """Restore this rank's state; never enumerate or seed other CUDA devices."""
    import random
    import numpy as np
    import torch
    if ('cuda' in state) != (device_rank is not None):
        raise ValueError('CUDA RNG state and rank device must be supplied together')
    if device_rank is not None and torch.cuda.current_device() != device_rank:
        raise ValueError('CUDA RNG restore must use the current rank device')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device_rank is not None:
        torch.cuda.set_rng_state(state['cuda'], device_rank)


def gather_rank_states(local_state):
    """All-gather small trusted checkpoint RNG objects on the initialized group."""
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError('rank state gathering requires an initialized group')
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local_state)
    return states
