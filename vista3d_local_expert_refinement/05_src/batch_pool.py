"""Ordered, bounded, sequence-seeded CPU batch generation.

The loader must be a picklable random sampler: each batch depends on the supplied
seed and immutable loader configuration. Global Python/NumPy/Torch CPU RNGs are
seeded automatically. A loader with private RNG/state must implement
``reseed(seed)`` (or ``reset_rng(seed)``), resetting *all* state affecting its next
batch. Arbitrary iterator counters, open resources, external randomness and
nondeterministic kernels cannot be reconstructed generically. In particular,
ordinary stateful finite iterators are not automatically made replayable.

State records the next *consumed* sequence, excluding speculative prefetch.
Determinism is independent of worker count for loaders satisfying this contract.
Do not change loader configuration/software between save and resume. Serial
execution preserves parent global RNG streams, but cannot forcibly interrupt a
blocking loader; use workers > 0 (or an outer process deadline) for cancellation.
Loaders and hooks must stay CPU-only. This iterator is single-consumer;
concurrent parent RNG use is unsupported.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import random
import sys
import threading
import time
import traceback


_OPTIONAL_RNG_MODULES = None


def private_memory_bytes():
    """Linux private resident snapshot; shared source pages are not multiplied."""
    if not sys.platform.startswith('linux'): return None
    try:
        from pathlib import Path
        rows=Path('/proc/self/smaps_rollup').read_text().splitlines()
        return sum(int(line.split()[1])*1024 for line in rows if line.startswith(('Private_Clean:','Private_Dirty:')))
    except (OSError,ValueError): return None


def shared_free_bytes():
    """Actual POSIX transport capacity, not host RAM; Windows has no /dev/shm."""
    if not sys.platform.startswith('linux'): return None
    try:
        stat = os.statvfs('/dev/shm')
        return stat.f_bavail * stat.f_frsize
    except OSError: return None


def allocate_shared(size):
    """Reserve owned pages before child write; never risk a catchless SIGBUS."""
    free = shared_free_bytes()
    if free is not None and size > free:
        raise OSError('shared_live_capacity')
    linux = sys.platform.startswith('linux')
    if linux and (free is None or not hasattr(os, 'posix_fallocate')):
        raise OSError('shared_safe_reservation_unavailable')
    from multiprocessing.shared_memory import SharedMemory
    # Register the intended unique name before creation. A killed rank cannot
    # unlink its buffers; its supervisor cleans only this invocation's names.
    invocation=os.environ.get('WHOLE_HEART_DDP_INVOCATION')
    owner=os.environ.get('VISTA_DDP_RANK_OWNER')
    if invocation and owner:
        from pathlib import Path
        from uuid import uuid4
        from server_data import write_json
        name='vista_'+Path(invocation).name+'_'+uuid4().hex
        write_json(Path(invocation)/'shared-memory'/f'{name}.json',dict(name=name,size=size,rank_owner=json.loads(owner)))
        segment=SharedMemory(name=name,create=True,size=size)
    else:segment = SharedMemory(create=True, size=size)
    try:
        if linux:
            # fallocate is atomic with respect to ENOSPC races across all ranks,
            # pools and unrelated users. Unsupported tmpfs also falls back.
            os.posix_fallocate(segment._fd, 0, size)
        return segment
    except BaseException:
        segment.close()
        segment.unlink()
        raise


def deterministic_seed(seed, epoch, rank, stage, sequence):
    """Stable 64-bit seed; never use Python's process-randomized hash()."""
    coordinates = json.dumps([int(seed), int(epoch), int(rank), str(stage), int(sequence)],
                             ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return int.from_bytes(hashlib.sha256(b'vista-batch-v1:' + coordinates).digest()[:8], 'big')


def _rng_modules():
    global _OPTIONAL_RNG_MODULES
    if _OPTIONAL_RNG_MODULES is None:
        modules = []
        for name in ('numpy', 'torch'):
            try:
                modules.append(importlib.import_module(name))
            except ModuleNotFoundError as error:
                if error.name != name:
                    raise  # An installed but broken dependency is not "optional absent".
                modules.append(None)
        _OPTIONAL_RNG_MODULES = tuple(modules)
    return _OPTIONAL_RNG_MODULES


@contextmanager
def _seeded(seed, preserve=False):
    python_state = random.getstate() if preserve else None
    existing_np = sys.modules.get('numpy')
    existing_torch = sys.modules.get('torch')
    numpy_state = existing_np.random.get_state() if preserve and existing_np is not None else None
    torch_state = existing_torch.get_rng_state() if preserve and existing_torch is not None else None
    np, torch = existing_np, existing_torch
    try:
        np, torch = _rng_modules()
        if preserve:
            if numpy_state is None and np is not None:
                numpy_state = np.random.get_state()
            if torch_state is None and torch is not None:
                torch_state = torch.get_rng_state()
        random.seed(seed)
        if np is not None:
            np.random.seed(seed % (2 ** 32))
        if torch is not None:
            # torch.manual_seed also touches CUDA's lazy seed state. Only seed CPU.
            torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        if preserve:
            random.setstate(python_state)
            if numpy_state is not None:
                np.random.set_state(numpy_state)
            if torch_state is not None:
                torch.set_rng_state(torch_state)


def _generate(loader, seed, preserve=False):
    with _seeded(seed, preserve):
        hook = getattr(loader, 'reseed', None)
        if not callable(hook):
            hook = getattr(loader, 'reset_rng', None)
        if callable(hook):
            hook(seed)
        return next(loader)


def _spawn_lock_handle(handle):
    return handle


class _SpawnLockFd:
    """Borrow the parent's FD; transfer it only through active spawn pickling."""

    def __init__(self, fd):
        self.fd = fd

    def __reduce__(self):
        from multiprocessing import context, reduction
        if context.get_spawning_popen() is None:
            raise RuntimeError('lock FD transfer requires active spawn serialization')
        # During Process.start(), DupFd uses Popen's inherited-FD list. Calling
        # it earlier uses resource_sharer's parent cache, whose duplicate would
        # leak if the child dies before detach(). No parent-side cache is used.
        return _spawn_lock_handle, (reduction.DupFd(self.fd),)


def _shared_encode(batch):
    """Describe dense CPU leaves; tensor bytes never enter the metadata pickle."""
    np, torch = _rng_modules()
    arrays = []
    def encode(value):
        kind = None
        if torch is not None and isinstance(value, torch.Tensor):
            if value.device.type != 'cpu' or value.layout != torch.strided:
                raise ValueError('unsupported_tensor')
            try: array = value.detach().contiguous().numpy()
            except (TypeError, RuntimeError) as error: raise ValueError('unsupported_dtype') from error
            kind = 'torch'
        elif np is not None and isinstance(value, np.ndarray):
            if value.dtype.hasobject: raise ValueError('object_array')
            # dtype.str drops structured field names/layout (and subarray
            # information). Keep those values lossless through byte-pipe pickle.
            if value.dtype.fields is not None or value.dtype.subdtype is not None:
                raise ValueError('structured_or_subarray_dtype')
            if value.dtype.metadata is not None:
                raise ValueError('dtype_metadata')
            array = np.ascontiguousarray(value); kind = 'numpy'
        if kind:
            offset = sum(a.nbytes for a in arrays)
            arrays.append(array)
            return _ArrayRef(kind, array.dtype.str, array.shape, offset)
        named_tuple = isinstance(value, tuple) and isinstance(getattr(value, '_fields', None), tuple)
        if isinstance(value, (dict, list, tuple)) and type(value) not in (dict, list, tuple) and not named_tuple:
            # Built-in reconstruction loses defaultdict factories and other
            # subclass behavior. Only namedtuple has an explicit decoder here.
            raise ValueError('unsupported_container_subclass')
        if isinstance(value, dict): return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list): return [encode(item) for item in value]
        if isinstance(value, tuple):
            values = [encode(item) for item in value]
            return type(value)(*values) if hasattr(value, '_fields') else tuple(values)
        return value
    encoded = encode(batch)
    return encoded, arrays, sum(a.nbytes for a in arrays)


class _ArrayRef:
    def __init__(self, kind, dtype, shape, offset):
        self.kind, self.dtype, self.shape, self.offset = kind, dtype, shape, offset


def _shared_decode(value, buffer):
    if isinstance(value, _ArrayRef):
        np, torch = _rng_modules()
        # Parent-owned copies allow batches to outlive reuse, pause and close.
        array = np.ndarray(value.shape, dtype=value.dtype, buffer=buffer, offset=value.offset).copy()
        return torch.from_numpy(array) if value.kind == 'torch' else array
    if isinstance(value, dict): return {k: _shared_decode(v, buffer) for k, v in value.items()}
    if isinstance(value, list): return [_shared_decode(v, buffer) for v in value]
    if isinstance(value, tuple):
        values = [_shared_decode(v, buffer) for v in value]
        return type(value)(*values) if hasattr(value, '_fields') else tuple(values)
    return value


def _worker_main(loader_bytes, connection, lock_handle, worker_threads=1):
    """Spawn target; owned duplicate closes normally, without explicit flock unlock."""
    lock_fd = None
    sequence = None
    loader = None
    shared = None
    try:
        if lock_handle is not None:
            lock_fd = lock_handle.detach()
            os.environ['WHOLE_HEART_LOCK_FD'] = str(lock_fd)
        while True:
            task = connection.recv()
            if task is None:
                return
            sequence, seed, buffer_info = task
            try:
                status = 'ok'
                if loader is None:
                    # Open resources only after inheriting the workflow lock.
                    loader = pickle.loads(loader_bytes)
                    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                        os.environ[name] = str(worker_threads)
                    _, torch = _rng_modules()
                    if torch is not None: torch.set_num_threads(worker_threads)
                    hook = getattr(loader, 'configure_worker_threads', None)
                    if callable(hook): hook(worker_threads)
                batch = _generate(loader, seed)
                mode, reason, tensor_bytes = 'pipe', None, 0
                try:
                    encoded, arrays, tensor_bytes = _shared_encode(batch)
                    if tensor_bytes:
                        if buffer_info is None or buffer_info[1] < tensor_bytes:
                            connection.send_bytes(pickle.dumps(('buffer_request', sequence, tensor_bytes)))
                            buffer_info, reason = connection.recv()
                        if buffer_info:
                            from multiprocessing.shared_memory import SharedMemory
                            if shared is None or shared.name != buffer_info[0]:
                                if shared is not None: shared.close()
                                shared = SharedMemory(name=buffer_info[0])
                            offset = 0
                            for array in arrays:
                                shared.buf[offset:offset+array.nbytes] = memoryview(array).cast('B')
                                offset += array.nbytes
                            batch, mode = encoded, 'shared'
                except (ValueError, OSError) as error:
                    reason = str(error)
                observer = getattr(loader, 'observations', None)
                observations = observer() if callable(observer) else {}
                _, torch = _rng_modules()
                observations.update(pid=os.getpid(), worker_private_bytes=private_memory_bytes(), worker_threads=worker_threads,
                                    torch_threads=torch.get_num_threads() if torch is not None else None)
                payload = dict(batch=batch, mode=mode, fallback_reason=reason,
                               tensor_bytes=tensor_bytes, worker=observations)
                packet = pickle.dumps(('ok', sequence, payload), protocol=pickle.HIGHEST_PROTOCOL)
            except StopIteration:
                status = 'stop'
                packet = pickle.dumps(('stop', sequence, None))
            except BaseException:
                status = 'error'
                packet = pickle.dumps(('error', sequence, traceback.format_exc()))
            connection.send_bytes(packet)
            if status != 'ok':
                return
    except BaseException:
        try:
            connection.send_bytes(pickle.dumps(('error', sequence, traceback.format_exc())))
        except (OSError, EOFError):
            pass
    finally:
        if shared is not None: shared.close()
        connection.close()
        if lock_fd is not None:
            os.close(lock_fd)


def _read_results(connection, results):
    """Keep potentially partial/large pipe reads off the bounded-wait consumer."""
    try:
        while True:
            wire = connection.recv_bytes()
            packet = pickle.loads(wire)
            results.put((connection, packet, time.monotonic(), len(wire)))
            del wire, packet  # Do not retain a full fallback pickle while idle.
    except BaseException as error:
        results.put((connection, ('disconnected', None, repr(error)), time.monotonic(), 0))


def _pin(batch):
    """Pin only in the caller; leave ordinary CPU scalars/objects unchanged."""
    if hasattr(batch, 'pin_memory'):
        return batch.pin_memory()
    if isinstance(batch, dict):
        return {key: _pin(value) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_pin(value) for value in batch]
    if isinstance(batch, tuple):
        values = [_pin(value) for value in batch]
        return type(batch)(*values) if hasattr(batch, '_fields') else tuple(values)
    return batch


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be an integer' + (f' >= {minimum}' if minimum is not None else ''))
    return value


class EpochBatchIterator:
    """A lazy spawn pool with at most ``prefetch`` unconsumed batch sequences.

    Each worker executes at most one batch at a time. Results may finish out of
    order but are consumed in sequence order. ``timeout_seconds`` includes child
    startup and generation/transfer for each dispatched batch. A failed/closed
    iterator owns no live worker children. ``pause`` drains/discards speculative
    batches; reset and restore retain healthy idle workers. ``close`` shuts down
    owned resources; an explicit later reset may reopen lazily for compatibility.
    """

    def __init__(self, loader, *, workers=0, seed=0, epoch=0, rank=0, stage='train',
                 prefetch=2, timeout_seconds=180, pin_memory=False, worker_threads=1,
                 phase_steps=None, max_shared_bytes=64 * 1024 * 1024):
        self.loader = loader
        self.workers = _integer(workers, 'workers', 0)
        self.prefetch = _integer(prefetch, 'prefetch', 1)
        self.worker_threads = _integer(worker_threads, 'worker_threads', 1)
        self.phase_steps = None if phase_steps is None else _integer(phase_steps, 'phase_steps', 0)
        self.max_shared_bytes = _integer(max_shared_bytes, 'max_shared_bytes', 0)
        self.seed = _integer(seed, 'seed')
        self.epoch = _integer(epoch, 'epoch', 0)
        self.rank = _integer(rank, 'rank', 0)
        if not isinstance(stage, str) or not stage:
            raise ValueError('stage must be a nonempty string')
        self.stage = stage
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be finite and positive')
        self.timeout_seconds = float(timeout_seconds)
        self.pin_memory = bool(pin_memory)
        self._next_index = self._next_submit = 0
        self._closed = False
        self._processes = []
        self._connections = {}
        self._threads = []
        self._pending = {}
        self._ready = {}
        self._results = queue.Queue()
        self._terminal_sequence = None
        self._paused = False
        self._buffers = {}
        self._sequence_slots = {}
        self._slot_uses = {}
        self._worker_observations = {}
        self._stats = dict(shared_batches=0, pipe_batches=0, shared_reuses=0,
                           tensor_bytes_shared=0, tensor_bytes_pipe=0, pipe_packet_bytes=0,
                           consumed_batches=0, discarded_batches=0, queue_depth_peak=0, peak_payload_bytes=0)
        self._fallback_reasons = set()
        hook = getattr(loader, 'configure_worker_threads', None)
        if callable(hook): hook(self.worker_threads)

    def __iter__(self):
        return self

    def _start(self):
        if self._processes:
            return
        from task_worker import child_options, register_child
        ctx = mp.get_context('spawn')
        descriptors = child_options().get('pass_fds', ())
        # Windows starts a process before serializing Process arguments. Validate
        # and freeze loader input first, avoiding an untracked bootstrap child
        # on serialization failure. Loaders should contain metadata, not live
        # multiprocessing primitives, memmaps or opened case caches.
        loader_bytes = pickle.dumps(self.loader, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            for _ in range(self.workers):
                parent, child = ctx.Pipe()
                lock_handle = None
                try:
                    # Each child reduces this borrowed FD inside its own
                    # Process.start(); no unconsumed parent transfer is created.
                    lock_handle = _SpawnLockFd(descriptors[0]) if descriptors else None
                    process = ctx.Process(target=_worker_main, args=(loader_bytes, child, lock_handle, self.worker_threads),
                                          name=f'batch-{self.stage}')
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._processes.append(process)
                self._connections[parent] = process
                if os.environ.get('WHOLE_HEART_MANAGED') or os.environ.get('WHOLE_HEART_JOB'):
                    register_child(process, f'batch:{self.stage}')
                reader = threading.Thread(target=_read_results, args=(parent, self._results), daemon=True,
                                          name=f'batch-result-{process.pid}')
                self._threads.append(reader)
                reader.start()
        except BaseException:
            self.close()
            raise

    def _fill(self):
        if self._paused or self._terminal_sequence is not None:
            return
        for connection, process in self._connections.items():
            if self._next_submit - self._next_index >= self.prefetch:
                break
            if self.phase_steps is not None and self._next_submit >= self.phase_steps:
                break
            if connection in self._pending:
                continue
            sequence = self._next_submit
            slot = next(index for index in range(self.prefetch) if index not in self._sequence_slots.values())
            self._sequence_slots[sequence] = slot
            self._stats['queue_depth_peak'] = max(self._stats['queue_depth_peak'],len(self._sequence_slots))
            self._pending[connection] = (sequence, time.monotonic())
            try:
                buffer = self._buffers.get(slot)
                connection.send((sequence, deterministic_seed(self.seed, self.epoch, self.rank, self.stage, sequence),
                                 (buffer.name, buffer.size) if buffer is not None else None))
            except (OSError, EOFError) as error:
                raise RuntimeError(self._description(process, sequence) + f' dispatch failed: {error}') from error
            self._next_submit += 1

    def _description(self, process, sequence):
        return f'batch worker pid={process.pid} stage={self.stage} sequence={sequence}'

    def _receive(self, *, block=True):
        while True:
            try:
                connection, packet, completed, packet_bytes = self._results.get_nowait()
            except queue.Empty:
                if not block:return False
                now = time.monotonic()
                for connection, (sequence, started) in self._pending.items():
                    if now - started >= self.timeout_seconds:
                        raise TimeoutError(self._description(self._connections[connection], sequence) +
                                           f' exceeded {self.timeout_seconds:g}s')
                try:
                    connection, packet, completed, packet_bytes = self._results.get(timeout=min(.05, self.timeout_seconds))
                except queue.Empty:
                    continue
            status, sequence, payload = packet
            if status == 'buffer_request':
                if connection not in self._pending or sequence != self._pending[connection][0]:
                    raise RuntimeError('invalid buffer request sequence')
                slot = self._sequence_slots[sequence]
                old = self._buffers.get(slot)
                used = sum(buffer.size for buffer in self._buffers.values())
                available = self.max_shared_bytes - used + (old.size if old is not None else 0)
                reason, info = 'shared_budget', None
                if old is not None:
                    # Fixed owned slot capacities avoid transient double allocation
                    # and mappings retained by idle children after a resize.
                    reason = 'shared_slot_capacity'
                elif payload <= available:
                    try:
                        buffer = allocate_shared(payload)
                        self._buffers[slot] = buffer
                        info, reason = (buffer.name, buffer.size), None
                    except OSError as error:
                        reason = f'shared_allocation:{error}'
                connection.send((info, reason))
                continue
            if status == 'disconnected':
                pending = self._pending.get(connection)
                if pending is None:
                    continue  # A terminal packet was already received for this worker.
                process = self._connections[connection]
                raise RuntimeError(self._description(process, pending[0]) +
                                   f' exited/disconnected (exitcode={process.exitcode}): {payload}')
            pending = self._pending.pop(connection, None)
            if pending is None or sequence != pending[0]:
                process = self._connections[connection]
                raise RuntimeError(self._description(process, sequence) + ' invalid response sequence')
            if completed - pending[1] > self.timeout_seconds:
                raise TimeoutError(self._description(self._connections[connection], sequence) +
                                   f' exceeded {self.timeout_seconds:g}s')
            if status == 'ok':
                mode = payload['mode']
                self._stats['peak_payload_bytes'] = max(self._stats['peak_payload_bytes'], payload['tensor_bytes'])
                self._stats[mode + '_batches'] += 1
                self._stats['tensor_bytes_' + mode] += payload['tensor_bytes']
                self._stats['pipe_packet_bytes'] += packet_bytes
                if mode == 'shared':
                    slot = self._sequence_slots[sequence]
                    if self._slot_uses.get(slot,0): self._stats['shared_reuses'] += 1
                    self._slot_uses[slot] = self._slot_uses.get(slot,0) + 1
                    seen = self._worker_observations.get(connection, {}).get('shared_batches', 0)
                    payload['worker']['shared_batches'] = seen + 1
                elif payload.get('fallback_reason'):
                    self._fallback_reasons.add(payload['fallback_reason'])
                self._worker_observations[connection] = payload['worker']
            self._ready[sequence] = (connection, status, payload)
            if status != 'ok':
                self._terminal_sequence = sequence if self._terminal_sequence is None else min(self._terminal_sequence, sequence)
            return True

    def ready(self):
        """Poll only already-delivered worker packets; never generate or wait.

        The caller retains all source/RNG ownership. Serial sources cannot
        promise nonblocking generation, so never speculatively advance them.
        """
        if self._closed or self._paused or not self.workers:return False
        if self.phase_steps is not None and self._next_index>=self.phase_steps:return False
        while self._next_index not in self._ready:
            if not self._receive(block=False):return False
            self._fill()
        return True

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self.phase_steps is not None and self._next_index >= self.phase_steps:
            raise StopIteration
        try:
            self._paused = False
            if not self.workers:
                started = time.monotonic()
                batch = _generate(self.loader, deterministic_seed(self.seed, self.epoch, self.rank, self.stage,
                                                                  self._next_index), preserve=True)
                if time.monotonic() - started > self.timeout_seconds:
                    raise TimeoutError(f'serial batch stage={self.stage} sequence={self._next_index} exceeded '
                                       f'{self.timeout_seconds:g}s (serial calls cannot be forcibly interrupted)')
            else:
                self._start()
                self._fill()
                while self._next_index not in self._ready:
                    self._receive()
                    self._fill()
                connection, status, batch = self._ready.pop(self._next_index)
                if status == 'stop':
                    raise StopIteration
                if status == 'error':
                    raise RuntimeError(self._description(self._connections[connection], self._next_index) + '\n' + batch)
                batch = (_shared_decode(batch['batch'], self._buffers[self._sequence_slots[self._next_index]].buf)
                         if batch['mode'] == 'shared' else batch['batch'])
                del self._sequence_slots[self._next_index]
            if not self.workers:
                try:
                    _,_,size=_shared_encode(batch)
                    self._stats['peak_payload_bytes']=max(self._stats['peak_payload_bytes'],size)
                except ValueError:pass
            if self.pin_memory:
                batch = _pin(batch)
            self._next_index += 1
            self._stats['consumed_batches'] += 1
            if self.workers:
                self._fill()
            return batch
        except BaseException:
            self.close()
            raise

    def state_dict(self):
        return dict(version=1, seed=self.seed, epoch=self.epoch, rank=self.rank,
                    stage=self.stage, next_index=self._next_index)

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError('unsupported batch iterator state version')
        try:
            seed = _integer(state['seed'], 'seed')
            epoch = _integer(state['epoch'], 'epoch', 0)
            rank = _integer(state['rank'], 'rank', 0)
            index = _integer(state['next_index'], 'next_index', 0)
            stage = state['stage']
        except KeyError as error:
            raise ValueError(f'incomplete batch iterator state: {error}') from error
        if not isinstance(stage, str) or not stage:
            raise ValueError('stage must be a nonempty string')
        self.pause()
        self.seed, self.rank, self.stage = seed, rank, stage
        self.set_epoch(epoch, start_index=index)

    def set_epoch(self, epoch, start_index=0):
        epoch = _integer(epoch, 'epoch', 0)
        index = _integer(start_index, 'start_index', 0)
        self.pause()
        self.epoch = epoch
        self._next_index = self._next_submit = index
        self._closed = False
        self._terminal_sequence = None

    def pause(self):
        """Reach idle within batch timeout, discard prefetch, retain consumed cursor.

        The next ``next()`` resumes automatically. No reader or worker is replaced.
        On failure every owned worker/buffer is closed before propagating error.
        """
        self._paused = True
        try:
            while self._pending:
                self._receive()
            # Surface speculative worker errors before reporting idle success.
            for connection, status, payload in self._ready.values():
                if status == 'error':
                    raise RuntimeError(self._description(self._connections[connection], 'prefetch') + '\n' + payload)
            dead = [p for p in self._processes if not p.is_alive()]
            if dead:
                raise RuntimeError('batch workers exited while pausing: ' + ', '.join(
                    f'pid={p.pid} exitcode={p.exitcode}' for p in dead))
            self._stats['discarded_batches'] += sum(status=='ok' for _,status,_ in self._ready.values())
            self._ready.clear()
            self._sequence_slots.clear()
            self._next_submit = self._next_index
            self._terminal_sequence = None
        except BaseException:
            self.close()
            raise

    def rewind(self, next_index):
        """Replay batches prepared by an outer transfer queue but not consumed.

        Caller must discard its outer queue before rewinding. State format v1
        continues to represent consumed position; outer consumers may override
        next_index in their own checkpoint snapshot before calling restore.
        """
        index = _integer(next_index, 'next_index', 0)
        if index > self._next_index:
            raise ValueError('cannot rewind beyond consumed cursor')
        self.pause()
        self._next_index = self._next_submit = index

    def set_phase_steps(self, steps):
        """Drain before changing the absolute phase stop (None means unbounded).

        Reset epoch/cursor first when shortening below the old consumed cursor.
        Healthy workers and their loader objects remain resident.
        """
        steps = None if steps is None else _integer(steps, 'phase_steps', 0)
        if steps is not None and steps < self._next_index:
            raise ValueError('phase_steps cannot be below consumed cursor; reset epoch first')
        self.pause()
        self.phase_steps = steps

    def reconfigure(self, *, workers, worker_threads, prefetch, max_shared_bytes):
        """Rebuild only a replayable source after outer consumer has rewound."""
        if not getattr(self.loader, 'replayable', False):
            raise ValueError('source does not support batch-boundary reconfiguration')
        self.pause()
        state = self.state_dict()
        self.close()
        self.workers = _integer(workers, 'workers', 0)
        self.worker_threads = _integer(worker_threads, 'worker_threads', 1)
        self.prefetch = _integer(prefetch, 'prefetch', 1)
        self.max_shared_bytes = _integer(max_shared_bytes, 'max_shared_bytes', 0)
        hook = getattr(self.loader, 'configure_worker_threads', None)
        if callable(hook): hook(self.worker_threads)
        self._next_index = self._next_submit = state['next_index']
        self._closed = False
        self._paused = True

    def observations(self):
        ready_pipe = sum(payload['tensor_bytes'] for _,status,payload in self._ready.values()
                         if status=='ok' and payload['mode']=='pipe')
        return dict(self._stats, worker_threads=self.worker_threads, paused=self._paused,
                    closed=self._closed, consumed_index=self._next_index,
                    submitted_index=self._next_submit, phase_steps=self.phase_steps,
                    pending_batches=len(self._pending), ready_batches=len(self._ready),
                    prefetch=self.prefetch, transport='shared_memory_with_byte_pipe_fallback',
                    max_shared_bytes=self.max_shared_bytes,
                    shared_slots=len(self._buffers),reserved_slots=len(self._sequence_slots),
                    shared_allocated_bytes=sum(b.size for b in self._buffers.values()),
                    ready_pipe_tensor_bytes=ready_pipe,
                    queue_retained_storage_bytes=sum(b.size for b in self._buffers.values())+ready_pipe,
                    fallback_reasons=sorted(self._fallback_reasons),
                    worker_pids=[p.pid for p in self._processes],
                    workers=list(self._worker_observations.values()))

    def close(self):
        self._closed = True
        self._paused = True
        # Do not unlock any flock: each child holds a duplicate of the workflow
        # open-file description until that child actually exits.
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 2
        for process in self._processes:
            process.join(max(0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + 2
        for process in self._processes:
            process.join(max(0, deadline - time.monotonic()))
        lingering = [process for process in self._processes if process.is_alive()]
        if lingering:
            # Retain ownership/handles and allow a later close() retry. Never
            # report cleanup success or release a workflow lock explicitly.
            raise RuntimeError('batch workers still alive after cleanup: ' +
                               ', '.join(str(process.pid) for process in lingering))
        for connection in self._connections:
            connection.close()
        deadline = time.monotonic() + .5
        for reader in self._threads:
            reader.join(max(0, deadline - time.monotonic()))
        self._processes.clear()
        self._connections.clear()
        self._threads.clear()
        self._pending.clear()
        self._ready.clear()
        self._results = queue.Queue()  # Release any unconsumed prefetched batches.
        for buffer in self._buffers.values():
            buffer.close()
            buffer.unlink()
        self._buffers.clear()
        self._sequence_slots.clear()
        self._slot_uses.clear()
        self._worker_observations.clear()

    _finish = close

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
