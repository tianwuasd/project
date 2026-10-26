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


def _worker_main(loader_bytes, connection, lock_handle):
    """Spawn target; owned duplicate closes normally, without explicit flock unlock."""
    lock_fd = None
    sequence = None
    loader = None
    try:
        if lock_handle is not None:
            lock_fd = lock_handle.detach()
            os.environ['WHOLE_HEART_LOCK_FD'] = str(lock_fd)
        while True:
            task = connection.recv()
            if task is None:
                return
            sequence, seed = task
            try:
                status = 'ok'
                if loader is None:
                    # Open resources only after inheriting the workflow lock.
                    loader = pickle.loads(loader_bytes)
                batch = _generate(loader, seed)
                # Serialize by value before sending. This also catches serialization
                # errors here and avoids Torch shared-storage lifetime dependencies.
                packet = pickle.dumps(('ok', sequence, batch), protocol=pickle.HIGHEST_PROTOCOL)
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
        connection.close()
        if lock_fd is not None:
            os.close(lock_fd)


def _read_results(connection, results):
    """Keep potentially partial/large pipe reads off the bounded-wait consumer."""
    try:
        while True:
            packet = pickle.loads(connection.recv_bytes())
            results.put((connection, packet, time.monotonic()))
    except BaseException as error:
        results.put((connection, ('disconnected', None, repr(error)), time.monotonic()))


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
    iterator owns no live worker children. ``set_epoch`` and ``load_state_dict``
    discard old prefetch and reopen lazily; ``close`` is idempotent.
    """

    def __init__(self, loader, *, workers=0, seed=0, epoch=0, rank=0, stage='train',
                 prefetch=2, timeout_seconds=180, pin_memory=False):
        self.loader = loader
        self.workers = _integer(workers, 'workers', 0)
        self.prefetch = _integer(prefetch, 'prefetch', 1)
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
                    process = ctx.Process(target=_worker_main, args=(loader_bytes, child, lock_handle),
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
        if self._terminal_sequence is not None:
            return
        for connection, process in self._connections.items():
            if self._next_submit - self._next_index >= self.prefetch:
                break
            if connection in self._pending:
                continue
            sequence = self._next_submit
            self._pending[connection] = (sequence, time.monotonic())
            try:
                connection.send((sequence, deterministic_seed(self.seed, self.epoch, self.rank, self.stage, sequence)))
            except (OSError, EOFError) as error:
                raise RuntimeError(self._description(process, sequence) + f' dispatch failed: {error}') from error
            self._next_submit += 1

    def _description(self, process, sequence):
        return f'batch worker pid={process.pid} stage={self.stage} sequence={sequence}'

    def _receive(self):
        while True:
            try:
                connection, packet, completed = self._results.get_nowait()
            except queue.Empty:
                now = time.monotonic()
                for connection, (sequence, started) in self._pending.items():
                    if now - started >= self.timeout_seconds:
                        raise TimeoutError(self._description(self._connections[connection], sequence) +
                                           f' exceeded {self.timeout_seconds:g}s')
                try:
                    connection, packet, completed = self._results.get(timeout=min(.05, self.timeout_seconds))
                except queue.Empty:
                    continue
            status, sequence, payload = packet
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
            self._ready[sequence] = (connection, status, payload)
            if status != 'ok':
                self._terminal_sequence = sequence if self._terminal_sequence is None else min(self._terminal_sequence, sequence)
            return

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
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
            if self.pin_memory:
                batch = _pin(batch)
            self._next_index += 1
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
        self.close()
        self.seed, self.rank, self.stage = seed, rank, stage
        self.set_epoch(epoch, start_index=index)

    def set_epoch(self, epoch, start_index=0):
        epoch = _integer(epoch, 'epoch', 0)
        index = _integer(start_index, 'start_index', 0)
        self.close()
        self.epoch = epoch
        self._next_index = self._next_submit = index
        self._closed = False
        self._terminal_sequence = None
        self._results = queue.Queue()

    def close(self):
        self._closed = True
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

    _finish = close

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
