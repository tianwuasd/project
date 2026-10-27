"""Two reusable CUDA input slots; checkpoint position advances only on commit."""
from concurrent.futures import ThreadPoolExecutor
import time


def _map(value, function):
    if isinstance(value, dict): return {k:_map(v, function) for k,v in value.items()}
    if isinstance(value, list): return [_map(v, function) for v in value]
    if isinstance(value, tuple): return tuple(_map(v, function) for v in value)
    return function(value)


def _copy_tree(value, previous, device, torch):
    if isinstance(value, torch.Tensor):
        if not (isinstance(previous, torch.Tensor) and previous.shape == value.shape
                and previous.dtype == value.dtype and previous.device == device):
            previous = torch.empty_like(value, device=device)
        previous.copy_(value, non_blocking=True)
        return previous
    if isinstance(value, dict):
        return {k:_copy_tree(v, previous.get(k) if isinstance(previous,dict) else None, device, torch)
                for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        result = [_copy_tree(v, previous[i] if isinstance(previous,(list,tuple)) and i<len(previous)
                           else None, device, torch) for i,v in enumerate(value)]
        return tuple(result) if isinstance(value,tuple) else result
    return value


class BatchTransfer:
    """Generation stays on caller thread, especially workers=0 RNG-scoped sources.

    Only pin/copy of already-owned batches runs in a background thread. No more
    than two slots exist, including the outstanding model batch. Caller MUST
    mark_consumed after a successful step. pause/state discard lookahead and
    rewind CPU source before capturing RNG or changing coordinates.
    """
    def __init__(self, source, device):
        import torch
        self.source, self.device, self.torch = source, torch.device(device), torch
        self.cuda = self.device.type == 'cuda'
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vista-pin') if self.cuda else None
        self.slots = [dict(future=None, data=None, host=None, ready=None, reuse=None) for _ in range(2)]
        self.queue, self.active = [], None
        self.consumed = source.state_dict()['next_index']
        self.exhausted = False
        self.closed = False
        self.stats = dict(prepared_batches=0, committed_batches=0, queue_peak=0,
            cpu_supply_seconds=0., pin_seconds=0., copy_seconds=0., copy_batches=0,
            generation_thread='caller; serial RNG cannot overlap model RNG',
            transfer_mode='cuda_two_slot_stream' if self.cuda else 'cpu_synchronous', slots=2 if self.cuda else 1)
        self.stats.update(pinned_bytes_peak=0, device_input_bytes_peak=0)

    def __iter__(self): return self

    def _prepare(self, index, batch):
        slot, torch = self.slots[index], self.torch
        with torch.cuda.device(self.device):
            # Before overwriting any storage, including on shape changes, wait
            # for the last consumer of this exact slot to finish.
            if slot['reuse'] is not None: slot['reuse'].synchronize()
            started = time.perf_counter()
            slot['host'] = _map(batch, lambda v: v.pin_memory() if isinstance(v,torch.Tensor)
                                and v.device.type == 'cpu' and not v.is_pinned() else v)
            self.stats['pin_seconds'] += time.perf_counter() - started
            start = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.stream):
                start.record(self.stream)
                slot['data'] = _copy_tree(slot['host'], slot['data'], self.device, torch)
                ready.record(self.stream)
            slot['start'], slot['ready'] = start, ready
            self.stats['pinned_bytes_peak'] = max(self.stats['pinned_bytes_peak'],self._slot_bytes('host'))
            self.stats['device_input_bytes_peak'] = max(self.stats['device_input_bytes_peak'],self._slot_bytes('data'))
        return slot['data']

    def _slot_bytes(self, key):
        seen = set()
        total = 0
        def count(value):
            nonlocal total
            if isinstance(value,self.torch.Tensor):
                storage = value.untyped_storage()
                identity = (str(value.device), storage.data_ptr())
                if identity not in seen:
                    seen.add(identity); total += storage.nbytes()
            return value
        for slot in self.slots: _map(slot.get(key),count)
        return total

    def _fill(self):
        bound = 2 if self.cuda else 1
        for index in range(bound):
            if self.exhausted or index in self.queue or index == self.active: continue
            started = time.perf_counter()
            try: batch = next(self.source)
            except StopIteration:
                self.exhausted = True; break
            finally: self.stats['cpu_supply_seconds'] += time.perf_counter() - started
            slot = self.slots[index]
            if self.cuda: slot['future'] = self.executor.submit(self._prepare,index,batch)
            else: slot['data'] = batch
            self.queue.append(index)
            self.stats['prepared_batches'] += 1
            self.stats['queue_peak'] = max(self.stats['queue_peak'], len(self.queue)+(self.active is not None))

    def __next__(self):
        if self.closed: raise RuntimeError('transfer closed')
        if self.active is not None: raise RuntimeError('mark_consumed or pause before requesting another batch')
        self._fill()
        if not self.queue: raise StopIteration
        self.active = self.queue.pop(0)
        slot = self.slots[self.active]
        if self.cuda:
            slot['future'].result()
            self.torch.cuda.current_stream(self.device).wait_event(slot['ready'])
        return slot['data']

    def mark_consumed(self):
        if self.active is None: raise RuntimeError('no outstanding batch')
        slot = self.slots[self.active]
        if self.cuda:
            reuse = self.torch.cuda.Event()
            reuse.record(self.torch.cuda.current_stream(self.device))
            slot['reuse'] = reuse
            # Upstream loss.detach().cpu() synchronizes the real step. Event
            # completion is still explicitly awaited for timing and storage.
            slot['ready'].synchronize()
            self.stats['copy_seconds'] += slot['start'].elapsed_time(slot['ready']) / 1000
            self.stats['copy_batches'] += 1
        self.consumed += 1
        self.stats['committed_batches'] += 1
        self.active = None

    def pause(self):
        if self.closed: return
        self.quiesce()
        self.queue.clear(); self.active = None; self.exhausted = False
        self.source.rewind(self.consumed)
        self.source.pause()
        # Inactive train/validation pools must not retain four device slots.
        # Reuse is within an active phase; crossing a phase releases inputs.
        for slot in self.slots:
            slot.update(future=None, data=None, host=None, ready=None, reuse=None)

    def quiesce(self):
        """Finish preparation but retain owned slots, e.g. before compile warmup."""
        # Settle every submitted task even when one raises, then propagate.
        error = None
        for slot in self.slots:
            if slot['future'] is not None:
                try: slot['future'].result()
                except BaseException as failure: error = error or failure
        if self.cuda:
            self.stream.synchronize()
            # An uncommitted step may still have queued device work.
            self.torch.cuda.current_stream(self.device).synchronize()
        self.source.pause()
        if error is not None: raise error

    def state_dict(self):
        self.pause()
        return dict(self.source.state_dict(), next_index=self.consumed)

    def load_state_dict(self, state):
        self.pause(); self.source.load_state_dict(state); self.consumed = state['next_index']

    def set_epoch(self, epoch, start_index=0):
        self.pause(); self.source.set_epoch(epoch,start_index); self.consumed = start_index

    def set_phase_steps(self, steps):
        self.pause(); self.source.set_phase_steps(steps)

    def rewind(self, next_index):
        if next_index > self.consumed: raise ValueError('cannot advance consumed cursor')
        self.pause(); self.source.rewind(next_index); self.consumed = next_index

    def observations(self):
        return dict(self.stats,consumed_index=self.consumed,prepared_unconsumed=len(self.queue)+(self.active is not None),
                    pinned_bytes=self._slot_bytes('host'), device_input_bytes=self._slot_bytes('data') if self.cuda else 0,
                    source=self.source.observations())

    def close(self):
        if self.closed: return
        try: self.pause()
        finally:
            if self.executor: self.executor.shutdown(wait=True, cancel_futures=True)
            self.source.close(); self.closed = True
            self.slots.clear()

    _finish = close
