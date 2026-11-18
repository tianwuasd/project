from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sys
from time import perf_counter

import numpy as np
import torch

from ..acceleration import OrderedSupply, attach, transfer_batch
from ..acceleration.stunet_supply import selection
from ..data.dataset import CaseDataset, sample_patch
from ..observability.events import event


def initialize(prepared, config, resources, handle, estimates):
    global _dataset, _config, _client, _estimates
    torch.set_num_threads(1)
    _dataset = CaseDataset(prepared, resources)
    _config, _client = config, attach(handle)
    _estimates = estimates


def build_patch(dataset, config, request):
    name, rng, context = selection(request)
    started, hits = perf_counter(), dataset.hits
    entry = dataset.get(name)
    phase = request[2]
    pair = sample_patch(entry, config['training']['patch'], rng,
                        config['training']['foreground_fraction'] if phase == 0 else 0.)
    return pair, entry['unknown_classes'], name, perf_counter()-started, dataset.hits > hits, context


def worker_patch(request):
    return case_patch(_dataset, _config, _client, _estimates, request)


def case_patch(dataset, config, client, estimates, request):
    name = selection(request)[0]
    began = perf_counter()
    with client.case(estimates[name], purpose='sample_patch:'+name):
        if config.get('scope')=='technical' and config.get('logging',{}).get('case_details'):
            with client.handle.condition:
                facts = dict(active_readers=client.handle.state['active_readers'],
                    workspace_reserved=client.handle.state['workspace_reserved'])
            event(config['run_dir'], 'sample_patch_admitted', case_id=name,
                workspace_bytes=estimates[name],wait_seconds=perf_counter()-began,**facts)
        return build_patch(dataset, config, request)


class BatchSupply:
    def __init__(self, prepared, config, slots, workers, cache_bytes, gate, owner, rank=0,
                 cancel=None, prefetch=1, handle=None):
        self.prepared, self.config, self.slots = prepared, config, slots
        self.workers, self.cache_bytes, self.rank = workers, cache_bytes, rank
        self.handle, self.prefetch, self.cancel = handle, prefetch, cancel
        self.estimates = dict(gate.estimates)
        self.inflight, self.transfer_handle = None, None

    @property
    def queue(self):
        return self.supply.queue

    def __enter__(self):
        self.client = attach(self.handle)
        resources = dict(self.config['resources'], cache_bytes=self.cache_bytes)
        self.dataset = CaseDataset(self.prepared, resources)
        self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=(self.prepared, self.config, resources, self.handle, self.estimates)) if self.workers else None
        self.supply = OrderedSupply(self.client, capacity=self.prefetch, build=self.build, collate=self.collate)
        self.supply.__enter__()
        return self

    def build(self, request):
        if self.pool:
            return self.pool.submit(worker_patch, request)
        return case_patch(self.dataset, self.config, self.client, self.estimates, request)

    def collate(self, values):
        results = [value.result() for value in values] if self.pool else values
        tensors = [torch.from_numpy(np.stack(items)) for items in zip(*(r[0] for r in results))]
        return dict(image=tensors[0], label=tensors[1], unknown_classes=[r[1] for r in results],
            case_ids=[r[2] for r in results], supply_seconds=sum(r[3] for r in results),
            cache_hits=sum(r[4] for r in results), cache_misses=sum(not r[4] for r in results),
            batch_index=results[0][5]['batch_index'])

    def submit(self, names, phase, attempt):
        requests = [(names, self.config['training'], phase, attempt, slot) for slot in self.slots]
        self.supply.submit(selection(requests[0])[2]['batch_index'], requests)

    def take_cpu(self):
        index = self.queue[0][0]
        batch = self.supply.take(index)
        self.inflight = index
        return batch

    def transfer(self, batch, device):
        self.transfer_handle = transfer_batch([batch['image'], batch['label']], device, client=self.client)
        batch['image'], batch['label'] = self.transfer_handle.tensors
        return batch

    def mark_consumed(self, discarded=False):
        if self.transfer_handle is not None:
            self.transfer_handle.release()
            self.transfer_handle = None
        self.supply.consumed(self.inflight)
        self.inflight = None

    def __exit__(self, *failure):
        if failure and failure[0] is not None:
            self.handle.cancel.set()
        if self.transfer_handle is not None:
            self.transfer_handle.release()
        self.supply.__exit__(*failure)
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.dataset.clear()
        self.client.close()
