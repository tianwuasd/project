"""Bounded, deterministic case sharding for frozen Xu predictors."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
import re
import threading

from .contracts import atomic_json


@dataclass(frozen=True)
class CasePlan:
    names: tuple
    devices: tuple
    requested: tuple
    host_budget: int | None
    estimated_lane_bytes: int | None

    def record(self):
        return dict(method='stable_round_robin_v1', requested=list(self.requested), devices=list(self.devices),
                    cases_per_device={dev:len(self.names[i::len(self.devices)]) for i,dev in enumerate(self.devices)},
                    idle_devices=list(self.requested[len(self.devices):]), cases=len(self.names),
                    host_budget_bytes=self.host_budget, estimated_lane_bytes=self.estimated_lane_bytes,
                    in_flight_cases_per_device=1)


def inference_lane_bytes(geometries, evaluation=False):
    """Conservative host working set for one CT, both model-load buffers and optional native metrics."""
    largest = 0
    for geo in geometries:
        working = 12*math.prod(geo.cache_shape)
        if evaluation:
            working += 64*math.prod(geo.native_shape_xyz)
        largest = max(largest,working)
    return 1536 * 1024**2 + largest


def case_plan(primary, names, cache, *, evaluation=False):
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected nonempty unique inference cases')
    raw = os.environ.get('XU_BRANCH_DEVICES')
    requested = json.loads(raw) if raw is not None else [str(primary)]
    if (not isinstance(requested,list) or not requested or not all(isinstance(s,str) for s in requested)
            or len(set(requested)) != len(requested) or requested[0] != str(primary)):
        raise ValueError('Inference devices must begin with the admitted primary device and be unique')
    if len(requested)==1:
        return CasePlan(names,tuple(requested),tuple(requested),None,None)
    if not all(re.fullmatch(r'cuda:[0-9]+',d) for d in requested):
        raise ValueError('Parallel frozen inference requires explicit allocated CUDA indices')
    from .xu_resources import applied_host_budget, require_applied_group
    require_applied_group(requested)
    budget = applied_host_budget()
    if not budget or budget <= 0:
        raise ValueError('Parallel inference lacks a reserved host-memory budget')
    lane_bytes = inference_lane_bytes((cache.geometry(n) for n in names), evaluation)
    lanes = min(len(requested),len(names),budget//lane_bytes)
    if not lanes:
        raise ValueError(f'Insufficient host-memory budget for one frozen inference lane: {budget} < {lane_bytes}')
    return CasePlan(names,tuple(requested[:lanes]),tuple(requested),budget,lane_bytes)


def _device_scope(device):
    if str(device).startswith('cuda'):
        import torch
        return torch.cuda.device(device)
    return nullcontext()


def map_cases(plan, make_worker):
    """One resident predictor per GPU. Workers return small records; image arrays stay local."""
    loading = threading.Lock()
    stop = threading.Event()

    def run_lane(lane):
        device = plan.devices[lane]
        rows = {}
        try:
            with _device_scope(device):
                # Do not materialize all CPU checkpoint state dictionaries simultaneously.
                with loading:
                    if stop.is_set():
                        return rows
                    process = make_worker(device)
                for name in plan.names[lane::len(plan.devices)]:
                    if stop.is_set():
                        break
                    rows[name] = process(name)
        except BaseException:
            stop.set()
            raise
        return rows

    if len(plan.devices)==1:
        return run_lane(0)
    with ThreadPoolExecutor(max_workers=len(plan.devices), thread_name_prefix='xu-infer') as pool:
        shards = list(pool.map(run_lane,range(len(plan.devices))))
    return {name:shards[i%len(shards)][name] for i,name in enumerate(plan.names)}


def record_plan(out, plan):
    atomic_json(out/'inference-plan.json', plan.record())


def shard_identity(plan):
    """The case sequence and ordered physical devices determine every inference assignment."""
    import torch
    cards = []
    for device in plan.devices:
        props = torch.cuda.get_device_properties(device)
        physical = getattr(props,'uuid',None)
        if not physical:
            raise ValueError('Parallel inference requires verifiable GPU UUIDs')
        cards.append(dict(uuid=str(physical),name=props.name,capability=[props.major,props.minor]))
    if len({card['uuid'] for card in cards}) != len(cards):
        raise ValueError('Duplicate physical GPU in frozen inference')
    return dict(method='stable_round_robin_v1', cards=cards)
