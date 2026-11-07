"""Short technical probe of simultaneous frozen Xu inference on the allocated CUDA pool."""
import copy
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from .contracts import atomic_json
from .xu_inference_pool import case_plan, map_cases, shard_identity
from .xu_resources import apply_env_limit


def _inputs():
    import torch
    from .xu_models import build_model
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2023)
        template = build_model('crop64','smoke').eval()
    image = torch.linspace(-1,1,64**3,dtype=torch.float32).reshape(1,1,64,64,64)
    return template, image


def _load(device, template, image, barrier):
    import torch
    model = copy.deepcopy(template).to(device)
    values = image.to(device)
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    identifier = str(props.uuid)
    def predict(name):
        # Record host boundaries around the actual forward and its completed CUDA events.
        ready = time.monotonic_ns()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        barrier.wait()
        started = time.monotonic_ns()
        with torch.inference_mode():
            begin.record()
            scores = model(values)
            end.record()
            end.synchronize()
            finished = time.monotonic_ns()
            if scores.shape != (1,11,64,64,64) or not torch.isfinite(scores).all().item():
                raise ValueError('Frozen CUDA model produced an unexpected or nonfinite output')
            labels = scores.argmax(1)
            checksum = int(labels.sum().item())
        return dict(case=name,device=device,uuid=identifier,ready_ns=ready,forward_start_ns=started,
                    cuda_completed_ns=finished,cuda_elapsed_ms=begin.elapsed_time(end),
                    label_sum=checksum,finite=True,shape=list(scores.shape))
    return predict


def _overlap(rows, devices, identity):
    """CUDA event durations are per GPU. The common monotonic clock compares host execution intervals."""
    if len(rows)!=len(devices) or len(identity['cards'])!=len(devices):
        raise ValueError('Missing inference evidence for an assigned GPU')
    for row,d,card in zip(rows,devices,identity['cards']):
        if (row['device']!=d or row['uuid']!=card['uuid'] or row.get('finite') is not True
                or not math.isfinite(row['cuda_elapsed_ms']) or row['cuda_elapsed_ms']<=0
                or not row['ready_ns']<=row['forward_start_ns']<row['cuda_completed_ns']):
            raise ValueError('Missing measured, finite CUDA forward for an assigned GPU')
    ns = min(r['cuda_completed_ns'] for r in rows)-max(r['forward_start_ns'] for r in rows)
    if ns<=0:
        raise ValueError('Measured host intervals around the completed CUDA forwards do not overlap')
    return ns/1e9


def run_worker(out,primary,requested):
    directory = Path(out)
    path = directory/'inference.json'
    devices = list(requested)
    result = dict(protocol='xu-inference-pool-probe-1', devices=devices, profile='smoke',input='synthetic',
                  branch='crop64', state='checking', cases=len(devices))
    if path.exists():
        raise ValueError('Inference probe requires a fresh report path')
    atomic_json(path,result)
    barrier = None
    try:
        if (not devices or devices[0]!=primary or len(set(devices))!=len(devices)
                or not all(str(d).startswith('cuda:') for d in devices)):
            raise ValueError('Expected ordered, distinct CUDA GPU indices starting with the assigned primary')
        selection = json.loads(os.environ.get('XU_BRANCH_DEVICES','null'))
        if selection is None:
            selection = [primary]
        if selection!=devices or not os.environ.get('XU_CUDA_ADMISSION'):
            raise ValueError('Probe devices differ from the authorized resource tickets')
        result['admission'] = apply_env_limit(primary)
        names = tuple(f'synthetic-{index:04d}' for index in range(len(devices)))
        geo = SimpleNamespace(cache_shape=(64,)*3,native_shape_xyz=(64,)*3)
        plan = case_plan(primary,names,SimpleNamespace(geometry=lambda name:geo))
        if list(plan.devices)!=devices:
            raise ValueError(f'Selected cards exceed fresh host/case capacity: {plan.record()}')
        result.update(state='running', plan=plan.record(),identity=shard_identity(plan))
        atomic_json(path,result)
        barrier = threading.Barrier(len(devices),timeout=60)
        template,image = _inputs()
        def make_worker(dev):
            try:
                return _load(dev,template,image,barrier)
            except BaseException:
                barrier.abort()
                raise
        rows = list(map_cases(plan,make_worker).values())
        result['forwards'] = rows
        result['host_forward_interval_overlap_seconds'] = _overlap(rows,devices,result['identity'])
        result['state'] = 'passed'
    except (Exception, KeyboardInterrupt) as exc:
        if barrier is not None:
            barrier.abort()
        result.update(state='failed', reason=str(exc) or type(exc).__name__, error_type=type(exc).__name__)
    finally:
        atomic_json(path,result)
    return result
