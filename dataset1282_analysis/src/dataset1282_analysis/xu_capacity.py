"""Preflight for the formal 3D and blood training stages. Run in a separate process."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from .contracts import atomic_json
from .observability.status import operation_lock
from .resources import GIB, snapshot
from .xu_resources import apply_env_limit, gpu_budget

STAGES = ('3d64', '3d128', 'blood-encoder', 'blood-sequence')


def specification(stage):
    """Count the actual paper model without materializing formal-size tensors."""
    import torch
    from .xu_blood_models import BloodSequence, SliceUNet
    from .xu_models import build_model

    with torch.device('meta'):
        if stage in ('3d64', '3d128'):
            model = build_model('crop' + stage[2:], 'paper')
            n = sum(p.numel() for p in model.parameters())
            spec = model.spec
            shape = [spec.batch, 1, spec.side, spec.side, spec.side]
            # Budget for large full-resolution skip, concat, forward/backward and Adam.
            feature_bytes = spec.batch * spec.width * spec.side**3 * 4
            working_bytes = 16*n + 7*feature_bytes
            frozen = 0
        elif stage in ('blood-encoder', 'blood-sequence'):
            encoder = SliceUNet('paper')
            sequence = BloodSequence(encoder.width)
            enc = sum(p.numel() for p in encoder.parameters())
            seq = sum(p.numel() for p in sequence.parameters())
            plane = 512*512 * encoder.width * 4
            if stage == 'blood-encoder':
                n, frozen, shape = enc, seq, [4,1,512,512]
                working_bytes = 16*enc + 4*seq + 7*4*plane
            else:
                n, frozen, shape = seq, enc, [1,7,512,512]
                # Frozen encoder weights, seven feature frames and recurrent activations.
                working_bytes = 16*seq + 4*enc + 96*plane
        else:
            raise ValueError(f'Unsupported paper stage {stage}')
    return dict(shape=shape, trainable_parameters=n, other_resident_parameters=frozen,
                feature_width=spec.width if stage.startswith('3d') else encoder.width,
                estimated_host_bytes=max(GIB, 8*(n+frozen)),
                estimated_working_bytes=working_bytes, dtype='float32', optimizer='Adam')


def device_demands(stage, devices, spec=None):
    """Estimate each process-local replica; one optimizer and the global objective live on primary."""
    from .xu_distributed import split_sizes
    names = list(map(str,devices))
    if not names or len(names) != len(set(names)):
        raise ValueError('Capacity requires an ordered list of distinct devices')
    spec = specification(stage) if spec is None else spec
    shape = spec['shape']
    temporal = stage == 'blood-sequence'
    batch = 7 if temporal else shape[0]
    chunks = split_sizes(batch, min(batch,len(names)))
    if len(names) == 1:
        return {names[0]:spec['estimated_working_bytes']}
    n, frozen = spec['trainable_parameters'], spec['other_resident_parameters']
    voxels = math.prod(shape[2:])
    unit = 4 * spec['feature_width'] * voxels
    demands = dict.fromkeys(names,0)
    for i,size in enumerate(chunks):
        if temporal:
            # Root retains all gathered frames for its single recurrent update.
            work = (spec['estimated_working_bytes'] + batch*unit if i == 0
                    else 4*frozen + 7*size*unit)
        else:
            classes = 11 if stage.startswith('3d') else 3
            # Gather all logits and their loss gradient on primary. Auxiliary cards
            # retain model replicas and local gradients but no resident Adam state.
            primary = 16*n + 4*frozen + 2*batch*classes*voxels*4
            work = (primary if i == 0 else 8*n) + 7*size*unit
        demands[names[i]] = work
    return demands


def admission(spec, host_free, cuda_free=None, *, total=None, policy='legacy',
              host_limit=None, cuda_limit=None):
    """Conservative admission only. A successful step is required to qualify the device."""
    work = spec['estimated_working_bytes']
    params = spec['trainable_parameters'] + spec['other_resident_parameters']
    # Host also constructs the modules and batches. Allow more workspace on CPU-only runs.
    host_need = 2*work + 2*GIB if cuda_free is None else max(GIB, 8*params)
    host_budget = max(0, min(int(.8*host_free), host_free - 4*GIB)) if host_free else 0
    if host_limit is not None:
        host_budget = min(host_budget, host_limit)
    data = dict(host_available_bytes=host_free, host_budget_bytes=host_budget,
                estimated_host_bytes=host_need, admission_policy='conservative workspace estimate')
    if cuda_free is not None:
        if total is None and policy == 'v2':
            raise ValueError('V2 GPU admission requires total memory')
        limit = gpu_budget(cuda_free, total, policy)
        if cuda_limit is not None:
            limit = min(limit, cuda_limit)
        data.update(cuda_free_bytes=cuda_free, cuda_allocator_budget_bytes=limit)
        if limit < work:
            return 'below_gpu_budget', data
    return ('below_host_budget' if host_need > host_budget else None), data


def _step_3d(stage, device, supervision, record=None):
    import numpy as np
    import torch
    from .xu_models import build_model
    from .xu_training import _epoch, learning_rate

    branch = 'crop' + stage[2:]
    model = build_model(branch, 'paper').to(device)
    spec = model.spec
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(0,spec.epochs))
    rng = np.random.default_rng(2023)
    shape = (spec.side,) * 3
    image = rng.standard_normal(shape, dtype=np.float32)
    label = rng.integers(0, 10, shape, dtype=np.int8)
    names = [f'probe-{i}' for i in range(spec.batch)]
    groups = {name: [10] if supervision == 'marginal' and i % 2 else []
              for i, name in enumerate(names)}
    before = model.head.weight.detach().clone()
    stats = {}
    loss = _epoch(model, names, groups, lambda name, branch: (image,label), branch, device,
                  torch.Generator(device=device).manual_seed(2023), optimizer, stats=stats)
    if record is not None:
        record['distribution'] = stats['distribution']
    return loss, (model.head.weight.detach() - before).abs().max().item()


def _step_blood(stage, device, record=None):
    import numpy as np
    import torch
    from .xu_blood import BloodForward, _batch, _objective
    from .xu_blood_models import BloodSequence, SliceUNet

    temporal = stage == 'blood-sequence'
    rng = np.random.default_rng(2023)
    planes = rng.standard_normal((7,512,512), dtype=np.float32)
    labels = rng.integers(0,3,(7,512,512),dtype=np.int8)

    class Planes:
        image = staticmethod(planes.__getitem__)
        label = staticmethod(labels.__getitem__)

        @staticmethod
        def window(z):
            return [min(6,max(0,z+i)) for i in range(-3,4)]

    encoder = SliceUNet('paper').to(device).train(not temporal)
    sequence = BloodSequence(encoder.width).to(device).train(temporal)
    model = sequence if temporal else encoder
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    images, target = _batch(Planes(), [3] if temporal else [1,2,3,4], temporal, device, rng)
    before = model.head.weight.detach().clone()
    forward = BloodForward(encoder,device,temporal,4)
    scores = forward(images,sequence)
    if record is not None:
        record['distribution'] = forward.distribution.stats
    loss = _objective(scores, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item(), (model.head.weight.detach() - before).abs().max().item()


def probe(stage, selected_device, supervision, output, policy='legacy'):
    import torch

    file = Path(output) / 'capacity.json'
    started = time.monotonic()
    report = dict(stage=stage, requested_device=selected_device, supervision=supervision,
                  profile='paper', input='synthetic', state='checking', optimizer_steps=0,
                  torch_version=str(torch.__version__))
    atomic_json(file, report)
    device = None
    measured = []
    try:
        device = torch.device(selected_device)
        if device.type not in ('cpu','cuda') or (device.type=='cpu' and device.index is not None):
            raise ValueError('Expected cpu or cuda[:index]')
        limits = apply_env_limit(device)
        policy = limits.get('policy', policy)
        report['resource_policy'] = policy
        available = snapshot()['available_memory_bytes']
        spec = specification(stage)
        report.update(spec)
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable in this Python')
            if device.index is None:
                device = torch.device('cuda',torch.cuda.current_device())
            torch.cuda.set_device(device)
            cards = limits.get('per_device',{str(device):limits})
            work = device_demands(stage,cards,spec)
            report['per_device'] = {}
            reason = None
            for name, limit in cards.items():
                free, total = torch.cuda.mem_get_info(name)
                scoped = dict(spec,estimated_working_bytes=work[name])
                denied, facts = admission(scoped,available,free,total=total,policy=policy,
                                          host_limit=limits.get('host_budget_bytes'),
                                          cuda_limit=limit.get('allocator_budget_bytes'))
                report['per_device'][name] = dict(facts, estimated_working_bytes=work[name],
                        cuda_total_bytes=total, gpu_name=torch.cuda.get_device_name(name),
                        uuid=str(getattr(torch.cuda.get_device_properties(name),'uuid','unavailable')))
                if denied:
                    reason = reason or denied
                    report['per_device'][name]['reason'] = denied
            report.update(report['per_device'][str(device)], runtime_cuda=torch.version.cuda)
        else:
            reason, facts = admission(spec, available,host_limit=limits.get('host_budget_bytes'))
            report.update(facts)
        report['device'] = str(device)
        if reason:
            report.update(state='not_admitted', reason=reason)
        else:
            for name,row in report.get('per_device',{}).items():
                torch.cuda.set_per_process_memory_fraction(row['cuda_allocator_budget_bytes']/row['cuda_total_bytes'],name)
                torch.cuda.reset_peak_memory_stats(name)
                measured.append(name)
            torch.set_num_threads(min(4,torch.get_num_threads()))
            torch.manual_seed(2023)
            report.update(state='running_training_step')
            atomic_json(file,report)
            if stage in ('3d64','3d128'):
                loss, delta = _step_3d(stage,device,supervision,record=report)
            else:
                loss, delta = _step_blood(stage,device,record=report)
            if not math.isfinite(loss) or not math.isfinite(delta) or delta <= 0:
                raise FloatingPointError('Expected a finite loss and an actual parameter update')
            if len(measured) > 1:
                from .xu_distributed import split_sizes
                count = 7 if stage == 'blood-sequence' else spec['shape'][0]
                sizes = split_sizes(count,min(count,len(measured)))
                expected = dict(zip(measured,sizes+(0,)*(len(measured)-len(sizes))))
                if report.get('distribution',{}).get('cases_by_device') != expected:
                    raise RuntimeError('Capacity forward did not use the promised stage allocation')
            report.update(state='passed', optimizer_steps=1, loss=loss, head_max_abs_update=delta)
    except (OSError, ValueError, RuntimeError, MemoryError, FloatingPointError) as error:
        report.update(state='failed', error_type=type(error).__name__, reason=str(error))
    finally:
        for name in measured:
            peaks = dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(name),
                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(name))
            report['per_device'][name].update(peaks)
            if name == str(device):
                report.update(peaks)
        report['seconds'] = time.monotonic()-started
        atomic_json(file,report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--policy', choices=('legacy','v2'), default='legacy')
    parser.add_argument('--supervision', choices=('marginal','complete'), default='marginal')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    with operation_lock(args.output,'xu-capacity',args.output):
        report = probe(args.stage,args.device,args.supervision,args.output,args.policy)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    if report['state'] == 'passed':
        return 0
    print(f"{args.stage}: {report['reason']} (report: {args.output.resolve()/'capacity.json'})",
          file=sys.stderr, flush=True)
    return 1
