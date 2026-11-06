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
                # After encoder fitting its last gradients can remain resident.
                # Allow room for seven feature frames and recurrent activations.
                working_bytes = 16*seq + 8*enc + 96*plane
        else:
            raise ValueError(f'Unsupported paper stage {stage}')
    return dict(shape=shape, trainable_parameters=n, other_resident_parameters=frozen,
                estimated_working_bytes=working_bytes, dtype='float32', optimizer='Adam')


def admission(spec, host_free, cuda_free=None):
    """Conservative admission only. A successful step is required to qualify the device."""
    work = spec['estimated_working_bytes']
    params = spec['trainable_parameters'] + spec['other_resident_parameters']
    # Host also constructs the modules and batches. Allow more workspace on CPU-only runs.
    host_need = 2*work + 2*GIB if cuda_free is None else max(GIB, 8*params)
    host_budget = max(0, min(int(.8*host_free), host_free - 4*GIB)) if host_free else 0
    data = dict(host_available_bytes=host_free, host_budget_bytes=host_budget,
                estimated_host_bytes=host_need, admission_policy='conservative workspace estimate')
    if cuda_free is not None:
        limit = max(0, min(int(.8*cuda_free), cuda_free - GIB))
        data.update(cuda_free_bytes=cuda_free, cuda_allocator_budget_bytes=limit)
        if limit < work:
            return 'below_gpu_budget', data
    return ('below_host_budget' if host_need > host_budget else None), data


def _step_3d(stage, device, supervision):
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
    loss = _epoch(model, names, groups, lambda name, branch: (image,label), branch, device,
                  torch.Generator(device=device).manual_seed(2023), optimizer)
    return loss, (model.head.weight.detach() - before).abs().max().item()


def _step_blood(stage, device):
    import numpy as np
    import torch
    from .xu_blood import _batch, _objective
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
    if temporal:
        # Mirror the resident encoder gradients possible after the previous encoder stage.
        for p in encoder.parameters():
            p.grad = torch.zeros_like(p)
        with torch.no_grad():
            frames = [encoder.features(images[:,i:i+1]) for i in range(7)]
        scores = sequence(frames)
    else:
        scores = encoder(images)
    loss = _objective(scores, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item(), (model.head.weight.detach() - before).abs().max().item()


def probe(stage, selected_device, supervision, output):
    import torch

    file = Path(output) / 'capacity.json'
    started = time.monotonic()
    report = dict(stage=stage, requested_device=selected_device, supervision=supervision,
                  profile='paper', input='synthetic', state='checking', optimizer_steps=0,
                  torch_version=str(torch.__version__))
    atomic_json(file, report)
    device = None
    measured_cuda = False
    try:
        device = torch.device(selected_device)
        if device.type not in ('cpu','cuda') or (device.type=='cpu' and device.index is not None):
            raise ValueError('Expected cpu or cuda[:index]')
        available = snapshot()['available_memory_bytes']
        spec = specification(stage)
        report.update(spec)
        free = None
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable in this Python')
            if device.index is None:
                device = torch.device('cuda',torch.cuda.current_device())
            torch.cuda.set_device(device)
            free, total = torch.cuda.mem_get_info(device)
            report.update(gpu_name=torch.cuda.get_device_name(device), cuda_total_bytes=total,
                          runtime_cuda=torch.version.cuda)
        report['device'] = str(device)
        reason, facts = admission(spec, available, free)
        report.update(facts)
        if reason:
            report.update(state='not_admitted', reason=reason)
        else:
            if device.type == 'cuda':
                torch.cuda.set_per_process_memory_fraction(facts['cuda_allocator_budget_bytes']/total,device)
                torch.cuda.reset_peak_memory_stats(device)
                measured_cuda = True
            torch.set_num_threads(min(4,torch.get_num_threads()))
            torch.manual_seed(2023)
            report.update(state='running_training_step')
            atomic_json(file,report)
            if stage in ('3d64','3d128'):
                loss, delta = _step_3d(stage,device,supervision)
            else:
                loss, delta = _step_blood(stage,device)
            if not math.isfinite(loss) or not math.isfinite(delta) or delta <= 0:
                raise FloatingPointError('Expected a finite loss and an actual parameter update')
            report.update(state='passed', optimizer_steps=1, loss=loss, head_max_abs_update=delta)
    except (OSError, ValueError, RuntimeError, MemoryError, FloatingPointError) as error:
        report.update(state='failed', error_type=type(error).__name__, reason=str(error))
    finally:
        if measured_cuda:
            report.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        report['seconds'] = time.monotonic()-started
        atomic_json(file,report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--supervision', choices=('marginal','complete'), default='marginal')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    with operation_lock(args.output,'xu-capacity',args.output):
        report = probe(args.stage,args.device,args.supervision,args.output)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    if report['state'] == 'passed':
        return 0
    print(f"{args.stage}: {report['reason']} (report: {args.output.resolve()/'capacity.json'})",
          file=sys.stderr, flush=True)
    return 1
