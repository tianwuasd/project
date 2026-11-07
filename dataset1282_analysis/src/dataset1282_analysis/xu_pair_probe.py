"""Exercise the formal 64³/128³ branches concurrently on two allocated CUDA devices."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from .contracts import atomic_json, read_json, sha256_file
from .observability.status import operation_lock, write_status
from .xu_dispatch import Children, plan_stage, reserve_pair, run_wave, select_devices
from .xu_pipeline import ROOT, _execute
from .xu_pipeline_stages import Stage
from .xu_resources import ResourceUnavailable


SIZES = ('3d64', '3d128')


def _selected(devices):
    import torch
    selected = select_devices(None, devices, torch)
    if len(selected) != 2:
        raise ValueError('Two physically distinct allocated CUDA GPUs are required')
    return selected


def _worker(out, size):
    import torch
    from .xu_capacity import probe
    from .xu_dispatch import device_identity

    request = read_json(out/'request.json')
    index = SIZES.index(size)
    card = request['devices'][index]
    ticket = json.loads(os.environ.get('XU_CUDA_ADMISSION', '{}'))
    if (ticket.get('policy') != 'v2' or ticket.get('device') != card['device']
            or not ticket.get('uuid') or ticket['uuid'].removeprefix('GPU-') != card['uuid'].removeprefix('GPU-')):
        raise RuntimeError('Worker CUDA allocation or UUID differs from the parent v2 admission')
    torch.cuda.set_device(card['device'])
    if device_identity(card['device'], torch) != card:
        raise RuntimeError('Physical GPU changed after parent v2 admission')
    directory = out/size
    row = dict(size=size, device=card['device'], uuid=card['uuid'], request=request['nonce'],
               pid=os.getpid(), ready_ns=time.monotonic_ns())
    atomic_json(directory/'ready.json', row)
    peer_path = out/SIZES[1-index]/'ready.json'
    deadline = time.monotonic() + request['barrier_seconds']
    while not peer_path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('The second admitted probe did not reach the launch barrier')
        time.sleep(.05)
    peer = read_json(peer_path)
    if peer['request'] != request['nonce'] or peer['uuid'] != request['devices'][1-index]['uuid']:
        raise RuntimeError('Unmatched peer at probe barrier')
    row.update(peer_pid=peer['pid'], probe_start_ns=time.monotonic_ns())
    try:
        result = probe(size, card['device'], 'marginal', directory, policy='v2')
        if result['state'] != 'passed':
            raise RuntimeError(f"{size} capacity: {result.get('reason', result['state'])}")
    finally:
        row['probe_end_ns'] = time.monotonic_ns()
        atomic_json(directory/'execution.json', row)


def _validate(out, request):
    """Use child monotonic clocks on this host, independently of launcher thread timings."""
    rows = []
    for index, size in enumerate(SIZES):
        card = request['devices'][index]
        directory = out/size
        row = read_json(directory/'execution.json')
        report = read_json(directory/'capacity.json')
        side = int(size[2:])
        if (row['request'] != request['nonce'] or row['device'] != card['device']
                or row['uuid'] != card['uuid'] or row['size'] != size
                or row['ready_ns'] > row['probe_start_ns'] or row['probe_start_ns'] >= row['probe_end_ns']):
            raise ValueError('Unmatched child timing or GPU identity')
        required = dict(stage=size, device=card['device'], profile='paper', input='synthetic',
                        resource_policy='v2', state='passed', optimizer='Adam', dtype='float32',
                        shape=[4,1,side,side,side], optimizer_steps=1)
        if (any(report.get(key) != val for key,val in required.items())
                or not math.isfinite(report['loss']) or not math.isfinite(report['head_max_abs_update'])
                or report['head_max_abs_update'] <= 0 or report['peak_cuda_allocated_bytes'] <= 0):
            raise ValueError(f'{size} lacks a measured formal CUDA Adam update')
        rows.append(dict(row, capacity_sha256=sha256_file(directory/'capacity.json'),
                         peak_cuda_allocated_bytes=report['peak_cuda_allocated_bytes']))
    left, right = rows
    if (left['pid'] == right['pid'] or left['peer_pid'] != right['pid'] or right['peer_pid'] != left['pid']
            or left['uuid'] == right['uuid'] or left['device'] == right['device']):
        raise ValueError('Missing distinct child processes or physical GPU identities')
    overlap = min(row['probe_end_ns'] for row in rows) - max(row['probe_start_ns'] for row in rows)
    if overlap <= 0:
        raise ValueError('The measured child capacity-probe intervals did not overlap')
    return dict(branches=rows, capacity_interval_overlap_seconds=overlap/1e9)


def run(args):
    out = args.output.resolve()
    if (out == ROOT or ROOT in out.parents or out in ROOT.parents
            or any((p/'.git').exists() for p in (out,*out.parents))):
        raise ValueError('Use a new private output directory outside the code checkout')
    if (args.threads is not None and args.threads <= 0) or not 0 <= args.wait_seconds <= 300:
        raise ValueError('Expected positive threads and wait seconds in 0..300')
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    with operation_lock(out, 'xu-pair-probe', out) as owner:
        state = dict(protocol='xu-dual-capacity-1', requested_devices=args.devices, policy='v2')
        children = Children(out, owner)
        try:
            try:
                cards = _selected(args.devices)
            except (ValueError, RuntimeError) as exc:
                raise ResourceUnavailable(str(exc)) from exc
            request = dict(state, devices=cards, nonce=uuid.uuid4().hex,
                           barrier_seconds=max(120,args.wait_seconds+60))
            atomic_json(out/'request.json', request)
            stages = tuple(Stage('crop'+size[2:], ('--device', card['device']), ())
                           for size,card in zip(SIZES,cards))
            # Both devices pass admission before either training subprocess starts.
            opts = argparse.Namespace(**vars(args), profile='paper', resource_policy='v2', prefetch=False)
            opts.wave_reservation = reserve_pair(opts, out)
            plans = {s.name:plan_stage(s, opts, out, children, 2, 'launch', {}) for s in stages}
            def launch(stage):
                worker_opts = copy.copy(opts)
                worker_opts.children = children
                worker_opts.device, worker_opts.overlay = plans[stage.name]
                size = '3d' + stage.name[4:]
                _execute('xu_pair_probe.py', ['_worker', out, size], size, out, owner, worker_opts)
            run_wave(stages, launch, children)
            state.update(_validate(out, request), state='passed')
        except Exception as exc:
            state.update(state='not_admitted' if isinstance(exc, ResourceUnavailable) else 'failed',
                         reason=str(exc), error_type=type(exc).__name__)
        except KeyboardInterrupt:
            state.update(state='interrupted', reason='Interrupted by operator')
        finally:
            children.stop()
            atomic_json(out/'result.json', state)
            write_status(out, **state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return 0 if state.get('state') == 'passed' else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_worker':
        sub = argparse.ArgumentParser()
        sub.add_argument('output', type=Path)
        sub.add_argument('size', choices=SIZES)
        args = sub.parse_args(argv[1:])
        _worker(args.output, args.size)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', required=True, help='Two allocated CUDA devices, e.g. cuda:0,cuda:1')
    parser.add_argument('--output', type=Path, required=True, help='New private output directory')
    parser.add_argument('--threads', type=int, help='Maximum CPU threads per worker within its allocation')
    parser.add_argument('--wait-seconds', type=int, default=0, help='Bound for each resource admission')
    try:
        return run(parser.parse_args(argv))
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
