"""Qualify usable devices in the allocated pool with actual fixed-batch Xu updates."""
from argparse import ArgumentParser
import os
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dataset1282_analysis.contracts import atomic_json, read_json
from dataset1282_analysis.observability.status import operation_lock
from dataset1282_analysis.xu_dispatch import Children, select_devices
from dataset1282_analysis.xu_fleet import available, distribute
from dataset1282_analysis.xu_pipeline import _execute
from dataset1282_analysis.xu_resources import combined_environment, stage_resources


def inference_check(args,out,owner,children,free,state):
    from types import SimpleNamespace
    from dataset1282_analysis.xu_inference_pool import inference_lane_bytes
    lane_bytes=inference_lane_bytes([SimpleNamespace(cache_shape=(64,)*3,native_shape_xyz=(64,)*3)])
    host=stage_resources(['cpu'],policy=args.resource_policy)['host_budget_bytes']
    cards=[card for card,budget in free.items() if budget>=256*1024**2][:host//lane_bytes]
    if not cards:
        raise ValueError('Insufficient host or CUDA admission for one synthetic inference lane')
    plan=stage_resources(cards,threads=args.threads,policy=args.resource_policy,owned_pids=children.pids(),
                         host_demand_bytes=lane_bytes*len(cards),cuda_demand_bytes=256*1024**2,
                         require_pid_evidence=args.resource_policy=='v2' and os.name=='posix')
    args.overlay=combined_environment(plan,cards)
    args.device=cards[0]
    args.assigned_devices=cards
    args.children=children
    atomic_json(out/'inference-admission.json',plan)
    # A CUDA driver stall cannot unwind a Python barrier; only the owned probe process is stopped.
    timeout = threading.Timer(180, children.stop)
    timeout.daemon = True
    timeout.start()
    try:
        _execute('xu_fleet_probe.py',['_infer',out,*cards],'inference',out,owner,args)
    finally:
        timeout.cancel()
        timeout.join()
    report=read_json(out/'inference.json')
    if report.get('state')!='passed' or report.get('devices')!=cards:
        raise ValueError('Missing successful forward from each assigned inference GPU')
    state['groups'].append(dict(devices=cards,report=str(out/'inference.json')))
    state['not_assigned_for_memory']=[d for d in free if d not in cards]


def capacity_groups(args,out,owner,children,remaining,state):
    from dataset1282_analysis.xu_capacity import device_demands, specification
    spec=specification(args.stage)
    maximum=7 if args.stage=='blood-sequence' else 4
    def need(_stage,width):
        devices=[f'cuda:{i}' for i in range(width)]
        return list(device_demands(args.stage,devices,spec).values())
    while remaining:
        found=distribute([args.stage],{args.stage:maximum},remaining,need)
        if not found:
            state['unqualified']=list(remaining)
            raise ValueError('Remaining cards cannot admit this formal shape with the selected global batch')
        group=found[args.stage]
        demand=device_demands(args.stage,group,spec)
        plan=stage_resources(group,threads=args.threads,policy=args.resource_policy,owned_pids=children.pids(),
                             cuda_demand_bytes=demand,host_demand_bytes=spec['estimated_host_bytes'],
                             require_pid_evidence=args.resource_policy=='v2' and os.name=='posix')
        args.overlay=combined_environment(plan,group)
        args.device=group[0]
        args.assigned_devices=group
        args.children=children
        path=out/f'group-{len(state["groups"])+1}'
        atomic_json(out/(path.name+'-admission.json'),plan)
        _execute('xu_capacity.py',['--stage',args.stage,'--device',args.device,'--policy',args.resource_policy,
                 '--supervision','marginal','--output',path],path.name,out,owner,args)
        result=read_json(path/'capacity.json')
        measured=result.get('per_device',{})
        if (result.get('state')!='passed' or result.get('optimizer_steps')!=1 or set(measured)!=set(group)
                or any(m.get('peak_cuda_allocated_bytes',0)<=0 for m in measured.values())):
            raise ValueError('Missing optimizer or per-device measurement')
        state['groups'].append(dict(devices=group,work=result.get('distribution'),report=str(path/'capacity.json')))
        for d in group: remaining.pop(d)


def run(args):
    import torch
    out = args.output.resolve()
    if any((p/'.git').exists() for p in (out,*out.parents)) or out.exists():
        raise ValueError('Select a new private output directory outside a Git checkout')
    if args.threads is not None and args.threads < 1:
        raise ValueError('Expected a positive numerical thread limit')
    out.mkdir(mode=0o700,parents=True)
    state = dict(stage=args.stage, requested=args.devices, groups=[], unavailable={})
    with operation_lock(out,'xu-fleet-probe',out) as owner:
        children=Children(out,owner)
        try:
            cards=select_devices(None,args.devices,torch)
            if any(row['device']=='cpu' for row in cards):
                raise ValueError('This diagnostic requires allocated CUDA devices')
            args.resource_policy = args.resource_policy or ('v2' if len(cards)>1 or os.name=='posix' else 'legacy')
            if len(cards)>1 and args.resource_policy!='v2':
                raise ValueError('GPU pools require v2 admission')
            pool=[c['device'] for c in cards]
            state.update(authorized=cards, policy=args.resource_policy)
            remaining, state['unavailable'] = available(pool,args,children)
            if not remaining:
                raise ValueError('No idle GPU passed admission')
            if args.stage=='inference':
                inference_check(args,out,owner,children,remaining,state)
            else:
                capacity_groups(args,out,owner,children,remaining,state)
            state['state']='passed'
        except (Exception, KeyboardInterrupt) as exc:
            state.update(state='failed',reason=str(exc))
        finally:
            children.stop()
            atomic_json(out/'result.json',state)
    print(state,flush=True)
    return 0 if state['state']=='passed' else 1


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='_infer':
        from dataset1282_analysis.xu_inference_probe import run_worker
        if len(argv)<3:
            raise ValueError('Missing output and assigned inference GPUs')
        result = run_worker(Path(argv[1]),argv[2],argv[2:])
        return 0 if result.get('state')=='passed' else 1
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--devices',default='auto')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=('inference','3d64','3d128','blood-encoder','blood-sequence'),default='3d64')
    parser.add_argument('--threads',type=int)
    parser.add_argument('--resource-policy',choices=('v2','legacy'))
    try:
        return run(parser.parse_args(argv))
    except (OSError,RuntimeError,ValueError) as exc:
        print(str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    sys.exit(main())
