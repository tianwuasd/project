import argparse
from collections import Counter
from math import prod
from pathlib import Path
from types import SimpleNamespace

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import operation_lock, write_status
from ..storage_paths import assert_output_location
from . import PROTOCOL
from .heldout import GROUPS, plan_inputs
from .source import freeze_source, implementation, inspect_source


def validate_output(args):
    from ..xu_pipeline import ROOT
    out = assert_output_location(args.output)
    config = read_json(args.config)
    for protected in (args.prepared.resolve(),args.training.resolve(),Path(config['data_root']).resolve(),ROOT):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError('Prediction output must be separate from input and training')
    for protected in (Path(config['metadata_excel']).resolve(),args.config.resolve(),args.policy.resolve()):
        if out == protected or out in protected.parents:
            raise ValueError('Prediction output contains an input configuration or workbook')
    return config


def open_job(args):
    from ..input_source import open_prepared as PreparedCache
    out = assert_output_location(args.output)
    config = validate_output(args)
    cache = PreparedCache(args.prepared,allow_technical_subset=args.technical_subset)
    cases = plan_inputs(cache,config,args.policy,args.groups,args.cases)
    source,_ = inspect_source(args.training,cache,args.mode)
    dataset = cache.run_dir/'preprocessed/Dataset282_CHDSource'
    job = dict(protocol=PROTOCOL,prepared=str(cache.run_dir),source=digest(source),
               mode=args.mode,cases=cases,technical_subset=args.technical_subset,
               plans=sha256_file(dataset/'nnUNetPlans.json'),dataset=sha256_file(dataset/'dataset.json'),
               implementation=implementation())
    path = out/'run.json'
    if path.exists():
        if not args.resume or read_json(path) != job:
            raise ValueError('Resume requires identical held-out inputs, models and implementation')
    else:
        if args.resume:
            raise ValueError('Prediction has no run to resume')
        atomic_json(path,job)
    freeze_source(args.training,cache,out,args.mode)
    return out,cache,job


def execute(args,out,owner,job,phase):
    import torch
    from ..xu_dispatch import Children, select_devices, prefetch_cpu_allocation
    from ..xu_gpu_probe import filter_physical_devices, gpu_snapshot
    from ..xu_pipeline import _execute
    from ..xu_reallocation import run_with_reallocation
    from ..xu_resources import ResourceUnavailable, stage_resources, combined_environment
    from ..xu_fleet import needs_for_width
    from ..xu_supply_budget import batch_bytes
    from .heldout import HeldoutCache, prepare_memory
    if phase == 'prepare' or args.device == 'cpu':
        devices = ['cpu']
    else:
        pool = select_devices(args.device,None if args.device else args.devices,torch)
        excluded = sorted({0,1,*args.exclude_physical_gpus})
        devices = [r['device'] for r in filter_physical_devices(pool,excluded,gpu_snapshot())]
        if not devices:
            raise ValueError('No authorized GPU remains after physical exclusions')
    if phase == 'prepare':
        host = prepare_memory(job)
        cuda = 0
    else:
        cache = HeldoutCache(out)
        voxels = max(prod(cache.geometry(n).cache_shape) for n in job['cases'])
        host = max(batch_bytes(voxels,1,128)*2,
                   max(prod(row['shape'])*64 for row in job['cases'].values())) + 2*1024**3
        cuda = needs_for_width('four-way' if job['mode']=='four-way' else 'with-blood',1,
                               read_json(out/'source/manifest.json')['run']['profile'])[0]
    args.resource_recovery = True
    args.resource_poll_seconds = args.wait_seconds
    args.prefetch_workers = 1
    args.prefetch_depth = 1
    args.max_reallocations = 3

    def attempt(running):
        from .workflow import verify_job
        verify_job(out)
        children = Children(out,owner)
        plan = None
        errors = []
        for candidate in devices:
            try:
                plan = stage_resources([candidate],threads=running.threads,policy='v2',
                    owned_pids=children.pids(),require_pid_evidence=candidate!='cpu',
                    host_demand_bytes=host,cuda_demand_bytes=0 if candidate=='cpu' else cuda)
            except ResourceUnavailable as error:
                errors.append(str(error))
                continue
            break
        if plan is None:
            raise ResourceUnavailable('; '.join(errors))
        device = next(iter(plan['devices']))
        threads,workers = prefetch_cpu_allocation(plan['cpu_slots'],plan['threads'],running.prefetch_workers)
        plan['threads'] = threads
        overlay = combined_environment(plan,[device])
        overlay.update(XU_PREFETCH='1' if phase=='predict' else '0',
                       XU_PREFETCH_BYTES=str(plan['host_per_stage_bytes']//2),
                       XU_PREFETCH_WORKERS=str(workers),XU_PREFETCH_DEPTH='1')
        settings = SimpleNamespace(threads=threads,children=children,overlay=overlay,
                                   device=device,assigned_devices=[device])
        atomic_json(out/'resources'/(phase+'-admission.json'),plan)
        try:
            _execute('xu_predict.py',['worker','--output',str(out),'--device',device,'--phase',phase],
                     phase,out,owner,settings)
        finally:
            children.stop()
        verify_job(out)
    run_with_reallocation(attempt,args,out)


def run(args):
    if args.threads < 1 or not 1 <= args.wait_seconds <= 300:
        raise ValueError('Positive CPU threads and wait-seconds within 1..300 are required')
    out = assert_output_location(args.output)
    validate_output(args)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError('New prediction requires a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    with operation_lock(out,'xu-model-run',out) as owner:
        try:
            out,_,job = open_job(args)
            print('Frozen held-out groups:',dict(Counter(r['role'] for r in job['cases'].values())),flush=True)
            execute(args,out,owner,job,'prepare')
            execute(args,out,owner,job,'predict')
            result = read_json(out/'complete.json')
            write_status(out,state='complete',stage='prediction',cases=len(result['cases']),
                         rejected=result['rejected'],active=[])
        except (Exception,KeyboardInterrupt) as error:
            write_status(out,state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                         error=str(error),active=[])
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Frozen dataset1282 held-out prediction')
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('run')
    for name in ('prepared','training','config','policy','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--groups',choices=GROUPS,nargs='+',default=list(GROUPS))
    p.add_argument('--case',dest='cases',action='append')
    p.add_argument('--mode',choices=('four-way','blood','vessels'),default='vessels')
    p.add_argument('--technical-subset',action='store_true')
    p.add_argument('--resume',action='store_true')
    selected = p.add_mutually_exclusive_group()
    selected.add_argument('--device')
    selected.add_argument('--devices',default='all-visible')
    p.add_argument('--exclude-physical-gpus',type=int,nargs='*',default=[0,1],
                   help='Additional physical exclusions; physical GPUs 0 and 1 are always excluded')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--wait-seconds',type=int,default=60)
    for name in ('status','evaluate','worker'):
        p = commands.add_parser(name)
        p.add_argument('--output',type=Path,required=True)
        if name == 'worker':
            p.add_argument('--phase',choices=('prepare','predict'),required=True)
            p.add_argument('--device',required=True)
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(read_json(args.output/'status.json'))
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        evaluate(assert_output_location(args.output))
    elif args.command == 'worker':
        from ..xu_resources import apply_env_limit
        from .workflow import prepare,predict
        apply_env_limit(args.device)
        output = assert_output_location(args.output)
        if args.phase == 'prepare':
            prepare(output)
        else:
            predict(output,args.device)
    else:
        run(args)
