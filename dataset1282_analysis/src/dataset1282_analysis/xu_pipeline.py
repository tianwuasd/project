"""Run the existing Xu stages in order and verify their artifacts before resuming."""
import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contracts import atomic_json, digest, read_json, sha256_file
from .xu_dispatch import Children, plan_stage, reserve_pair, run_wave, select_devices, waves
from .observability.status import lock_active, operation_lock, write_status


ROOT = Path(__file__).resolve().parents[2]
PADDING = 20
STAGE_NAMES = ('crop64', 'crop128', 'localize', 'audit-roi', 'all64', 'all128',
               'evaluate-all64', 'evaluate-all128', 'four-way', 'targets', 'init64',
               'init128', 'six-way', 'blood', 'with-blood', 'vessels')


def _input_file(path):
    if path is None:
        return None
    return dict(path=str(path), sha256=sha256_file(path))


def _sources(technical):
    source = Path(__file__).resolve().parent
    files = [*source.glob('xu_*.py'), source/'training/marginal.py', source/'contracts.py',
             source/'observability/status.py', ROOT/'scripts/xu_seg.py', ROOT/'scripts/xu_pipeline.py']
    files = [p for p in files if p.name not in ('xu_pair_probe.py','xu_inference_probe.py')]  # Diagnostics.
    if technical:
        files = [p for p in files if p.name != 'xu_capacity.py']
    else:
        files.append(ROOT/'scripts/xu_capacity.py')
    return {p.relative_to(ROOT).as_posix():sha256_file(p) for p in files}


def _contract(cache, args):
    import torch
    from .xu_workflow import artifact_identity
    from .xu_roi import DEFAULT_PADDING_PERCENT

    if DEFAULT_PADDING_PERCENT != PADDING:
        raise ValueError('Pipeline and locator margin protocols differ')
    counts = {r: dict(all=len(cache.case_ids(r)), complete=len(cache.case_ids(r, complete_only=True)))
              for r in ('train', 'development')}
    technical = cache.scope == 'technical_subset'
    if technical != args.technical_subset or technical != (args.profile == 'smoke'):
        raise ValueError('Use smoke with --technical-subset for technical data; paper requires the full cache')
    if not technical and counts != {'train': {'all':590, 'complete':552},
                                    'development': {'all':128, 'complete':122}}:
        raise ValueError(f'Prepared cohort differs from the approved dataset1282 fixed split: {counts}')
    devices = select_devices(args.device, args.devices, torch)
    if args.resource_policy is None:
        args.resource_policy = 'v2' if len(devices) > 1 or os.name == 'posix' else 'legacy'
    if len(devices) > 1 and args.resource_policy != 'v2':
        raise ValueError('Multi-device scheduling requires the v2 resource policy')
    return dict(protocol='xu-model-run-3', fleet=True, prepared=str(cache.run_dir), request=cache.request_digest,
                artifacts=artifact_identity(cache), scope=cache.scope, counts=counts,
                implementation=_sources(technical), profile=args.profile, supervision=args.supervision,
                seed=args.seed, padding_percent=PADDING, device=devices[0], torch=str(torch.__version__),
                devices=devices, resource_policy=args.resource_policy, prefetch=args.prefetch,
                prefetch_mib=args.prefetch_mib,
                cpu_threads=args.threads,
                libraries={n:version(n) for n in ('numpy','scipy','nibabel','blosc2','scikit-image')},
                supplied_crop64=_input_file(args.crop64), supplied_crop128=_input_file(args.crop128),
                reviews=_input_file(args.reviews))


def _files(directory, required):
    for name in required:
        if not (directory/name).is_file():
            raise ValueError(f'Missing required output: {directory/name}')
    files = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if relative.parts[0] == '.runtime':
            continue
        if path.resolve() != path:
            raise ValueError(f'Unexpected link in stage output: {relative}')
        if path.is_file():
            files[relative.as_posix()] = sha256_file(path)
    return files


def _tail(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 6000))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    return next((line.strip()[:750] for line in reversed(lines) if line.strip()), '')


class StageError(RuntimeError):
    pass


def _execute(script, argv, name, out, owner, args):
    log = out/'logs'/f'{name}-{time.time_ns()}.log'
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    if args.threads is not None:
        env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS=str(args.threads))
    args.children.ensure_running()
    for key in ('XU_RESOURCE_POLICY','XU_HOST_BUDGET_BYTES','XU_HOST_DEMAND_BYTES',
                'XU_WAVE_RESERVATION','XU_CUDA_ADMISSION','XU_CUDA_ADMISSIONS','XU_BRANCH_DEVICES',
                'XU_PREFETCH','XU_PREFETCH_BYTES'):
        env.pop(key, None)
    env.update(args.overlay)
    command = [sys.executable, '-B', '-u', str(ROOT/'scripts'/script), *map(str, argv)]
    print(f'Running {name} | {log}', flush=True)
    with log.open('xb') as stream:
        stream.write((json.dumps(command, ensure_ascii=False)+'\n').encode('utf-8'))
        stream.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            args.children.attach(child, name, args.device, log, getattr(args,'assigned_devices',None))
            code = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            args.children.detach(child)
    if code:
        raise StageError(f'{name} exited {code}: {_tail(log)} (log: {log})')


def _restore_attempt(stage, directory, out):
    if not directory.exists():
        return []
    if lock_active(directory/'.runtime/operation.lock'):
        raise RuntimeError(f'{stage.name} is still owned by an active process')
    if stage.resume_marker and (directory/stage.resume_marker).is_file():
        return ['--resume']
    # Fresh-only stages (including ROI audits) cannot adopt partial files. Retain the failed attempt.
    archive = out/'attempts'/f'{stage.name}-{time.time_ns()}'
    archive.parent.mkdir(exist_ok=True)
    directory.rename(archive)
    return []


def _check_manifests(stage, out, records):
    for flag, source, files in [
            ('--candidates','localize',('candidates.json',)),
            ('--boxes','audit-roi',('boxes.json','localization_qc.json')),
            ('--targets','targets',('targets.json',))]:
        if flag in stage.argv:
            for name in files:
                if sha256_file(out/source/name) != records[source]['files'][name]:
                    raise ValueError(f'{stage.name} input changed: {source}/{name}')
    if '--fusion' in stage.argv:
        source = Path(stage.argv[stage.argv.index('--fusion')+1]).name
        if sha256_file(out/source/'complete.json') != records[source]['files']['complete.json']:
            raise ValueError(f'Fusion result changed before {stage.name}')


def _check_used_weights(stage, directory, records, contract):
    command = stage.argv[0]
    if command=='localize':
        used = read_json(directory/'candidates.json')['payload']['checkpoints']
    elif command=='evaluate':
        branch = stage.argv[stage.argv.index('--branch')+1]
        used = {branch:read_json(directory/'evaluation.json')['checkpoint_sha256']}
    elif command=='fuse':
        used = read_json(directory/'job.json')['checkpoints']
    else:
        return
    for branch, actual in used.items():
        supplied = contract.get('supplied_'+branch)
        expected = supplied['sha256'] if supplied else records[branch]['files']['best.pt']
        if actual != expected:
            raise ValueError(f'{stage.name} consumed different {branch} weights than the frozen run')


def _capacity(stage, checked, out, owner, args):
    if args.profile != 'paper' or stage.skip_reason:
        return
    if stage.name == 'blood':
        sizes = ('blood-encoder', 'blood-sequence')
    elif stage.name in ('crop64', 'all64', 'init64'):
        # Validate the larger branch before spending a full training run on its smaller partner.
        sizes = ('3d64',) if args.stop_after == stage.name or len(args.device_names) > 1 else ('3d64','3d128')
    elif stage.name in ('crop128', 'all128', 'init128'):
        sizes = ('3d128',)
    else:
        return
    for size in sizes:
        group = tuple(getattr(args,'assigned_devices',None) or [args.device])
        if (group, size) in checked:
            continue
        directory = out/'capacity'/f'{size}-{time.time_ns()}'
        try:
            _execute('xu_capacity.py', ['--stage',size, '--device',args.device, '--supervision',
                     args.supervision,'--policy',args.resource_policy,'--output',str(directory)],
                     'capacity-'+size, out, owner, args)
        except StageError as exc:
            if (directory/'capacity.json').is_file():
                evidence = read_json(directory/'capacity.json')
                raise StageError(f'Capacity {size}: {evidence.get("reason", evidence["state"])}. '
                                 f'See {directory/"capacity.json"}') from exc
            raise
        report = read_json(directory/'capacity.json')
        if report.get('state') != 'passed' or report.get('optimizer_steps') != 1:
            raise StageError(f'Capacity check did not complete a training step for {size}: {directory}')
        per_device = report.get('per_device', {})
        checked[group, size] = {d: values['peak_cuda_reserved_bytes'] for d,values in per_device.items()
                                if 'peak_cuda_reserved_bytes' in values}
        checked[group, size].setdefault(args.device, report.get('peak_cuda_reserved_bytes', 0))
        if len(group)==1:
            checked[args.device, size] = checked[group,size][args.device]


def _result(out, cache, contract, records):
    from .xu_pipeline_stages import auxiliary_counts
    from .xu_targets import anatomy
    comparisons = {}
    for name in ('evaluate-all64','evaluate-all128','four-way','six-way','with-blood','vessels'):
        if records[name]['state'] == 'complete':
            report = read_json(out/name/'evaluation.json')
            comparisons[name] = {key:report[key] for key in ('full_annotation','observed_structure')}
    return dict(contract=digest(contract), scope=contract['scope'], supervision=contract['supervision'],
                padding_percent=PADDING, counts=contract['counts'],
                stages={k:v['state'] for k,v in records.items()},
                skipped={k:v['reason'] for k,v in records.items() if v['state']=='skipped'},
                auxiliary_eligible=auxiliary_counts(cache, out/'targets', contract['supervision']),
                anatomy=anatomy(cache),
                evaluation='fixed development set used for checkpoint selection',
                comparisons=comparisons, native_predictions='vessels/cases')


def _stage_spec(stage, contract_id, group=None):
    return digest(dict(contract=contract_id, name=stage.name, argv=list(map(str,stage.argv)),
                       devices=group, required=list(stage.required), skip=stage.skip_reason))


def _run_stage(stage, args, out, owner, contract, contract_id, checked, records, concurrent):
    directory = out/stage.name
    if stage.name not in STAGE_NAMES or stage.name in records or directory.resolve()!=directory:
        raise ValueError('Unsafe or duplicate pipeline stage')
    receipt = out/'receipts'/(stage.name+'.json')
    spec = _stage_spec(stage, contract_id, getattr(args,'assigned_devices',None))
    if _sources(args.technical_subset) != contract['implementation']:
        raise ValueError('Implementation changed while the pipeline was running')
    if receipt.exists():
        entry = read_json(receipt)
        if entry['spec'] != spec or entry['state'] != ('skipped' if stage.skip_reason else 'complete'):
            raise ValueError(f'Invalid receipt for {stage.name}')
        if not stage.skip_reason and _files(directory, stage.required) != entry['files']:
            raise ValueError(f'Completed artifacts changed: {stage.name}')
        print(f'Verified {stage.name}', flush=True)
    else:
        started = time.time()
        if stage.skip_reason:
            entry = dict(state='skipped', reason=stage.skip_reason)
            print(f'Skipped {stage.name}: {stage.skip_reason}', flush=True)
        else:
            ticket = out/'started'/(stage.name+'.json')
            if directory.exists() and not ticket.exists():
                raise ValueError(f'Unowned output directory: {directory}')
            if ticket.exists() and read_json(ticket).get('spec') != spec:
                raise ValueError(f'Unmatched previous attempt for {stage.name}')
            if stage.name=='audit-roi' and (out/'rejected-roi.json').exists():
                rejection = read_json(out/'rejected-roi.json')
                if (rejection.get('spec') != spec or
                        rejection.get('qc_sha256') != sha256_file(directory/'localization_qc.json')):
                    raise ValueError('Rejected ROI evidence changed')
                raise StageError('This frozen run already failed the ROI audit. See audit-roi/localization_qc.json')
            if stage.name == 'targets' and _input_file(args.reviews) != contract['reviews']:
                raise ValueError('External review changed since the run started')
            if stage.name=='localize' and args.crop64 and (
                    _input_file(args.crop64) != contract['supplied_crop64'] or
                    _input_file(args.crop128) != contract['supplied_crop128']):
                raise ValueError('External locator weights changed since the run started')
            _check_manifests(stage,out,records)
            if 'devices' in contract:
                args.device, args.overlay = plan_stage(stage, args, out, args.children, concurrent, 'admit', checked)
            _capacity(stage, checked, out, owner, args)
            if 'devices' in contract:
                args.device, args.overlay = plan_stage(stage, args, out, args.children, concurrent, 'launch', checked)
            atomic_json(ticket, dict(spec=spec))
            recovery = _restore_attempt(stage, directory, out)
            _execute('xu_seg.py', [*stage.argv,*recovery], stage.name, out, owner, args)
            _check_used_weights(stage,directory,records,contract)
            if stage.name=='targets' and args.reviews:
                details = read_json(directory/'target-request.json')
                if (_input_file(args.reviews) != contract['reviews'] or
                    digest(read_json(args.reviews)['cases']) != details['reviews']):
                    raise ValueError('Auxiliary targets used a different review book')
            entry = dict(state='complete', files=_files(directory, stage.required))
        entry.update(spec=spec, seconds=time.time()-started)
        atomic_json(receipt, entry)
    return entry


def _run_locked(cache, args, out, owner, contract):
    from .xu_pipeline_stages import iter_stages
    marker = out/'run.json'
    if marker.exists():
        if not args.resume or read_json(marker) != contract:
            raise ValueError('Use --resume with identical inputs, model settings and implementation, or a fresh output')
    else:
        if args.resume or any(p.name != '.runtime' for p in out.iterdir()):
            raise ValueError('A new pipeline requires an empty directory; resume requires its run.json')
        atomic_json(marker, contract)

    contract_id = digest(contract)
    checked, records = {}, {}
    children = Children(out, owner)
    stage_name = 'setup'
    active = {}
    try:
        from .xu_fleet import assign_wave
        def groups():
            nonlocal stage_name
            for candidates in waves(iter_stages(cache,args,out),getattr(args,'device_names',()),args.stop_after):
                stage_name = candidates[0].name
                if not contract.get('fleet'):
                    yield [(stage,None) for stage in candidates]
                else:
                    yield from assign_wave(candidates,args,out,cache,children,contract_id)
        for assigned in groups():
            members = {stage.name:group for stage,group in assigned}
            wave = [stage for stage,group in assigned]
            stage_name = wave[0].name
            active = {stage.name: stage for stage in wave}
            runnable = sum(not stage.skip_reason and not (out/'receipts'/(stage.name+'.json')).exists()
                           for stage in wave)
            reservation = reserve_pair(args, out) if runnable == 2 and 'devices' in contract else None

            def worker(stage):
                branch_args = copy.copy(args)
                branch_args.children = children
                branch_args.assigned_devices = members[stage.name]
                if contract.get('fleet') and (out/'assignments'/(stage.name+'.json')).is_file():
                    branch_args.inference_worker_bytes = read_json(out/'assignments'/(stage.name+'.json')).get('host_per_inference_case') or 0
                branch_args.wave_reservation = reservation
                branch_args.peer_name = next((s.name for s in wave if s.name != stage.name), None)
                try:
                    return _run_stage(stage, branch_args, out, owner, contract, contract_id,
                                      checked, records, max(1, runnable))
                except (Exception, KeyboardInterrupt) as exc:
                    exc.xu_stage = stage.name
                    raise
            for stage, entry in zip(wave, run_wave(wave, worker, children)):
                records[stage.name] = entry
            if args.stop_after in active and not (out/'result.json').exists():
                write_status(out,state='stopped_after_stage',stage=args.stop_after,finished=list(records))
                return

        if set(records) != set(STAGE_NAMES):
            raise ValueError('Pipeline ended before recording every required stage')
        result = _result(out, cache, contract, records)
        if (out/'result.json').exists() and read_json(out/'result.json') != result:
            raise ValueError('Final result differs from verified stage evidence')
        if not (out/'result.json').exists():
            atomic_json(out/'result.json',result)
        write_status(out, state='complete',stage='vessels',finished=list(records),result=str(out/'result.json'))
        print(f'Completed | {out/"result.json"}', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        children.stop()
        stage_name = getattr(exc,'xu_stage',stage_name)
        for name in active:
            receipt = out/'receipts'/(name+'.json')
            if name not in records and receipt.is_file():
                records[name] = read_json(receipt)
        state = 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        qc = out/'audit-roi'/'localization_qc.json'
        if stage_name=='audit-roi' and isinstance(exc,StageError) and qc.is_file() and read_json(qc).get('accepted') is False:
            state = 'roi_rejected'
            if not (out/'rejected-roi.json').exists():
                atomic_json(out/'rejected-roi.json', dict(spec=_stage_spec(active[stage_name],contract_id,members[stage_name]),
                                                          qc_sha256=sha256_file(qc)))
        previous = read_json(out/'status.json') if (out/'status.json').exists() else {}
        write_status(out,state=state,stage=stage_name,error=str(exc),finished=list(records),
                     last_command=previous.get('stage'),log=previous.get('log'),
                     roi_report=str(qc) if state=='roi_rejected' else None)
        raise


def _run(args):
    from .xu_geometry import PreparedCache
    out, prepared = args.output.resolve(), args.prepared.resolve()
    if out==prepared or out in prepared.parents or prepared in out.parents:
        raise ValueError('Use separate prepared and output directories')
    if ROOT == out or ROOT in out.parents:
        raise ValueError('Place private experiment outputs outside the code checkout')
    if bool(args.crop64) != bool(args.crop128):
        raise ValueError('Provide both locator checkpoints or train both in this run')
    if args.threads is not None and args.threads < 1:
        raise ValueError('Thread count must be positive')
    for name in ('crop64','crop128','reviews'):
        path = getattr(args,name)
        if path is not None:
            path = path.resolve(strict=True)
            if not path.is_file() or out in path.parents:
                raise ValueError('External inputs must be files outside the pipeline output')
            setattr(args,name,path)
    if not 0 <= getattr(args,'wait_seconds',0) <= 300:
        raise ValueError('Resource wait must be between 0 and 300 seconds')
    if getattr(args,'prefetch_mib',4096) < 1:
        raise ValueError('The prefetch memory ceiling must be positive')
    args.prepared, args.output = prepared, out
    cache = PreparedCache(prepared,allow_technical_subset=args.technical_subset)
    contract = _contract(cache,args)
    # The frozen run and stage receipts must use the same spelling for CUDA aliases.
    args.device = contract['device']['device']
    args.device_names = [row['device'] for row in contract.get('devices', [contract['device']])]
    out.mkdir(exist_ok=True,parents=True)
    with operation_lock(out,'xu-model-run',out) as owner:
        _run_locked(cache,args,out,owner,contract)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    run = commands.add_parser('run')
    run.add_argument('--prepared',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--supervision',choices=('marginal','complete'),default='marginal')
    run.add_argument('--profile',choices=('paper','smoke'),default='paper')
    run.add_argument('--technical-subset',action='store_true')
    selection = run.add_mutually_exclusive_group()
    selection.add_argument('--device', help='Select a single CUDA device or CPU instead of the automatic pool')
    selection.add_argument('--devices', help='auto (default) scans the allocated visible pool, or list cuda:0,cuda:1,...')
    run.add_argument('--resource-policy',choices=('legacy','v2'),
                     help='Linux and multi-GPU default to v2; a single local Windows GPU defaults to legacy')
    run.add_argument('--wait-seconds',type=int,default=0, help='Bounded wait per resource check, 0 to 300 seconds')
    run.add_argument('--prefetch-mib',type=int,default=4096, help='Maximum additional CPU batch memory per branch (MiB)')
    run.add_argument('--prefetch',action='store_true',
                     help='Enable one bounded CPU look-ahead batch per 3D training branch')
    run.add_argument('--threads',type=int,help='CPU numerical threads per child process')
    run.add_argument('--seed',type=int,default=2023)
    run.add_argument('--crop64',type=Path,help='Optional frozen 64-cube whole-scan locator')
    run.add_argument('--crop128',type=Path,help='Optional matching 128-cube whole-scan locator')
    run.add_argument('--reviews',type=Path,help='Private verified cut planes and optional inner interfaces')
    run.add_argument('--resume',action='store_true')
    run.add_argument('--stop-after',choices=STAGE_NAMES,help='Stop after committing this stage; continue with --resume')
    commands.add_parser('status').add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == 'status':
            row = read_json(args.output/'status.json')
            row['owner_active'] = lock_active(args.output/'.runtime/operation.lock')
            print(json.dumps(row,ensure_ascii=False,indent=2))
        else:
            _run(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc),file=sys.stderr,flush=True)
        return 1
