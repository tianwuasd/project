import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest import TestCase


def validation_worker(rank, config, prepared, common, allocation, gate, port, trained):
    import torch
    import torch.distributed as dist
    from chd_segmentation.artifacts.store import write_json
    from chd_segmentation.models.execution import model_memory
    from chd_segmentation.models.q_route1.adapter import QAdapter
    from chd_segmentation.models.q_route1.validation_runtime import validate_parallel
    from chd_segmentation.models.supply import AdmittedCases
    from chd_segmentation.models.workflow import load_state

    torch.set_num_threads(1)
    with model_memory(common) as memory:
        dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}?use_libuv=0', rank=rank, world_size=2)
        torch.manual_seed(config['seed'])
        bundle = QAdapter(config, prepared, 'base', initialize=not trained)
        if trained:
            state = load_state(Path(config['run_dir'])/'base/latest.pt')
            bundle.model.load_state_dict(state['model'])
            del state
        bundle.model.eval()
        dataset = AdmittedCases(prepared,dict(cache_bytes=16*1024**2,reserve_ram_bytes=512*1024**2),
                                 rank=rank,gate=gate,device=torch.device('cpu'))
        value = validate_parallel(bundle,dataset,['ct_1003','ct_1012'],rank,common,allocation,
                                  gate,torch.device('cpu'),memory,2)
        write_json(Path(config['run_dir'])/f'validation-result-{rank}.json',dict(loss=value))
        dist.destroy_process_group()


def verify_parallel(config, prepared, trained=True):
    import socket
    import torch
    from chd_segmentation.artifacts.store import read_json
    from chd_segmentation.models.execution import admitted
    from chd_segmentation.models.supply import CpuReady, OrderedSupply

    os.environ['USE_LIBUV'] = '0'
    with admitted(config,prepared,'base') as (common,allocation):
        session = common.pop('resource_session')
        context = torch.multiprocessing.get_context('spawn')
        root = Path(config['run_dir'])/'base'
        ready = CpuReady(context,2,root,owner=allocation['owner'],handle=common['resource_handle'])
        gate = OrderedSupply(context,allocation['shares'],allocation['supply']['reader_slots'],ready,root,
            capacity_bytes=allocation['supply']['breakdown']['case_workspaces_bytes'],
            estimates=allocation['supply']['case_estimates'],handle=common['resource_handle'])
        with socket.socket() as socket_owner:
            socket_owner.bind(('127.0.0.1',0))
            port = socket_owner.getsockname()[1]
        workers = torch.multiprocessing.spawn(validation_worker,
            args=(config,prepared,common,allocation,gate,port,trained),nprocs=2,join=False)
        while not workers.join(timeout=1):
            session.service_replans()
    values = [read_json(Path(config['run_dir'])/f'validation-result-{rank}.json')['loss'] for rank in range(2)]
    assert values[0] == values[1]
    return values[0]


def final_checks(args):
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project/'src'),str(Path(args.tool_root)/'src')]
    root, report = Path(args.work_root), Path(args.report)
    runtime = root/'runtime'
    os.environ.update(TEMP=str(runtime),TMP=str(runtime),TMPDIR=str(runtime),MPLCONFIGDIR=str(runtime/'matplotlib'),
                      CHD_MINI_TOOL_ROOT=args.tool_root,CHD_MINI_ATTEMPT_LEDGER=str(root/'attempts.json'),CHD_MINI_STAGE='q_base')
    import torch
    from chd_mini.attempts import AttemptLedger
    from chd_segmentation.artifacts.store import file_digest,read_json,write_json
    from chd_segmentation.data.dataset import CaseDataset
    from chd_segmentation.models.q_route1.adapter import QAdapter
    from chd_segmentation.models.workflow import options,train,load_state

    started = perf_counter()
    torch.set_num_threads(2)
    config = options(root/'model.json')
    checkpoint = root/'course/q/base/latest.pt'
    before = file_digest(checkpoint)
    state = train(config,root/'course/prepared.json','base',resume=True,stop_after=2)
    assert state['restored_exact'] and state['update']==2 and file_digest(checkpoint)==before
    resume_seconds = perf_counter()-started
    prepared = read_json(root/'course/prepared.json')
    parallel = verify_parallel(config,prepared)
    bundle = QAdapter(config,prepared,'base',initialize=False)
    saved = load_state(checkpoint)
    bundle.model.load_state_dict(saved['model'])
    del saved
    bundle.model.eval()
    dataset = CaseDataset(prepared,dict(cache_bytes=16*1024**2,reserve_ram_bytes=512*1024**2))
    with torch.inference_mode():
        serial = bundle.validate(dataset,['ct_1003','ct_1012'])
    assert abs(serial-parallel)<2e-5,(serial,parallel)
    assert len(AttemptLedger(root/'attempts.json').read()['attempts'])==2
    assert read_json(root/'course/resources/active.json')==[]
    write_json(report/'final-runtime.json',dict(state='passed',training_updates_added=0,
        total_training_updates=2,backend='gloo',processes=2,restored_exact=True,
        checkpoint_unchanged=True,serial_validation_loss=serial,parallel_validation_loss=parallel,
        resume_seconds=resume_seconds,seconds=perf_counter()-started,gpu_used=False))


def run(args):
    root, report = Path(args.work_root), Path(args.report)
    root.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    runtime = root / 'runtime'
    runtime.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(runtime), TMP=str(runtime), TMPDIR=str(runtime),
                      MPLCONFIGDIR=str(runtime / 'matplotlib'), PYTHONDONTWRITEBYTECODE='1')
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project/'src'), str(project/'tests')]
    tool = Path(args.tool_root).resolve(strict=True)
    sys.path.insert(0, str(tool/'src'))
    from chd_mini.attempts import AttemptLedger
    ledger = AttemptLedger(root/'attempts.json')
    ledger.initialize('chd-q-validation-adaptive-20261120')
    os.environ.update(CHD_MINI_TOOL_ROOT=str(tool),CHD_MINI_ATTEMPT_LEDGER=str(root/'attempts.json'),CHD_MINI_STAGE='q_base')
    import nibabel as nib
    import numpy as np
    import psutil
    import torch
    from monai.inferers import sliding_window_inference
    from chd_segmentation.artifacts.store import digest, read_json, write_json
    from chd_segmentation.config import DEFAULTS, load_config
    from chd_segmentation.preprocessing.prepare import prepare
    from chd_segmentation.models.offline import preprocess_model
    from chd_segmentation.models.workflow import options, train, load_state
    from chd_segmentation.models.q_route1.adapter import QAdapter, allowed
    from chd_segmentation.models.q_route1.losses import segmentation_loss
    from chd_segmentation.models.q_route1.validation import validate_records
    from chd_segmentation.models.q_route1.validation_runtime import assignments, case_demand, divide_budget
    from chd_segmentation.models.q_route1.validation_store import ValidationWorkspace
    from verify_q_validation_real import entry_from_source

    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(20261120)
    manifest = read_json(args.manifest)
    rows = [deepcopy(next(row for row in manifest['cases'] if row['case_id']==name))
            for name in ('ct_1003', 'ct_1012')]
    source_rows = [dict(row, roles=[role]) for row,role in zip(rows, ('train','validation'))]
    write_json(root/'manifest.json', dict(dataset='imagechd', scope='technical', data_root=args.data_root,
                                        cases=source_rows, plan_scope='technical'))
    shape = nib.load(Path(args.data_root)/rows[0]['image']).shape
    roi = [[n//2-16,n//2+16] for n in shape]
    source = deepcopy(DEFAULTS)
    source.update(dataset='imagechd', scope='technical', manifest=str(root/'manifest.json'),
                  pretrained=args.pretrained, work_root=str(root/'course'), run_dir=str(root/'course/source'))
    source['training'].update(patch=[32]*3, batch=4, attempts=2)
    source['technical'].update(roi_xyz=roi, allow_role_overlap=True)
    source['resources'].update(workers=0, ram_fraction=1., reserve_ram_bytes=512*1024**2)
    write_json(root/'source-config.json', source)
    source_path = root/'course/source.json'
    if not source_path.exists():
        prepare(load_config(root/'source-config.json'), source_path)
    model = dict(model='q_route1',technical=True,device='cpu',precision='fp32',updates=2,
        patch=32,local_size=32,run_dir=str(root/'course/q'),work_root=str(root/'course'),
        pretrained=args.pretrained,accumulation=4,validate_every=1,
        technical_roles=dict(base=dict(train=['ct_1003'],validation=['ct_1012'],test=[])),
        execution=dict(cpu_processes=2,workers=0,cache_bytes=16*1024**2,validation_block_bytes=1024**2,
                       reserve_ram_bytes=512*1024**2,ram_fraction=1.,wait_seconds=60),
        logging=dict(heartbeat_seconds=15,case_details=True))
    model_path = root/'model.json'
    write_json(model_path, model)
    prepared_path = root/'course/prepared.json'
    preprocess_started = perf_counter()
    if not prepared_path.exists():
        preprocess_model(model_path, root/'source-config.json', source_path, root/'data', prepared_path)
    preprocess_seconds = perf_counter()-preprocess_started
    config = options(model_path)
    if args.resume_checks_only:
        second = read_json(root/'course/q/base/summary.json')
        training_seconds = None
    else:
        verify_parallel(config, read_json(prepared_path), trained=False)
        training_started = perf_counter()
        first = train(config, prepared_path, 'base', stop_after=1)
        assert first['update']==1
        second = train(config, prepared_path, 'base', resume=True, stop_after=2)
        training_seconds = perf_counter()-training_started
    assert second['update']==2 and second['restored_exact']
    checkpoint = load_state(root/'course/q/base/latest.pt')
    assert len(checkpoint['rank_states'])==2 and checkpoint['batch_layout']==[2,2]
    del checkpoint
    gc.collect()
    events = [json.loads(line) for line in (root/'course/q/base/events.jsonl').read_text(encoding='utf-8').splitlines()]
    updates = [row for row in events if row['event']=='model_optimizer_completed']
    assert [row['update'] for row in updates]==[1,2]
    plans = [row for row in events if row['event']=='q_validation_memory_plan']
    assert len(plans)>=2 and all(row['memory_bytes']<=100_000_000_000 and row['safety_margin']==1.5 for row in plans)
    parallel_loss = verify_parallel(config, read_json(prepared_path))
    later = [json.loads(line) for line in (root/'course/q/base/events.jsonl').read_text(encoding='utf-8').splitlines()]
    active = [row for row in later if row['event']=='q_validation_case_completed'][-2:]
    assert {row['rank'] for row in active}=={0,1}

    numeric_started = perf_counter()
    numeric = []
    for row in rows:
        entry = entry_from_source(row, Path(args.data_root))
        prepared = dict(identity=digest(dict(case=row['case_id'],roi=[64]*3)),dataset='imagechd',scope='technical',
            manifest=dict(cases=[row]),plan=dict(configurations={'3d_fullres':dict(spacing=entry['processed_spacing'])}))
        local = dict(config,run_dir=str(root/'course/numeric'),execution=dict(config['execution']))
        bundle = QAdapter(local, prepared, 'base')
        bundle.model.eval()
        class RealCase:
            def get(self, name):
                assert name==entry['case_id']
                return entry
        with torch.inference_mode():
            logits = sliding_window_inference(torch.from_numpy(entry['image'])[None], [32]*3, 1,
                        lambda value: bundle.model(value)[0], overlap=.5, mode='gaussian')
            support = torch.from_numpy(allowed(entry['label'],entry['unknown_classes'],'imagechd'))[None]
            # 沿用现有概率路径的1e-8下界，比较资源调整前后的同一损失。
            expected = logits.softmax(1).clamp_min(1e-8).log()
            reference = float(segmentation_loss(expected, support, torch.from_numpy(entry['label']>=0)[None,None]))
            demand = case_demand('base',entry['label'].shape,[32]*3,local['execution'])
            bundle.validation_policy = dict(quota_bytes=demand['desired'],demand=demand,rank=0)
            actual = validate_records(bundle, RealCase(), [entry['case_id']])[0]['loss']
            bundle.config['execution']['validation_storage'] = 'disk'
            bundle.config['execution']['validation_max_block_bytes'] = 1024**2
            disk_demand = case_demand('base',entry['label'].shape,[32]*3,bundle.config['execution'])
            bundle.validation_policy = dict(quota_bytes=1024**2,demand=disk_demand,rank=0)
            disk = validate_records(bundle, RealCase(), [entry['case_id']])[0]['loss']
        assert abs(actual-reference)<2e-5 and abs(disk-reference)<2e-5, (reference,actual,disk)
        numeric.append(dict(case_id=entry['case_id'],shape=list(entry['label'].shape),reference_loss=reference,
                            memory_loss=actual,disk_loss=disk,unknown_classes=entry['unknown_classes']))
        del bundle,entry,logits,support,expected
        gc.collect()
    full_shapes = {row['case_id']:list(nib.load(Path(args.data_root)/row['image']).shape) for row in rows}
    split = assignments([row['case_id'] for row in rows], full_shapes, 4)
    assert sum(map(len,split))==2 and len([group for group in split if group])==2
    full = [case_demand('base',shape,[128]*3,dict(validation_block_bytes=64*1024**2)) for shape in full_shapes.values()]
    quotas = divide_budget([row['desired'] for row in full],[row['minimum'] for row in full],50_000_000_000)
    assert all(quota-row['arrays']>512*1024**2 for quota,row in zip(quotas,full))
    with TestCase().assertRaises(InterruptedError):
        with ValidationWorkspace(local,dict(case_id='ct_1012',stage='cancel'),1024**2,
                                 storage='memory',array_budget_bytes=4*64**3) as work:
            work.create('layer',(1,64,64,64),np.float32)
            raise InterruptedError('检查验证取消清理')
    work_root = root/'course/prediction-work'
    assert not work_root.exists() or not list(work_root.iterdir())
    assert read_json(root/'course/resources/active.json')==[]
    assert len(ledger.read()['attempts'])==2 and all(row['outcome']=='successful' for row in ledger.read()['attempts'])
    report_data = dict(state='passed',training_updates=2,replica_processes=2,backend='gloo',gpu_used=False,
        rank_assignments=[row['assignments'] for row in later if row['event']=='q_validation_assignment'],
        parallel_validation_loss=parallel_loss,
        restored_model_optimizer_rank_rng=second['restored_exact'],numeric=numeric,memory_plans=plans,
        original_header_only_shapes=full_shapes,full_shape_planned_quotas=quotas,inactive_ranks_checked=True,
        cancellation_cleaned=True,seconds=perf_counter()-started,preprocess_seconds=preprocess_seconds,
        training_and_resume_seconds=training_seconds,numeric_seconds=perf_counter()-numeric_started,
        peak_rss_bytes=psutil.Process().memory_info().peak_wset,
        scope='two real CT regions; full volumes are header-only; CUDA/NCCL and full-volume throughput unverified')
    write_json(report/'acceptance.json',report_data)
    print(json.dumps(report_data,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('work-root','report','manifest','data-root','pretrained','tool-root'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--resume-checks-only', action='store_true')
    parser.add_argument('--final-checks-only', action='store_true')
    args = parser.parse_args()
    (final_checks if args.final_checks_only else run)(args)
