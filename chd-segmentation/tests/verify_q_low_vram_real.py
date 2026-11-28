import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from time import perf_counter

import numpy as np
import psutil
import torch
import torch.distributed as dist

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))

from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.acceleration import ResourceSession, RuntimeRequest, ResourceDemand
from chd_segmentation.acceleration.legacy import SessionGuard
from chd_segmentation.acceleration.model_supply import CpuReady, OrderedSupply
from chd_segmentation.models.supply import AdmittedCases
from chd_segmentation.models.q_route1.adapter import QAdapter
from chd_segmentation.models.q_route1.training_retry import accumulate, random_state, restore_random
from chd_segmentation.models.q_route1.training_inputs import prepare_inputs
from chd_segmentation.models.workflow import options, save_state, load_state
from chd_segmentation.training.checkpoint import tensors_equal
from chd_segmentation.config import DEFAULTS


def prepare(root, args):
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import prepare as build
    from chd_segmentation.preprocessing.offline import preprocess
    manifest = read_json(args.manifest)
    rows = []
    for name,role in (('ct_1003','train'),('ct_1012','validation')):
        row = deepcopy(next(r for r in manifest['cases'] if r['case_id']==name))
        row.update(image=str(Path(args.data_root)/row['image']),
                   label=str(Path(args.data_root)/row['label']),roles=[role])
        rows.append(row)
    manifest.update(cases=rows,scope='technical',binding_required=False,data_root=args.data_root)
    write_json(root/'manifest.json',manifest)
    cfg = dict(dataset='imagechd',scope='technical',fold=0,manifest=str(root/'manifest.json'),
        pretrained=args.pretrained,
        work_root=str(root/'course'),run_dir=str(root/'course/prepare'),result_root=str(root/'results'),
        technical=dict(roi_xyz=[[192,320],[192,320],[32,160]],allow_role_overlap=False),
        planning=dict(mode='auto',samples_per_case=1000,max_samples=1000,
                      target_spacing_xyz=[.30469,.30469,.75]),
        preprocessing=dict(root=str(root/'data'),statistics_source='technical-train'),
        training=dict(patch=[32]*3,batch=2),resources=dict(device='cpu',cpu_threads=1,cpu_processes=1,
            workers=0,max_workers=1,max_readers=1,cache_bytes=0,memory_limit_bytes=8*2**30,
            reserve_ram_bytes=256*2**20),logging=dict(heartbeat_seconds=15,case_details=False))
    write_json(root/'prepare-config.json',cfg)
    cfg = load_config(root/'prepare-config.json')
    if getattr(args,'technical_cpu',False):
        cfg['resources']['technical_cpu'] = True
        cfg['resources']['ram_fraction'] = 1.
    build(cfg,root/'course/source.json')
    return preprocess(cfg,root/'course/source.json',root/'course/prepared.json')


def config(root, device='cpu'):
    path = root/'model-config.json'
    write_json(path,dict(model='q_route1',run_dir=str(root/'models'),work_root=str(root),
        device=device,technical=True,updates=2,patch=32,local_size=32,accumulation=2,
        training_cache=dict(enabled=True),execution=dict(workers=0,low_vram_block_bytes=2*2**20),
        cpu_preparation=dict(max_threads=1),logging=dict(heartbeat_seconds=15,case_details=False)))
    return options(path)


def gates(root, handle, shares, records):
    context = torch.multiprocessing.get_context('spawn')
    ready = CpuReady(context,len(shares),root,handle=handle)
    estimates = [dict(case_id=row['case_id'],workspace_bytes=2*2**20) for row in records]
    return OrderedSupply(context,shares,len(shares),ready,root,capacity_bytes=8*2**20,estimates=estimates,handle=handle)


def dataset(prepared, root, gate, rank, device):
    return AdmittedCases(prepared,dict(cache_bytes=0),with_label=True,log_dir=root,
                         rank=rank,gate=gate,device=device)


def errors(expected, model):
    diffs = []
    for reference,parameter in zip(expected,model.parameters()):
        actual = parameter.grad.cpu()
        torch.testing.assert_close(actual,reference,atol=3e-5,rtol=3e-3)
        diffs.append(float((actual-reference).abs().max()))
    return max(diffs)


def cpu_rank(rank, root, prepared, cfg, handle, gate, port, do_update):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=2,rank=rank)
    try:
        device = torch.device('cpu')
        torch.manual_seed(20261124)
        bundle = QAdapter(cfg,prepared,'localizer')
        opt = bundle.optimizer()
        common = dict(resources={**DEFAULTS['resources'],**dict(handle.resources), 'poll_seconds':.1})
        with SessionGuard(handle) as guard:
            guard.config = dict(resources=common['resources'],resource_handle=handle,work_root=str(root))
            bundle.training_guard = guard
            data = dataset(prepared,root,gate,rank,device)
            rng = np.random.default_rng(20261124+rank)
            snapshot = random_state(bundle,rng,device)
            began = perf_counter()
            loss = accumulate(bundle,opt,data,['ct_1003'],rng,0,rank,dict(shares=[1,1]),gate,device,guard,common)
            baseline = [p.grad.clone() for p in bundle.model.parameters()]
            normal_seconds = perf_counter()-began
            restore_random(snapshot,bundle,rng,device)
            bundle.low_vram = True
            began = perf_counter()
            low_loss = accumulate(bundle,opt,data,['ct_1003'],rng,0,rank,dict(shares=[1,1]),gate,device,guard,common)
            difference = errors(baseline,bundle.model)
            assert abs(loss-low_loss)<1e-5
            # 两个rank各一次更新，整轮合计两次；前面的比较只计算梯度。
            from chd_segmentation.models.execution import synchronize_gradients
            synchronize_gradients(bundle.model)
            if do_update:
                write_json(root/f'rank{rank}-attempt.json',dict(attempts=1))
                opt.step()
                write_json(root/f'rank{rank}-updates.json',dict(updates=1))
            saved = dict(model=bundle.model.state_dict(),optimizer=opt.state_dict(),random=random_state(bundle,rng,device))
            save_state(root/f'rank{rank}.pt',saved)
            loaded = load_state(root/f'rank{rank}.pt')
            bundle.model.load_state_dict(loaded['model'])
            opt.load_state_dict(loaded['optimizer'])
            assert tensors_equal(bundle.model.state_dict(),loaded['model'])
            assert tensors_equal(opt.state_dict(),loaded['optimizer'])
            restore_random(loaded['random'],bundle,rng,device)
            restored = random_state(bundle,rng,device)
            a,b = restored.pop('numpy'),deepcopy(loaded['random']).pop('numpy')
            assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
            assert tensors_equal(restored,{k:v for k,v in loaded['random'].items() if k!='numpy'})
            write_json(root/f'rank{rank}.json',dict(updates=int(do_update),normal_seconds=normal_seconds,
                low_seconds=perf_counter()-began,gradient_max_abs_error=difference,loss=low_loss,
                model_optimizer_rng_restored=True,peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None)))
            data.clear()
    finally:
        dist.destroy_process_group()


def gpu(root, prepared, cfg, handle):
    if not torch.cuda.is_available():
        return dict(state='not_measured',reason='no_cuda')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,2*2**30/total),device)
    torch.manual_seed(20261124)
    cfg = dict(cfg,device=str(device),execution=dict(cfg['execution'],vram_reserve_bytes=total))
    bundle = QAdapter(cfg,prepared,'localizer')
    bundle.model.to(device)
    bundle.training_guard = None
    from chd_segmentation.data.dataset import CaseDataset
    data = CaseDataset(prepared,dict(cache_bytes=0))
    result = dict(state='passed',cases=[],optimizer_updates=0)
    for name in ('ct_1003','ct_1012'):
        entry = data.get(name)
        bundle.model.zero_grad(set_to_none=True)
        bundle.low_vram = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        began = perf_counter()
        loss = bundle.loss(entry,np.random.default_rng(1),0)
        loss.backward()
        expected_loss = float(loss.detach())
        expected = [p.grad.cpu().clone() for p in bundle.model.parameters()]
        normal_peak = torch.cuda.max_memory_allocated(device)
        normal_seconds = perf_counter()-began
        del loss
        bundle.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        bundle.low_vram, bundle.low_vram_block_bytes = True, 2*2**20
        began = perf_counter()
        loss = bundle.loss(entry,np.random.default_rng(1),0)
        loss.backward()
        low_loss = float(loss.detach())
        difference = errors(expected,bundle.model)
        assert abs(low_loss-expected_loss)<1e-5
        result['cases'].append(dict(case_id=name,shape=list(entry['label'].shape),
            normal_peak_bytes=normal_peak,low_peak_bytes=torch.cuda.max_memory_allocated(device),
            normal_seconds=normal_seconds,low_seconds=perf_counter()-began,
            gradient_max_abs_error=difference,loss_abs_error=abs(low_loss-expected_loss)))
        del loss,entry,expected
    write_json(root/'gpu-baseline.json',result)
    bundle.model.zero_grad(set_to_none=True)
    data.clear()
    gc.collect()
    torch.cuda.empty_cache()
    cap = math_cap(result['cases'])
    torch.cuda.set_per_process_memory_fraction(cap/total,device)
    bundle.low_vram = False
    gate = gates(root/'gpu',handle,[2],prepared['manifest']['cases'])
    data = dataset(prepared,root/'gpu',gate,0,device)
    rng = np.random.default_rng(20261124)
    port = free_port()
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=1,rank=0)
    try:
        with SessionGuard(handle) as guard:
            guard.config = dict(resources={**DEFAULTS['resources'],**dict(handle.resources)},
                                resource_handle=handle,work_root=str(root))
            bundle.training_guard = guard
            opt = bundle.optimizer()
            value = accumulate(bundle,opt,data,['ct_1003'],rng,0,0,dict(shares=[2]),gate,device,guard,
                               dict(resources=dict(poll_seconds=.1)))
            assert np.isfinite(value) and bundle.low_vram
            events = [json.loads(s) for s in (root/'gpu/events.jsonl').read_text(encoding='utf-8').splitlines()]
            retries = [r for r in events if r['event']=='q_low_vram_retry']
            assert retries and all(r['gradients_discarded'] and r['random_state_replayed'] for r in retries)
            samples = [r['case_id'] for r in events if r['event']=='model_sample_consumed']
            assert samples[:2]==samples[-2:]
            result.update(actual_oom_recovered=True,allocator_limit_bytes=cap,retries=len(retries),
                          sampled_cases_replayed=samples)
            data.clear()
    finally:
        dist.destroy_process_group()
    torch.cuda.set_per_process_memory_fraction(min(1.,2*2**30/total),device)
    return result


def math_cap(rows):
    normal = max(r['normal_peak_bytes'] for r in rows)
    low = max(r['low_peak_bytes'] for r in rows)
    assert low < normal, (normal,low)
    return max(low+8*2**20,((normal+low)//2//(2*2**20))*(2*2**20))


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--gpu-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_LIBUV='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    root = Path('D:/projects/back/t17/chd-q-low-vram-20261124')
    report = Path('D:/projects/back/sync-reports/chd-q-low-vram-20261124/report.json')
    if report.exists():
        report = report.with_name(f'report-{len(list(report.parent.glob("report*.json")))+1}.json')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'.gitignore').write_text('*\n',encoding='utf-8')
    report.parent.mkdir(parents=True,exist_ok=True)
    result = dict(state='failed',optimizer_updates=0,
                  scope='两例原始≤128³区域；实际CPU进程与更新数见分项，CUDA零更新',gpu_only=args.gpu_only)
    began = perf_counter()
    try:
        mark = perf_counter()
        prepared = prepare(root,args)
        result['preprocessing_seconds'] = perf_counter()-mark
        cfg = config(root)
        resources = dict(device='cpu',cpu_fraction=.8,ram_fraction=.8,reserve_ram_bytes=256*2**20,
            memory_limit_bytes=8*2**30,max_wait=120,poll_seconds=.1,memory_margin_fraction=.5)
        request = RuntimeRequest('q-low-vram','technical',root/'resources',root/'events',resources,
                                 ResourceDemand(fixed_bytes=4*2**30,cpu_slots=2,
                                     workspace_bytes=8*2**20,reader_slots=2,local_batches=[1,1]))
        consumed = sum(read_json(p).get('optimizer_attempts',0) for p in report.parent.glob('report*.json'))
        assert consumed in (0,2)
        if not args.gpu_only:
            with ResourceSession(request) as session:
                gate = gates(root/'cpu',session.handle,[1,1],prepared['manifest']['cases'])
                torch.multiprocessing.spawn(cpu_rank,args=(root,prepared,cfg,session.handle,gate,free_port(),consumed==0),nprocs=2)
                result['cpu_ranks'] = [read_json(root/f'rank{i}.json') for i in range(2)]
        result['previous_optimizer_attempts'] = consumed
        gpu_request = RuntimeRequest('q-low-vram-gpu','technical',root/'resources',root/'gpu-events',resources,
            ResourceDemand(fixed_bytes=2*2**30,cpu_slots=1,workspace_bytes=8*2**20,reader_slots=1,local_batches=[2]))
        with ResourceSession(gpu_request) as session:
            result['gpu'] = gpu(root,prepared,cfg,session.handle)
        result['state'] = 'passed'
    finally:
        result['optimizer_updates'] = sum(read_json(p)['updates'] for p in root.glob('rank*-updates.json'))
        result['optimizer_attempts'] = sum(read_json(p)['attempts'] for p in root.glob('rank*-attempt.json'))
        result.update(seconds=perf_counter()-began,error=repr(sys.exception()) if sys.exception() else None,
                      peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None))
        files = [p for p in root.rglob('*') if p.is_file()]
        assert len(files)<=100 and all(root.resolve() in p.resolve().parents and not p.is_symlink() for p in files)
        amount = sum(p.stat().st_size for p in files)
        assert amount<=2*2**30
        result['cleanup'] = dict(files=len(files),bytes=amount,root=str(root))
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        gc.collect()
        shutil.rmtree(root)
        result['cleanup']['root_absent'] = not root.exists()
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(state=result['state'],seconds=result['seconds'],error=result['error'])),flush=True)


if __name__=='__main__':
    main()
