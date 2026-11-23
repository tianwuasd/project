import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest import TestCase

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))

from verify_q_large_blocks_real import entry_for
from chd_segmentation.artifacts.store import read_json
from chd_segmentation.models.workflow import options,load_state,save_state
from chd_segmentation.models.contracts import require_resume_contract
from chd_segmentation.models.q_route1.adapter import QAdapter
from chd_segmentation.models.q_route1.training_inputs import prepare_inputs
from chd_segmentation.models.q_route1.training_cache import TrainingCache
from chd_segmentation.training.checkpoint import tensors_equal
from chd_segmentation.training.stopping import resolve_budget


def worker(rank,root,config,prepared,entries):
    torch.set_num_threads(2)
    torch.manual_seed(20261123)
    dist.init_process_group('gloo',init_method=(root/'gloo').as_uri(),rank=rank,world_size=2)
    try:
        bundle = QAdapter(config,prepared,'localizer')
        optimizer = bundle.optimizer()
        began = perf_counter()
        loss = bundle.loss(entries[0],np.random.default_rng(rank),0)
        loss.backward()
        for parameter in bundle.model.parameters():
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(2)
        optimizer.step()
        state = dict(model=bundle.model.state_dict(),optimizer=optimizer.state_dict(),update=1,
                     rng=torch.get_rng_state())
        target = root/f'rank{rank}.pt'
        save_state(target,state)
        restored = load_state(target)
        bundle.model.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
        assert tensors_equal(bundle.model.state_dict(),state['model'])
        assert tensors_equal(optimizer.state_dict(),state['optimizer'])
        (root/f'rank{rank}.json').write_text(json.dumps(dict(updates=1,loss=float(loss.detach()),
            seconds=perf_counter()-began,rss_bytes=psutil.Process().memory_info().rss)),encoding='utf-8')
    finally:
        dist.destroy_process_group()


def run(args,root,result):
    torch.set_num_threads(2)
    manifest = read_json(args.manifest)
    rows = [next(row for row in manifest['cases'] if row['case_id']==name) for name in ('ct_1003','ct_1012')]
    entries = [entry_for(row,Path(args.data_root),index)[0] for index,row in enumerate(rows)]
    for entry in entries:
        entry['properties'] = dict(class_locations={int(label):np.column_stack((
            np.zeros(len(points),dtype=np.int64),points)) for label in np.unique(entry['label'])
            if label>0 for points in [np.argwhere(entry['label']==label)[:32]]})
    prepared = dict(dataset='imagechd',identity='real-imagechd-regions-20261123',scope='technical',
        technical=dict(roi_xyz=[64,64,64]),storage=dict(root=str(root/'source-regions')),
        plan=read_json(args.plan),dataset_json=dict(labels=manifest.get('labels',{})),manifest=dict(cases=rows))
    roles = {stage:dict(train=['ct_1003'],validation=['ct_1012'],test=[])
             for stage in ('base','localizer','local')}
    raw = dict(model='q_route1',run_dir=str(root/'run'),work_root=str(root),technical=True,
        updates=2,patch=32,local_size=32,technical_roles=roles,pretrained=args.pretrained,
        execution=dict(workers=0),logging=dict(heartbeat_seconds=0,case_details=True))
    path = root/'config.json'
    path.write_text(json.dumps(raw),encoding='utf-8')
    config = options(path)
    assert resolve_budget(dict(epochs=5,updates=None),'q_route1','localizer')['updates']==1250
    assert resolve_budget(dict(epochs=20,updates=None),'q_route1','local')['updates']==5000
    assert resolve_budget(dict(epochs=None,updates=None),'q_route1','localizer')['epochs']==5
    assert options(path,dict(epochs=1,technical=False,patch=128,local_size=128))['updates'] is None
    old = dict(config,updates=5000,epochs=None,updates_per_epoch=None)
    new = dict(config,updates=1250,epochs=5,updates_per_epoch=250,resume_budget=True,validate_every=250)
    assert not require_resume_contract(old,prepared,new,prepared,'localizer')
    with TestCase().assertRaises(ValueError):
        require_resume_contract(old,prepared,dict(new,learning_rate=.03),prepared,'localizer')
    result['cache'] = []
    cache = TrainingCache(config,prepared,'localizer')
    for entry in entries:
        estimate = int(np.prod([(n+31)//32*8 for n in entry['label'].shape])*23)+65536
        began = perf_counter()
        first = cache.get(entry,lambda:prepare_inputs(entry,'imagechd',2*2**30),estimate)
        cold = perf_counter()-began
        began = perf_counter()
        second = cache.get(entry,lambda:prepare_inputs(entry,'imagechd',2*2**30),estimate)
        warm = perf_counter()-began
        for a,b in zip(first,second):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        result['cache'].append(dict(case_id=entry['case_id'],cold_seconds=cold,warm_seconds=warm))
    assert len(list(cache.path.glob('*.pt')))==2
    mp.spawn(worker,args=(root,config,prepared,entries),nprocs=2,join=True)
    a,b = [load_state(root/f'rank{rank}.pt') for rank in range(2)]
    assert tensors_equal(a['model'],b['model'])
    result['ranks'] = [read_json(root/f'rank{rank}.json') for rank in range(2)]
    result['optimizer_updates'] = sum(row['updates'] for row in result['ranks'])
    locator = QAdapter(config,prepared,'localizer')
    locator.model.load_state_dict(a['model'])
    base = QAdapter(config,prepared,'base')
    parents = {stage:dict(model=bundle.model.state_dict(),spec=bundle.spec,stage=stage,
        prepared=prepared,scope='technical',config=config,training_cases=['ct_1003'],
        initialization_sha256=None,identity=stage,best=0.,history=[])
        for stage,bundle in (('base',base),('localizer',locator))}
    local_config = dict(config,parents=dict(base='in-memory-base',localizer='in-memory-localizer'))
    local = QAdapter(local_config,prepared,'local',parent_states=parents)
    del a,b,base,locator
    gc.collect()
    outputs = []
    began = perf_counter()
    for entry in entries:
        probability,features,candidates = local.frozen(entry)
        outputs.append(dict(probability=probability.clone(),features=features.clone(),
                            candidates=[asdict(value) for value in candidates]))
    result['frozen_cold_seconds'] = perf_counter()-began
    began = perf_counter()
    for entry,expected in zip(entries,outputs):
        probability,features,candidates = local.frozen(entry)
        torch.testing.assert_close(probability,expected['probability'],atol=0,rtol=0)
        torch.testing.assert_close(features,expected['features'],atol=0,rtol=0)
        assert [asdict(value) for value in candidates]==expected['candidates']
    result['frozen_warm_seconds'] = perf_counter()-began
    assert len(list(local.training_cache.path.glob('*.pt')))==2
    assert TrainingCache(dict(config,input_source={'intensity_mode':'source_window'}),prepared,
                         'localizer').identity!=cache.identity
    assert TrainingCache(config,prepared,'local',parents='different').identity!=local.training_cache.identity
    result['cases'] = [dict(case_id=e['case_id'],shape=list(e['label'].shape)) for e in entries]
    result['rss_bytes'] = psutil.Process().memory_info().rss


def boundaries(args,root,result):
    torch.set_num_threads(2)
    manifest = read_json(args.manifest)
    rows = [next(r for r in manifest['cases'] if r['case_id']==name) for name in ('ct_1003','ct_1012')]
    entries = [entry_for(row,Path(args.data_root),i)[0] for i,row in enumerate(rows)]
    prepared = dict(dataset='imagechd',identity='real-regions-boundaries',technical=dict(roi_xyz=[64]*3),
                    storage=dict(root=str(root)),plan=read_json(args.plan))
    config = dict(model='q_route1',method='M07',patch=32,local_size=32,run_dir=str(root/'run'),work_root=str(root))
    cache = TrainingCache(config,prepared,'localizer')
    intervals = {}
    def task(entry):
        def produce():
            began = perf_counter()
            value = prepare_inputs(entry,'imagechd',2*2**30)
            intervals[entry['case_id']] = (began,perf_counter())
            return value
        return cache.get(entry,produce,200000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(task,entries))
    result['parallel_intervals'] = intervals
    assert max(t[0] for t in intervals.values())<min(t[1] for t in intervals.values())
    assert not list(cache.path.glob('*.pending')) and not list(cache.path.glob('*.reserve.json'))
    small = TrainingCache(dict(config,work_root=str(root/'bounded'),
        training_cache=dict(localizer_bytes=200000)),prepared,'localizer')
    for entry,value in zip(entries,values):
        small.get(entry,lambda:prepare_inputs(entry,'imagechd',2*2**30),200000)
    files = list(small.path.glob('*.pt'))
    assert len(files)==1 and sum(p.stat().st_size for p in files)<=small.limit
    oversized = TrainingCache(dict(config,work_root=str(root/'oversized'),
        training_cache=dict(localizer_bytes=100)),prepared,'localizer')
    oversized.get(entries[0],lambda:prepare_inputs(entries[0],'imagechd',2*2**30),200000)
    assert not oversized.path.exists()
    cfg = dict(config,technical=True,device='cpu',seed=20261123,parents={},learning_rate=.0002,
               pretrained=None,updates=2,execution={},logging=dict(heartbeat_seconds=0,case_details=True))
    bundle = QAdapter(cfg,prepared,'localizer')
    optimizer = bundle.optimizer()
    bundle.schedule(optimizer,50)
    save_state(root/'schedule.pt',dict(optimizer=optimizer.state_dict()))
    restored = bundle.optimizer()
    restored.load_state_dict(load_state(root/'schedule.pt')['optimizer'])
    bundle.config = dict(cfg,technical=False,updates=1250)
    bundle.schedule(restored,51)
    assert all(g['peak_lr']==.0002 and np.isclose(g['lr'],.0002*52/500) for g in restored.param_groups)
    bundle.schedule(restored,700)
    expected = .0002*(.01+.99*(1+np.cos(np.pi*200/750))/2)
    assert all(np.isclose(g['lr'],expected) for g in restored.param_groups)
    result.update(optimizer_updates=0,capacity_bytes=small.limit,retained_cache_files=len(files),
                  budget_schedule_checked=True,peak_rss_bytes=None,peak_vram_bytes=None)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    for name in ('manifest','data-root','plan','pretrained'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--cache-only',action='store_true')
    args = parser.parse_args()
    root = Path('D:/projects/back/t17/chd-q-epochs-cache-20261123').resolve()
    report = Path('D:/projects/back/sync-reports/chd-q-epochs-cache-20261123')/('boundaries.json' if args.cache_only else 'report.json')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'.gitignore').write_text('*\n',encoding='utf-8')
    report.parent.mkdir(parents=True,exist_ok=True)
    result = dict(state='failed',optimizer_updates=0,gpu_compute=False,
        scope='两例真实区域≤64³；双CPU/Gloo合计两次参数更新；冻结上游零更新')
    if args.cache_only:
        result['scope']='同两例≤64³；零更新缓存并发/容量/日程数值检查'
    began = perf_counter()
    try:
        if args.cache_only:
            boundaries(args,root,result)
        else:
            run(args,root,result)
        result['state']='passed'
    finally:
        result['seconds']=perf_counter()-began
        files = [p for p in root.rglob('*') if p.is_file()]
        assert len(files)<=100 and all(root in p.resolve().parents and not p.is_symlink() for p in files)
        amount = sum(p.stat().st_size for p in files)
        assert amount<=2*2**30
        result['cleanup']=dict(files=len(files),bytes=amount,root=str(root),state='pending')
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        shutil.rmtree(root)
        result['cleanup'].update(state='complete',root_absent=not root.exists())
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
