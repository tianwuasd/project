import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest import TestCase

import numpy as np
import psutil
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from verify_q_large_blocks_real import entry_for
from chd_segmentation.artifacts.store import read_json
from chd_segmentation.acceleration import ResourceSession,RuntimeRequest,ResourceDemand,SafeBoundary
from chd_segmentation.acceleration.legacy import SessionGuard
from chd_segmentation.acceleration.accounting import available
from chd_segmentation.models.q_route1.preparation import preparation
from chd_segmentation.models.q_route1.training_inputs import input_plan,prepare_inputs,pooled_image
from chd_segmentation.models.q_route1.adapter import QAdapter,allowed
from chd_segmentation.models.q_route1.geometry import sample_context
from chd_segmentation.models.q_route1.targets import localizer_loss
from chd_segmentation.models.q_route1.losses import segmentation_loss,event_loss
from chd_segmentation.models.workflow import options,save_state,load_state
from chd_segmentation.training.checkpoint import tensors_equal


def run(args,root,result):
    torch.set_num_threads(1)
    torch.manual_seed(20261123)
    manifest = read_json(args.manifest)
    rows = [next(r for r in manifest['cases'] if r['case_id']==n) for n in ('ct_1003','ct_1012')]
    entries = [entry_for(row,Path(args.data_root),i)[0] for i,row in enumerate(rows)]
    plan = read_json(args.plan)
    prepared = dict(dataset='imagechd',identity='real-q-parallel-regions',scope='technical',
        technical=dict(roi_xyz=[64]*3),storage=dict(root=str(root)),plan=plan,
        dataset_json={},manifest=dict(cases=rows))
    roles = {s:dict(train=['ct_1003'],validation=['ct_1012'],test=[]) for s in ('base','localizer','local')}
    raw = dict(model='q_route1',run_dir=str(root/'run'),work_root=str(root),technical=True,
        updates=2,patch=32,local_size=32,technical_roles=roles,pretrained=args.pretrained,
        training_cache=dict(enabled=False),execution=dict(workers=0))
    cfg_path = root/'config.json'
    cfg_path.write_text(json.dumps(raw),encoding='utf-8')
    config = options(cfg_path)
    minimum = max(input_plan(e['label'].shape,e['processed_spacing'],2*2**30)['minimum_workspace_bytes'] for e in entries)
    resources = dict(device='cpu',cpu_fraction=.8,ram_fraction=.8,reserve_ram_bytes=256*2**20,
        memory_limit_bytes=8*2**30,memory_margin_fraction=.5,max_wait=60,poll_seconds=.2)
    request = RuntimeRequest('q-parallel-real','technical',root/'resources',root/'events',resources,
        ResourceDemand(fixed_bytes=psutil.Process().memory_info().rss+2*2**30,cpu_slots=3))
    result['cases'] = []
    with ResourceSession(request) as session,SessionGuard(session.handle) as guard:
        guard.config = dict(resources=resources,scope='technical',work_root=str(root),
                            resource_handle=session.handle)
        parallel_config = dict(config,cpu_preparation=dict(max_threads=2,block_bytes=minimum,
            max_block_bytes=minimum*2,growth_seconds=.01))
        retained = []
        for entry in entries:
            began = perf_counter()
            expected = prepare_inputs(entry,'imagechd',2*2**30)
            serial_seconds = perf_counter()-began
            began = perf_counter()
            with preparation(parallel_config,guard,entry) as runtime:
                actual = prepare_inputs(entry,'imagechd',minimum,guard=guard,parallel=runtime)
                metrics = runtime.executor.metrics()
            parallel_seconds = perf_counter()-began
            for a,b in zip(expected,actual):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
            began = perf_counter()
            with preparation(dict(config,cpu_preparation=dict(max_threads=2)),guard,entry) as runtime:
                auto = prepare_inputs(entry,'imagechd',2*2**30,guard=guard,parallel=runtime)
                auto_metrics = runtime.executor.metrics()
            auto_seconds = perf_counter()-began
            for a,b in zip(expected,auto):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
            with preparation(parallel_config,guard,entry) as runtime:
                pooled = pooled_image(entry,minimum,guard=guard,parallel=runtime)
                supports = allowed(entry['label'],entry['unknown_classes'],'imagechd',runtime)
            for a,b in zip(pooled_image(entry,minimum),pooled):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            np.testing.assert_array_equal(allowed(entry['label'],entry['unknown_classes'],'imagechd'),supports)
            assert metrics['peak_case_threads']>1
            retained.append(actual)
            result['cases'].append(dict(case_id=entry['case_id'],shape=list(entry['label'].shape),
                serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,metrics=metrics,
                auto_seconds=auto_seconds,auto_metrics=auto_metrics))
        if args.preparation_only:
            check_resources(root,session,guard,parallel_config,entries[0],result)
            return
        locator = QAdapter(config,prepared,'localizer')
        optimizer = locator.optimizer()
        logits = locator.model.forward_pooled(*retained[0][:2])['logits']
        loss = localizer_loss(logits,*retained[0][2:])
        loss.backward()
        if not args.zero_updates:
            optimizer.step()
            result['optimizer_updates']+=1
        save_state(root/'locator.pt',dict(model=locator.model.state_dict(),optimizer=optimizer.state_dict()))
        checkpoint = load_state(root/'locator.pt')
        locator.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        assert tensors_equal(optimizer.state_dict(),checkpoint['optimizer'])
        base = QAdapter(config,prepared,'base')
        parents = {s:dict(model=b.model.state_dict(),spec=b.spec,stage=s,prepared=prepared,scope='technical',
            config=config,training_cases=['ct_1003'],initialization_sha256=None,identity=s,best=0.,history=[])
            for s,b in (('base',base),('localizer',locator))}
        local = QAdapter(dict(parallel_config,parents={'base':'in-memory','localizer':'in-memory'}),
                         prepared,'local',parent_states=parents)
        local.training_guard = guard
        probability,features,candidates = local.frozen(entries[0])
        serial_crops = [local.crop(entries[0],c,probability,parallel=False) for c in candidates[:4]]
        crops = []
        with preparation(parallel_config,guard,entries[0]) as runtime:
            runtime.map(range(min(4,len(candidates))),32**3*192,
                lambda i:local.crop(entries[0],candidates[i],probability,parallel=False),
                lambda i,value:crops.append(value))
        for a,b in zip(serial_crops,crops):
            torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
            torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
            np.testing.assert_array_equal(a[2],b[2])
        inputs,editable,support = local.crop(entries[0],candidates[0],probability)
        context = sample_context(features,candidates[0].center,samples=4)
        out = local.model(inputs,context,editable)
        core = (slice(None),slice(None),*out['core'])
        target = torch.from_numpy(support)[None]
        loss = segmentation_loss(out['z0'][core],target[core],editable[core])+event_loss(out['q_logits'],target[core],editable[core])
        assert torch.isfinite(loss)
        optimizer = local.optimizer()
        loss.backward()
        if not args.zero_updates:
            optimizer.step()
            result['optimizer_updates']+=1
        save_state(root/'local.pt',dict(model=local.model.state_dict(),optimizer=optimizer.state_dict()))
        checkpoint = load_state(root/'local.pt')
        optimizer.load_state_dict(checkpoint['optimizer'])
        assert tensors_equal(local.model.state_dict(),checkpoint['model'])
        assert tensors_equal(optimizer.state_dict(),checkpoint['optimizer'])
        result['local_loss'] = float(loss.detach())
        check_resources(root,session,guard,parallel_config,entries[0],result)
    result['peak_rss_bytes'] = getattr(psutil.Process().memory_info(),'peak_wset',None)


def check_resources(root,session,guard,parallel_config,entry,result):
    session._sample()
    with preparation(dict(parallel_config,cpu_preparation=dict(max_threads=2,block_bytes=2*2**30)),guard,entry) as runtime:
        granted = runtime.executor.cpu.request(2)
        base = session.handle.state['cpu_slots']
        assert read_json(root/'resources/active.json')[0]['cpu_slots']==base+max(0,granted-1)
        session.replan(ResourceDemand(fixed_bytes=session.capacity,cpu_slots=base+1),
                       safe_boundary=SafeBoundary('block_group'))
        record = read_json(root/'resources/active.json')[0]
        assert record['model_cpu_claims'] and record['cpu_slots']==base+1+max(0,granted-1)
        result['cpu_base_and_claims_preserved'] = True
        before = runtime.executor._capacity()
        with session.handle.condition:
            free = available(session.handle.state,protect_pool=True)
        amount = free//2
        assert amount>0
        with guard.client.reserve(amount,purpose='real_parallel_quota_hold',wait=False) as ticket:
            assert ticket is not None
            during = runtime.executor._capacity()
        after = runtime.executor._capacity()
        assert during[1]<before[1] and after[1]>=during[1]
        result['dynamic_block_budgets'] = [before,during,after]
    with session.handle.condition:
        assert not session.handle.state['claims'] and session.handle.state['reserved']==0
    registry = read_json(root/'resources/active.json')
    assert not registry[0].get('model_cpu_claims')
    result['cpu_claims_released'] = True
    session.handle.cancel.set()
    with TestCase().assertRaises((MemoryError,InterruptedError)):
        with preparation(parallel_config,guard,entry) as runtime:
            runtime.executor._capacity()
    result['cancel_checked'] = True


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    for name in ('manifest','data-root','plan','pretrained'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--zero-updates',action='store_true')
    parser.add_argument('--preparation-only',action='store_true')
    args = parser.parse_args()
    if args.preparation_only:
        args.zero_updates = True
    root = Path('D:/projects/back/t17/chd-q-cpu-parallel-20261123').resolve()
    report = Path('D:/projects/back/sync-reports/chd-q-cpu-parallel-20261123')/('preparation-report.json' if args.preparation_only else 'zero-update-report.json' if args.zero_updates else 'report.json')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'.gitignore').write_text('*\n',encoding='utf-8')
    report.parent.mkdir(parents=True,exist_ok=True)
    result = dict(state='failed',optimizer_updates=0,gpu_compute=False,scope='两例≤64³；定位器与局部各一次CPU更新')
    if args.zero_updates:
        result['scope']='同两例≤64³，零参数更新；资源及准备路径检查'
    began = perf_counter()
    try:
        run(args,root,result)
        result['state']='passed'
    finally:
        result['seconds']=perf_counter()-began
        result['error']=str(sys.exception()) if sys.exception() else None
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
