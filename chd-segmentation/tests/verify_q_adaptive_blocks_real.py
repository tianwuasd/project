import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import psutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
sys.path.insert(0,str(PROJECT/'tests'))

def compare(entry, root, result):
    import numpy as np
    import torch
    from chd_segmentation.acceleration import ResourceSession,ResourceDemand,RuntimeRequest
    from chd_segmentation.acceleration.legacy import SessionGuard
    from chd_segmentation.acceleration.accounting import available
    from chd_segmentation.models.q_route1.training_inputs import (
        input_plan,planned_sections,boundary,pooled_block,target_block,prepare_inputs)

    plan = input_plan(entry['label'].shape,entry['processed_spacing'],2*2**30)
    reference = prepare_inputs(entry,'imagechd',2*2**30)
    resources = dict(device='cpu',cpu_fraction=.8,ram_fraction=.8,reserve_ram_bytes=256*2**20,
        memory_limit_bytes=4*2**30,memory_margin_fraction=.5,max_wait=45,poll_seconds=.2)
    rss = psutil.Process().memory_info().rss
    request = RuntimeRequest(entry['case_id'],'technical',root/'resources',root/'events',resources,
                             ResourceDemand(fixed_bytes=rss+plan['workspace_bytes']+128*2**20))
    with ResourceSession(request) as session, SessionGuard(session.handle) as guard:
        handle = session.handle
        values = [torch.empty_like(value) for value in reference]
        coverage = np.zeros(plan['pooled_shape'],np.uint8)
        iterator = planned_sections(plan['pooled_shape'],plan['halo'],2*2**30,guard)
        records = []
        held = None
        try:
            for index in range(100):
                session._sample()
                if index in (0,2):
                    with handle.condition:
                        free = available(handle.state,protect_pool=True)
                        result['budget_before_hold'] = dict(capacity=handle.state['capacity'],
                            rss=handle.state['task_rss'],free=free,paused=handle.state['paused'])
                    amount = free-plan['minimum_workspace_bytes']-16*2**20
                    assert amount>0,'真实资源余量不足以验证额度变化'
                    held = guard.client.reserve(amount,purpose='adaptive_validation_peer',wait=False)
                    assert held.__enter__() is not None,'真实非等待票据未获准；本例动态检查未执行'
                section,amount,done,total = next(iterator)
                with boundary(guard,amount):
                    x,valid = pooled_block(entry,section,True)
                    heat,mask = target_block(entry,'imagechd',section,plan['halo'])
                    for target,source in zip(values,(x,valid,heat,mask)):
                        target[(0,slice(None),*section)] = source
                    assert not coverage[section].any()
                    coverage[section] += 1
                records.append(dict(block=done,total_estimate=total,bytes=amount,
                                    extent=[s.stop-s.start for s in section],quota_held=held is not None))
                if held is not None:
                    held.__exit__(None,None,None)
                    held = None
                if coverage.all():
                    break
            assert coverage.all()
            result['blocks'] = records
            assert records[1]['bytes']>records[0]['bytes']
            assert records[2]['bytes']<=records[1]['bytes']
            for expected,actual in zip(reference,values):
                torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-6)
            actual = prepare_inputs(entry,'imagechd',2*2**30,guard=guard,log_dir=root/'events')
            for expected,value in zip(reference,actual):
                torch.testing.assert_close(expected,value,atol=1e-6,rtol=1e-6)
            with handle.condition:
                assert handle.state['reserved']==0 and not handle.state['claims']
            result.update(blocks=records,complete_without_overlap=True,inputs_equal=True,
                          final_claims_empty=True,peak_rss_bytes=psutil.Process().memory_info().peak_wset)
        finally:
            if held is not None:
                held.__exit__(None,None,None)


def main():
    import torch
    from verify_q_large_blocks_real import entry_for
    from chd_segmentation.artifacts.store import read_json

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--data-root',required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path('D:/projects/back/t17/chd-q-large-blocks-20261122-adaptive').resolve()
    report = Path('D:/projects/back/sync-reports/chd-q-large-blocks-20261122/adaptive.json')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'.gitignore').write_text('*\n',encoding='utf-8')
    started = perf_counter()
    result = dict(state='running',optimizer_updates=0,cases=[])
    if report.exists():
        previous = read_json(report)
        result['prior_attempts'] = previous.get('prior_attempts',[])+[
            {key:previous[key] for key in ('state','seconds','optimizer_updates','cleanup','error') if key in previous}]
    try:
        manifest = read_json(args.manifest)
        for index,name in enumerate(('ct_1003','ct_1012')):
            row = next(row for row in manifest['cases'] if row['case_id']==name)
            entry,seconds = entry_for(row,Path(args.data_root),index)
            record = dict(case_id=name,read_seconds=seconds)
            result['cases'].append(record)
            began = perf_counter()
            compare(entry,root,record)
            record['seconds']=perf_counter()-began
        result['state']='passed'
    finally:
        if sys.exc_info()[0] is not None:
            result.update(state='failed',error_type=sys.exc_info()[0].__name__,
                          error=str(sys.exc_info()[1]) or sys.exc_info()[0].__name__)
        result['seconds']=perf_counter()-started
        files = [path for path in root.rglob('*') if path.is_file()]
        assert len(files)<=40 and all(root in path.resolve().parents and not path.is_symlink() for path in files)
        amount = sum(path.stat().st_size for path in files)
        assert amount<=256*2**20
        result['cleanup']=dict(files=len(files),bytes=amount,state='pending')
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        shutil.rmtree(root)
        result['cleanup'].update(state='complete',root_absent=not root.exists())
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(state=result['state'],seconds=result['seconds'],report=str(report)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
