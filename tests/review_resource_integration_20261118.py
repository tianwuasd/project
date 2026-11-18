import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter, sleep

import torch


def collate_samples(samples):
    return samples


def check(args):
    from review_models_20261118 import setup
    engine, root, reports = setup(args)
    from chd_segmentation.acceleration import attach, SafeBoundary, HostCoordinator
    from chd_segmentation.artifacts.store import read_json, write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import load_prepared
    from chd_segmentation.training.session import admitted_training, supply_demand, supply_layout
    from chd_segmentation.training.batches import BatchSupply
    from chd_segmentation.training.build_gate import BuildGate
    from chd_segmentation.training.engine import rebalance_supply
    torch.set_num_threads(1)
    began = perf_counter()
    config = load_config(root/'course/stu.json')
    config['run_dir'] = str(root/'checks/resource-integration')
    config['resources'].update(workers=1, max_workers=1, prefetch=1, wait_seconds=15)
    prepared = load_prepared(root/'course/shared-offline.json')
    evidence = dict(cases=['ct_1001','ct_1012'],optimizer_updates=0)
    context = mp.get_context('spawn')
    with admitted_training(config,prepared,{'review':'data-only'}) as (allocation,session):
        handle = allocation['resource_handle']
        client = attach(handle)
        from chd_segmentation.training.budget import cpu_plan
        from chd_segmentation.acceleration.host import host_capacity
        waiting_config = deepcopy(config)
        waiting_config['resources']['cpu_fraction'] = 1/host_capacity()['cores']
        occupied = read_json(Path(config['work_root'])/'resources/active.json')
        waiting = cpu_plan(waiting_config,prepared,1,peers=occupied)
        if waiting['admitted'] or waiting['available_cpu_slots']!=0:
            raise AssertionError('真实已登记CPU占用未形成等待需求')
        waiting_demand = supply_demand(waiting,1,waiting_config['resources'],[1])
        if waiting_demand.memory_bytes<1 or supply_layout(waiting)['workers']!=[0]:
            raise AssertionError('资源不足的等待需求缺少合法预算或布局')
        evidence['cpu_waiting_plan'] = dict(available_cpu_slots=waiting['available_cpu_slots'],
            minimum_bytes=waiting_demand.memory_bytes,layout=supply_layout(waiting),host=waiting['facts'])
        gate = BuildGate(context,allocation['supply']['reader_slots'],
            allocation['supply']['workspace_capacity_bytes'],allocation['supply']['case_estimates'])
        supply = BatchSupply(prepared,config,[0],1,0,gate,allocation['owner'],handle=handle)
        with supply:
            supply.submit(['ct_1001'],0,0)
            warmup = supply.take_cpu()
            supply.mark_consumed()
            del warmup
            amount = gate.estimates['ct_1001']
            with client.case(amount,purpose='real_reader_contention'):
                supply.submit(['ct_1012'],1,0)
                sleep(.35)
                if supply.queue[0][1].done():
                    raise AssertionError('真实病例额度占用期间worker没有等待')
            batch = supply.take_cpu()
            if batch['case_ids']!=['ct_1012'] or not torch.isfinite(batch['image']).all():
                raise AssertionError('真实worker未交付所选病例')
            supply.mark_consumed()
            workers = [dict(row) for pid,row in handle.members.items()
                       if pid not in (os.getpid(),session.manager._process.pid)]
            if not workers:
                raise AssertionError('真实供数worker没有登记到公共成员表')
            evidence['stu_worker'] = dict(members=workers,case_bytes=amount,
                image_shape=list(batch['image'].shape),reader_contention_wait_verified=True)
            del batch
            config['resources']['workers'] = 0
            job = dict(allocation,config=config,prepared=prepared,shares=[1],
                coordinator=HostCoordinator(context,1),control_batch=1)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(rebalance_supply,job,supply,gate,torch.device('cpu'),0,handle.cancel)
                while not future.done():
                    session.service_replans()
                    sleep(.02)
                if future.result()!='applied':
                    raise AssertionError('真实供数重规划没有应用')
            if handle.state['layout']['workers']!=[0] or handle.state['plan_generation']<1:
                raise AssertionError('供数重规划未更新同一公共句柄')
            supply.submit(['ct_1012'],1,0)
            batch = supply.take_cpu()
            if batch['case_ids']!=['ct_1012']:
                raise AssertionError('重规划后未交付真实验证病例')
            supply.mark_consumed()
            evidence['stu_replan'] = dict(layout=dict(handle.state['layout']),
                generation=handle.state['plan_generation'],validation_shape=list(batch['image'].shape))
            del batch
        from chd_segmentation.models.workflow import options
        from chd_segmentation.models.medsam2.official import effective_config
        from chd_segmentation.models.memory import case_estimates
        med = options(root/'course/med.json')
        med_source = root/'course/med-offline.json'
        med_prepared = load_prepared(med_source)
        recipe = effective_config(med,med_prepared,med_source,Path(config['run_dir']),['ct_1001'])
        estimates = case_estimates(med,med_prepared,'sequence')
        plan = deepcopy(job['supply'])
        workspace = max(row['workspace_bytes'] for row in estimates)
        plan['ram_bytes'] += workspace-plan['workspace_capacity_bytes']
        plan.update(workspace_capacity_bytes=workspace,reader_slots=1,workers=[1],cpu_slots=2)
        session.replan(supply_demand(plan,1,config['resources'],[1]),
            safe_boundary=SafeBoundary('before_case'),layout=supply_layout(plan))
        from hydra.utils import instantiate
        from chd_segmentation.models.supply import CpuReady,OrderedSupply
        from chd_segmentation.models.medsam2.data import BudgetDataLoader,ResourceWorkerInit,set_read_gate
        ready = CpuReady(context,1,config['run_dir'],owner=allocation['owner'],handle=handle)
        med_gate = OrderedSupply(context,[1],1,ready,config['run_dir'],capacity_bytes=workspace,
            estimates=estimates,unit='sequence_batch',handle=handle)
        med_gate.start(0,'real_med_loader',[1])
        set_read_gate(med_gate,1)
        leaf = recipe['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]
        dataset = instantiate(leaf)
        os.environ['PYTHONPATH'] = os.pathsep.join([str(engine/'src'),med['official_root'],med['dependency_root']])
        loader = BudgetDataLoader(dataset,num_workers=1,multiprocessing_context=context,
            batch_sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dataset),1,False),
            collate_fn=collate_samples,worker_init_fn=ResourceWorkerInit({'resource_handle':handle},allocation['owner'],med_gate))
        iterator = iter(loader)
        samples = next(iterator)
        med_gate.mark_ready(0)
        med_gate.deliver(0)
        sample = samples[0]
        med_members = [dict(row) for pid,row in handle.members.items()
                       if pid not in (os.getpid(),session.manager._process.pid)
                       and pid not in [row['pid'] for row in workers]]
        if len(sample.frames)!=8 or not med_members or not all(frame.objects for frame in sample.frames):
            raise AssertionError('官方真实DataLoader序列或worker登记不完整')
        evidence['med_loader'] = dict(frames=len(sample.frames),frame_shape=list(sample.frames[0].data.shape),
            objects_per_frame=[len(frame.objects) for frame in sample.frames],members=med_members,
            workspace_bytes=workspace)
        med_gate.finish(0,'real_med_loader')
        if list(iterator):
            raise AssertionError('本轮Med DataLoader处理超过一个真实样本')
        del iterator,loader,samples,sample,dataset
        from chd_segmentation.models.cases import ModelCases
        q_source = load_prepared(root/'course/q-wide-offline.json')
        q = options(root/'course/q-local.json')
        q_estimates = case_estimates(q,q_source,'local')
        workspace = max(workspace,max(row['workspace_bytes'] for row in q_estimates))
        plan['ram_bytes'] += workspace-plan['workspace_capacity_bytes']
        plan.update(workspace_capacity_bytes=workspace,workers=[0],cpu_slots=1)
        session.replan(supply_demand(plan,1,config['resources'],[1]),
            safe_boundary=SafeBoundary('batch_consumed'),layout=supply_layout(plan))
        q_gate = OrderedSupply(context,[1],1,ready,config['run_dir'],capacity_bytes=workspace,
            estimates=q_estimates,unit='case_entry',handle=handle)
        q_gate.start(0,'real_q_read',[1])
        with q_gate.build(0,0,0):
            q_gate.reserve(0,'ct_1012')
            entry = ModelCases(q_source,q,with_label=True).get('ct_1012')
            evidence['q_read'] = dict(shape=list(entry['image'].shape),
                workspace_reserved=handle.state['workspace_reserved'],active_readers=handle.state['active_readers'])
        q_gate.mark_ready(0)
        q_gate.deliver(0)
        q_gate.finish(0,'real_q_read')
        q_gate.start(0,'cancelled_q_read',[1])
        def cancelled_build():
            with q_gate.build(0,0,0):
                q_gate.reserve(0,'ct_1001')
                return ModelCases(q_source,q,with_label=True).get('ct_1001')
        with ThreadPoolExecutor(max_workers=1) as pool:
            with client.case(workspace,purpose='real_cancel_contention'):
                future = pool.submit(cancelled_build)
                sleep(.25)
                if future.done():
                    raise AssertionError('共享额度占用期间Q没有等待')
                handle.cancel.set()
                failure = future.exception(timeout=5)
                if not isinstance(failure,InterruptedError):
                    raise AssertionError('Q/Med同句柄取消没有传递')
        evidence['shared_control'] = dict(q_handle_same=q_gate.handle is handle,
            med_handle_same=med_gate.handle is handle,cancel_shared=ready.cancel is handle.cancel,
            cancellation_exception=type(failure).__name__,active_readers=handle.state['active_readers'],
            workspace_reserved=handle.state['workspace_reserved'])
        if handle.state['active_readers'] or handle.state['workspace_reserved']:
            raise AssertionError('检查结束仍持有病例额度')
        evidence['peak_task_rss_bytes'] = handle.state['peak_task_rss']
    import psutil
    observed = workers+med_members
    live = [row for row in observed if psutil.pid_exists(row['pid']) and
            psutil.Process(row['pid']).create_time()==row['birth']]
    if live:
        raise AssertionError('真实供数检查结束后仍有worker存活')
    evidence.update(state='completed',seconds=perf_counter()-began,all_workers_exited=True)
    write_json(reports/'resource-integration.json',evidence)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',required=True)
    parser.add_argument('--work',required=True)
    parser.add_argument('--reports',required=True)
    check(parser.parse_args())
