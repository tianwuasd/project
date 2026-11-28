import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
from time import perf_counter

ROOT = Path('D:/projects/back/t17/chd-q-local-speed-20261128')
os.environ['CHD_Q_MEMORY_TEST_ROOT'] = str(ROOT)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

NAMES = ['ct_1003','ct_1012']


def dependencies():
    global prepare,bundles,close_errors,ResourceSession,RuntimeRequest,ResourceDemand,SessionGuard
    global CpuReady,OrderedSupply,read_json,write_json,file_digest,DEFAULTS,CaseDataset,execution_config,case_estimates
    global LocalWorkspace,prepare_candidates,validity,bounded_candidates,LayerInference,localizer_forward
    global ValidationWorkspace,AdmittedCases,window_probabilities,np,psutil,torch,dist
    from verify_q_local_memory_real import prepare, bundles, close_errors
    from chd_segmentation.acceleration import ResourceSession, RuntimeRequest, ResourceDemand
    from chd_segmentation.acceleration.legacy import SessionGuard
    from chd_segmentation.acceleration.model_supply import CpuReady, OrderedSupply
    from chd_segmentation.artifacts.store import read_json, write_json, file_digest
    from chd_segmentation.config import DEFAULTS
    from chd_segmentation.data.dataset import CaseDataset
    from chd_segmentation.models.execution import execution_config
    from chd_segmentation.models.memory import case_estimates
    from chd_segmentation.models.q_route1.local_inputs import LocalWorkspace
    from chd_segmentation.models.q_route1.preparation import prepare_candidates
    from chd_segmentation.models.q_route1.validation import validity
    from chd_segmentation.models.q_route1.validation_candidates import bounded_candidates
    from chd_segmentation.models.q_route1.validation_network import LayerInference, localizer_forward
    from chd_segmentation.models.q_route1.validation_store import ValidationWorkspace
    from chd_segmentation.models.supply import AdmittedCases
    from chd_segmentation.prediction.blocks import window_probabilities
    import numpy as np
    import psutil
    import torch
    import torch.distributed as dist


@contextmanager
def measured(result):
    process, stopped = psutil.Process(), threading.Event()
    result['baseline_rss_bytes'] = process.memory_info().rss
    result['peak_rss_bytes'] = result['baseline_rss_bytes']
    cpu, began = process.cpu_times(), perf_counter()
    def sample():
        while not stopped.wait(.025):
            result['peak_rss_bytes'] = max(result['peak_rss_bytes'],process.memory_info().rss)
    monitor = threading.Thread(target=sample)
    monitor.start()
    try:
        yield
    finally:
        stopped.set()
        monitor.join()
        result['seconds'] = perf_counter()-began
        current = process.cpu_times()
        result['cpu_seconds'] = current.user+current.system-cpu.user-cpu.system
        result['average_cpu_cores'] = result['cpu_seconds']/result['seconds']
        result['incremental_peak_bytes'] = result['peak_rss_bytes']-result['baseline_rss_bytes']
        result['vram_peak_bytes'] = None


def common_config(bundle, world=1):
    config = execution_config(bundle.config)
    config['resources'] = dict(DEFAULTS['resources'],**config['resources'])
    config['resources'].update(device='cpu',technical_cpu=True,ram_fraction=1.,cpu_processes=world,cpu_threads=4,
        reserve_ram_bytes=256*2**20,memory_limit_bytes=8*2**30,max_wait=60)
    return config


def request(config, root, world=1, records=()):
    workspace = sum(sorted((r['workspace_bytes'] for r in records),reverse=True)[:world])
    demand = ResourceDemand(fixed_bytes=(2+world)*2**30,workspace_bytes=workspace,
        reader_slots=min(world,len(records)),cpu_slots=world,local_batches=[1]*world)
    return RuntimeRequest(str(root),'local',root/'resources',root/'runtime-log',config['resources'],demand)


def direct(bundle, entry, root, mode):
    config = dict(bundle.config,work_root=str(root),run_dir=str(root/'log'))
    identity = dict(case=entry['case_id'],case_id=entry['case_id'])
    if mode=='baseline':
        work = ValidationWorkspace(config,identity,8*2**20,storage='disk')
    else:
        directory = root/entry['case_id']
        directory.mkdir(parents=True,exist_ok=True)
        work = LocalWorkspace(config,identity,64*2**20,directory=directory,storage='disk')
    with work, torch.no_grad():
        probability = window_probabilities(entry['image'],bundle.base,bundle.spec['patch'],11,
            dict(overlap=.5,blend='gaussian',window_batch=1),bundle.device,work,prefetch=mode!='baseline')
        work.drop(work.maps['counts'])
        valid = validity(entry,work,False)
        logits,features = localizer_forward(bundle.locator,entry['image'],valid,bundle.device,work)
        work.drop(valid)
        heat = LayerInference(work,bundle.device).unary(logits,torch.sigmoid)
        work.drop(logits)
        candidates = bounded_candidates(heat,entry['label'].shape,work,(4,4,8),.2,.8)
        values = torch.from_numpy(np.array(probability,copy=True))[None],torch.from_numpy(np.array(features,copy=True))[None],candidates
    return values


def cases(args, prepared):
    root = ROOT/args.mode
    root.mkdir(exist_ok=True)
    bundle, config = bundles(root,prepared,args.pretrained)
    bundle.config['cpu_preparation'] = dict(enabled=True,max_threads=4)
    dataset = CaseDataset(prepared,dict(cache_bytes=0))
    result = dict(mode=args.mode,cases=[],optimizer_updates=0)
    def run():
        for name in NAMES:
            entry = dataset.get(name)
            row = dict(case_id=name,shape=list(entry['label'].shape))
            with measured(row):
                probability,features,candidates = bundle.frozen(entry) if args.mode=='auto' else direct(bundle,entry,root,args.mode)
                reference = ROOT/'baseline'/name
                reference.mkdir(parents=True,exist_ok=True)
                candidate_records = [asdict(value) for value in candidates]
                if args.mode=='baseline':
                    np.save(reference/'probability.npy',probability.numpy(),allow_pickle=False)
                    np.save(reference/'features.npy',features.numpy(),allow_pickle=False)
                    write_json(reference/'candidates.json',candidate_records)
                else:
                    row['errors'] = dict(probability=close_errors(probability,torch.from_numpy(np.load(reference/'probability.npy'))),
                        features=close_errors(features,torch.from_numpy(np.load(reference/'features.npy'))))
                    expected = read_json(reference/'candidates.json')
                    assert [(list(c['start']),c['source'],c['task']) for c in candidate_records]==[
                        (c['start'],c['source'],c['task']) for c in expected]
                    np.testing.assert_allclose([c['score'] for c in candidate_records],
                        [c['score'] for c in expected],atol=2e-5,rtol=5e-4)
                row['candidates'] = len(candidates)
                del probability,features
            bundle.clear_frozen()
            result['cases'].append(row)
            gc.collect()
        if args.mode=='auto':
            bundle.prepare_training(dataset,NAMES)
            result['records'] = [asdict(value) for value in bundle.sampler.records]
            result['sampler'] = bundle.sampler.state_dict()
            bundle.clear_frozen()
    if args.mode=='auto':
        common = common_config(bundle)
        with ResourceSession(request(common,root)) as session:
            guard = SessionGuard(session.handle)
            guard.config = common
            bundle.training_guard = guard
            run()
            result['capacity_bytes'] = session.capacity
            result['peak_task_rss_bytes'] = session.state['peak_task_rss']
    else:
        run()
    write_json(ROOT/(args.mode+'.json'),result)


def worker(rank, prepared, pretrained, handle, gate, port):
    dependencies()
    root = ROOT/'parallel'/str(rank)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    bundle,_ = bundles(root,prepared,pretrained)
    bundle.config['work_root'] = str(ROOT/'parallel')
    bundle.config['cpu_preparation'] = dict(enabled=True,max_threads=4)
    bundle.training_cache.path = ROOT/'parallel'/'shared'/'local'
    guard = SessionGuard(handle)
    guard.config = common_config(bundle,2)
    bundle.training_guard = guard
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,timeout=timedelta(seconds=180))
    try:
        dataset = AdmittedCases(prepared,dict(cache_bytes=0),rank=rank,gate=gate,device=torch.device('cpu'))
        result = dict(rank=rank,optimizer_updates=0)
        with measured(result):
            prepare_candidates(bundle,dataset,NAMES,rank,2,gate,torch.device('cpu'))
        result['records'] = [asdict(value) for value in bundle.sampler.records]
        assert result['records']==read_json(ROOT/'auto.json')['records']
        bundle.clear_frozen()
        gate.ready.wait(rank,'shared-case-test')
        bundle.training_cache.path = ROOT/'parallel'/'contended'/'local'
        with measured(result.setdefault('shared_case',{})):
            entry = CaseDataset(prepared,dict(cache_bytes=0)).get(NAMES[0])
            value = bundle.frozen(entry)
            assert torch.isfinite(value[0]).all()
            gate.ready.wait(rank,'shared-case-readers')
            bundle.clear_frozen()
        write_json(ROOT/f'rank{rank}.json',result)
    finally:
        dist.destroy_process_group()


def lock_checks(session,root):
    from concurrent.futures import ThreadPoolExecutor
    from unittest import TestCase
    from filelock import Timeout
    from chd_segmentation.models.q_route1.local_cache import BuildLock
    guard = SessionGuard(session.handle)
    path = next((root/'contended'/'local-blocks').glob('*.case')).with_suffix('.lock')
    def waiting(timeout):
        lock = BuildLock(path,timeout=timeout)
        lock.guard = guard
        with lock.acquire(poll_interval=.05):
            return True
    result = {}
    with BuildLock(path), ThreadPoolExecutor(1) as pool:
        began = perf_counter()
        with TestCase().assertRaises(Timeout):
            pool.submit(waiting,.2).result(timeout=3)
        result['explicit_lock_timeout_seconds'] = perf_counter()-began
        future = pool.submit(waiting,-1)
        assert not session.cancel.wait(.2) and not future.done()
        session.cancel.set()
        with TestCase().assertRaises(InterruptedError):
            future.result(timeout=3)
        session.cancel.clear()
        result['lock_cancellation'] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest','data-root','pretrained','report'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('all','prepare','baseline','disk','auto','parallel','locks'),default='all')
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'.gitignore').write_text('*\n!.gitignore\n',encoding='utf-8')
    if args.mode=='all':
        started = perf_counter()
        peaks = dict(files=0,bytes=0,sampling='各子流程退出后清点')
        original = [Path(args.pretrained),*[Path(args.data_root)/(name+suffix) for name in NAMES for suffix in ('_image.nii.gz','_label.nii.gz')]]
        def checksum(path):
            with path.open('rb') as stream:
                return hashlib.file_digest(stream,'sha256').hexdigest()
        hashes = {str(p):checksum(p) for p in original}
        for mode in ('prepare','baseline','disk','auto','parallel'):
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--mode',mode,
                '--manifest',args.manifest,'--data-root',args.data_root,'--pretrained',args.pretrained,'--report',args.report],check=True)
            files = [p for p in ROOT.rglob('*') if p.is_file()]
            peaks['files'] = max(peaks['files'],len(files))
            peaks['bytes'] = max(peaks['bytes'],sum(p.stat().st_size for p in files))
        assert all(checksum(Path(p))==sha for p,sha in hashes.items())
        files = [p for p in ROOT.rglob('*') if p.is_file()]
        result = dict(state='passed',optimizer_updates=0,seconds=perf_counter()-started,
            scope='两例真实CT的原始≤128³区域；CPU；零训练更新',original_hashes=hashes,
            artifact_peaks=peaks,
            original_files_unchanged=True,files=len(files),bytes=sum(p.stat().st_size for p in files),
            results={mode:json.loads((ROOT/(mode+'.json')).read_text(encoding='utf-8')) for mode in ('baseline','disk','auto','parallel')})
        assert max(result['files'],peaks['files'])<=650 and max(result['bytes'],peaks['bytes'])<=4*2**30
        Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(state='passed',seconds=result['seconds'],files=result['files'],bytes=result['bytes'])))
        return
    dependencies()
    torch.set_num_threads(1)
    if args.mode=='prepare':
        args.technical_cpu = True
        write_json(ROOT/'prepared.json',prepare(ROOT,args))
        return
    prepared = read_json(ROOT/'prepared.json')
    if args.mode=='locks':
        common = common_config(bundles(ROOT/'auto',prepared,args.pretrained)[0])
        with ResourceSession(request(common,ROOT/'locks')) as session:
            result = lock_checks(session,ROOT/'parallel')
        write_json(ROOT/'locks.json',result)
        return
    if args.mode!='parallel':
        cases(args,prepared)
        return
    root = ROOT/'parallel'
    common = execution_config(bundles(ROOT/'auto',prepared,args.pretrained)[0].config)
    common['resources'] = dict(DEFAULTS['resources'],**common['resources'])
    common['resources'].update(device='cpu',technical_cpu=True,ram_fraction=1.,cpu_processes=2,cpu_threads=4,
        reserve_ram_bytes=256*2**20,memory_limit_bytes=8*2**30,max_wait=60)
    records = case_estimates(common,prepared,'local')
    context = mp.get_context('spawn')
    with socket.socket() as connection:
        connection.bind(('127.0.0.1',0))
        port = connection.getsockname()[1]
    with ResourceSession(request(common,root,2,records),context=context,layout=dict(workers=[0,0])) as session:
        ready = CpuReady(context,2,root,handle=session.handle)
        gate = OrderedSupply(context,[1,1],2,ready,root,estimates=records,handle=session.handle)
        children = [context.Process(target=worker,args=(rank,prepared,args.pretrained,session.handle,gate,port)) for rank in range(2)]
        for child in children:
            child.start()
        for child in children:
            child.join(300)
            assert child.exitcode==0, (child.pid,child.exitcode)
        result = dict(ranks=[read_json(ROOT/f'rank{rank}.json') for rank in range(2)],
            peak_task_rss_bytes=session.state['peak_task_rss'],capacity_bytes=session.capacity,optimizer_updates=0)
        write_json(ROOT/'parallel.json',result)
        result.update(lock_checks(session,root))
    write_json(ROOT/'parallel.json',result)


if __name__=='__main__':
    main()
