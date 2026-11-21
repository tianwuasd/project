import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))


def prepare(args):
    from chd_segmentation.artifacts.store import read_json,write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import prepare as prepare_source
    from chd_segmentation.preprocessing.offline import preprocess

    manifest = read_json(args.manifest)
    rows = []
    for name,role in [('ct_1003','train'),('ct_1012','validation')]:
        row = deepcopy(next(row for row in manifest['cases'] if row['case_id']==name))
        row.update(image=str(Path(args.data_root)/row['image']),label=str(Path(args.data_root)/row['label']),roles=[role])
        rows.append(row)
    manifest.update(cases=rows,dataset='imagechd',scope='technical',plan_scope='technical',
                    data_root=args.data_root,binding_required=False)
    root = Path(args.work)
    course = root/'course'
    write_json(course/'manifest.json',manifest)
    settings = dict(dataset='imagechd',scope='technical',fold=0,manifest=str(course/'manifest.json'),
        pretrained=args.pretrained,work_root=str(course),run_dir=str(course/'prepare'),result_root=str(root/'result'),
        technical=dict(roi_xyz=[[224,288],[224,288],[64,128]],allow_role_overlap=False),
        planning=dict(mode='auto',samples_per_case=1000,max_samples=1000,
                      target_spacing_xyz=[.6,.6,.75]),
        training=dict(patch=[32]*3,batch=2),
        preprocessing=dict(root=str(root/'data'),statistics_source='technical-train'),
        resources=dict(device='cpu',workers=0,cache_bytes=0,max_workers=1,max_readers=1,
                       reserve_ram_bytes=256*2**20,wait_seconds=90),
        logging=dict(heartbeat_seconds=15,case_details=True))
    write_json(course/'prepare.json',settings)
    config = load_config(course/'prepare.json')
    prepare_source(config,course/'source.json')
    prepared = preprocess(config,course/'source.json',course/'prepared.json')
    assert all(max(row['shape_czyx'][1:])<=64 for row in prepared['storage']['cases'].values())
    q = dict(model='q_route1',technical=True,device='cpu',precision='fp32',threads=1,updates=1,
        accumulation=2,patch=32,local_size=32,validate_every=1,pretrained=None,
        technical_roles={'localizer':dict(train=['ct_1003'],validation=['ct_1012'],test=[])},
        work_root=str(course),run_dir=str(course/'run'),
        execution=dict(cpu_processes=2,workers=0,max_workers=1,max_readers=1,cache_bytes=0,prefetch=1,
                       reserve_ram_bytes=256*2**20,wait_seconds=90,validation_block_bytes=64*2**20),
        logging=dict(heartbeat_seconds=15,case_details=True))
    write_json(course/'q.json',q)
    return dict(cases=['ct_1003','ct_1012'],max_source_roi=[64]*3,processed={k:v['shape_czyx']
                for k,v in prepared['storage']['cases'].items()},updates=0)


def compare(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from chd_segmentation.artifacts.store import read_json
    from chd_segmentation.data.dataset import CaseDataset
    from chd_segmentation.models.q_route1.adapter import allowed
    from chd_segmentation.models.q_route1.localizer import Localizer
    from chd_segmentation.models.q_route1.targets import build_heatmap_targets,localizer_loss
    from chd_segmentation.models.q_route1.training_inputs import prepare_inputs,input_plan

    torch.set_num_threads(1)
    torch.manual_seed(20261111)
    dataset = CaseDataset(read_json(Path(args.work)/'course/prepared.json'),dict(cache_bytes=0))
    results = []
    for index,name in enumerate(('ct_1003','ct_1012')):
        entry = dataset.get(name)
        if index:
            shape = (entry['label'].shape[0]-1,*entry['label'].shape[1:])
            region = tuple(slice(0,n) for n in shape)
            entry = dict(entry,image=entry['image'][(slice(None),*region)],label=entry['label'][region])
        mark = perf_counter()
        image,label = torch.from_numpy(np.array(entry['image'],copy=True))[None],entry['label']
        valid = torch.from_numpy(np.array(label>=0,copy=True))[None,None]
        heat,mask = build_heatmap_targets(allowed(label,entry['unknown_classes'],'imagechd'),label>=0,entry['processed_spacing'])
        padding = [v for n in reversed(label.shape) for v in (0,(-n)%32)]
        expected = (F.avg_pool3d(F.pad(image,padding,mode='replicate'),4,4),
                    F.avg_pool3d(F.pad(valid.float(),padding),4,4),heat,mask)
        actual = prepare_inputs(entry,'imagechd',64*2**20)
        errors = []
        for a,b in zip(expected,actual):
            if a.dtype == torch.bool:
                assert torch.equal(a,b)
                errors.append(0.)
            else:
                errors.append(float((a-b).abs().max()))
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        assert mask.any()
        model = Localizer()
        loss = localizer_loss(model(image,valid)['logits'],heat,mask)
        loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        parameters = {p.data_ptr() for p in model.parameters()}
        saved = 0
        def pack(value):
            nonlocal saved
            if value.data_ptr() not in parameters:
                saved += value.numel()*value.element_size()
            return value
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
            other = localizer_loss(model.forward_pooled(*actual[:2])['logits'],*actual[2:])
        other.backward()
        maximum = 0.
        for expected_grad,parameter in zip(gradients,model.parameters()):
            maximum = max(maximum,float((expected_grad-parameter.grad).abs().max()))
            torch.testing.assert_close(expected_grad,parameter.grad,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(loss,other,atol=1e-6,rtol=1e-6)
        estimate = input_plan(label.shape,entry['processed_spacing'],64*2**20)
        assert saved <= estimate['backward_bytes']
        results.append(dict(case_id=name,shape=list(label.shape),input_errors=errors,
                            supervised=int(mask.sum()),target_nonzero=[int((value>0).sum()) for value in heat[0]],
                            reliable_mask=[int(value.sum()) for value in mask[0]],
                            task_max_errors=[float((a-b).abs().max()) for a,b in zip(heat[0],actual[2][0])],
                            loss=float(other.detach()),gradient_max_error=maximum,
                            saved_activation_bytes=saved,backward_budget_bytes=estimate['backward_bytes'],
                            seconds=perf_counter()-mark))
        dataset.clear()
    return dict(cases=results,optimizer_updates=0)


def training(args,resume=False):
    from chd_segmentation.models.workflow import options,train
    from chd_segmentation.artifacts.store import read_json
    root = Path(args.work)/'course'
    tool = Path('D:/projects/project/chd-segmentation-assistant3/chd-mini-workflow')
    sys.path.insert(0,str(tool/'src'))
    from chd_mini.attempts import AttemptLedger
    ledger = Path(args.work)/'attempts.json'
    AttemptLedger(ledger).initialize('q-r2-localizer-20261121')
    if not resume and AttemptLedger(ledger).read()['attempts']:
        raise RuntimeError('本轮全局更新尝试已经领取，禁止追加训练')
    os.environ.update(CHD_MINI_TOOL_ROOT=str(tool),CHD_MINI_ATTEMPT_LEDGER=str(ledger),CHD_MINI_STAGE='q_localizer')
    summary = train(options(root/'q.json'),root/'prepared.json','localizer',resume=resume)
    assert summary['update']==1
    assert (root/'run/localizer/latest.pt').is_file() and (root/'run/localizer/best.pt').is_file()
    if resume:
        assert summary['restored_exact']
    return dict(update=summary['update'],resumed=resume,restored_exact=summary['restored_exact'],
                successful_optimizer_calls=0 if resume else 2,world=summary['world_size'],
                loss=summary['history'][-1]['loss'])


def hold_cpu(root,resources,ready,finished):
    from chd_segmentation.acceleration import ResourceSession,ResourceDemand,RuntimeRequest
    from chd_segmentation.acceleration.host import host_capacity
    request = RuntimeRequest('occupied','technical',Path(root)/'registry',Path(root)/'occupied',resources,
        ResourceDemand(fixed_bytes=128*2**20,cpu_slots=max(1,int(host_capacity()['cores']*.8))))
    with ResourceSession(request):
        ready.set()
        if not finished.wait(5):
            raise TimeoutError('控制测试额度持有到达5秒上限')


def controls(args):
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    import threading
    import multiprocessing as mp
    from time import sleep
    from chd_segmentation.acceleration import ResourceSession,ResourceDemand,RuntimeRequest
    from chd_segmentation.acceleration.host import host_capacity
    from chd_segmentation.acceleration.limits import reserve_margin,raw_capacity
    from chd_segmentation.acceleration.gpu_query import query_nvidia_smi

    root = Path(args.work)/'control'
    p = reserve_margin(3248516930,3401968025,.5)
    assert p['admitted'] and p['margin_bytes']==153451095
    assert not reserve_margin(3248516930,3000000000,.5)['admitted']
    facts = host_capacity()
    r = dict(cpu_fraction=.8,ram_fraction=.8,reserve_ram_bytes=256*2**20,
             memory_limit_bytes=8*2**30,max_wait=2,poll_seconds=.05)
    def execute(name,demand,settings,cancel=None):
        request = RuntimeRequest(name,'technical',root/'registry',root/name,settings,demand)
        with ResourceSession(request,cancel=cancel):
            return True
    with ThreadPoolExecutor(max_workers=1) as pool:
        impossible = pool.submit(execute,'hard',ResourceDemand(fixed_bytes=raw_capacity(r,facts)+1),r)
        assert isinstance(impossible.exception(timeout=5),MemoryError)
        context = mp.get_context('spawn')
        ready,finished = context.Event(),context.Event()
        holder = context.Process(target=hold_cpu,args=(str(root),r,ready,finished))
        holder.start()
        assert ready.wait(5)
        try:
            waited = pool.submit(execute,'limited',ResourceDemand(fixed_bytes=128*2**20),dict(r,max_wait=.3))
            assert isinstance(waited.exception(timeout=5),TimeoutError)
            cancel = threading.Event()
            waited = pool.submit(execute,'cancel',ResourceDemand(fixed_bytes=128*2**20),dict(r,max_wait=None),cancel)
            sleep(.3)
            assert not waited.done()
            cancel.set()
            assert isinstance(waited.exception(timeout=5),InterruptedError)
        finally:
            finished.set()
            holder.join(timeout=5)
        assert holder.exitcode==0
    query_nvidia_smi(['--query-gpu=index','--format=csv,noheader'],log_dir=root/'query')
    with ThreadPoolExecutor(max_workers=1) as pool:
        timed = pool.submit(query_nvidia_smi,['--query-gpu=index','--format=csv,noheader'],
            resources=dict(gpu_query_timeout_seconds=1e-9,gpu_query_retries=2,gpu_query_retry_seconds=.01),
            log_dir=root/'query-timeout')
        assert isinstance(timed.exception(timeout=5),subprocess.TimeoutExpired)
    return dict(margin=p,hard_limit_rejected=True,dynamic_timeout=True,cancelled=True,
                real_gpu_query=True,real_gpu_timeout=True,gpu_compute=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','compare','train','resume','controls'))
    parser.add_argument('--work',required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--pretrained')
    parser.add_argument('--report',required=True)
    args = parser.parse_args()
    root = Path(args.work).resolve()
    if root != Path('D:/projects/back/t17/chd-q-r2-localizer-20261121').resolve():
        raise ValueError('短测目录与批准范围不符')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    began = perf_counter()
    result = (prepare(args) if args.phase=='prepare' else compare(args) if args.phase=='compare' else
              controls(args) if args.phase=='controls' else training(args,args.phase=='resume'))
    from chd_segmentation.artifacts.store import write_json
    write_json(args.report,dict(phase=args.phase,seconds=perf_counter()-began,result=result))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
