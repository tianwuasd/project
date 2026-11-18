from datetime import timedelta
import os
from pathlib import Path
from time import perf_counter


def check(args):
    if not args.execute_cuda:
        raise ValueError('CUDA检查需要单独获准并显式传入--execute-cuda')
    from review_models_20261118 import setup
    engine,root,reports = setup(args)
    import torch
    import torch.distributed as dist
    from chd_segmentation.acceleration import ResourceSession,RuntimeRequest,ResourceDemand,attach
    from chd_segmentation.acceleration.distributed import distributed_backend
    from chd_segmentation.artifacts.store import read_json,write_json
    from chd_segmentation.models.workflow import options
    from chd_segmentation.models.execution import execution_config,verify_device,rank_measurement
    if torch.cuda.is_initialized():
        raise RuntimeError('GPU准入前CUDA已初始化')
    config = options(root/'course/med-gpu.json')
    common = execution_config(config)
    directory = root/'checks/cuda-collective'
    directory.mkdir(parents=True,exist_ok=False)
    resources = dict(common['resources'],precision='fp32',max_wait=30,
        gpu_spec=dict(resource_model='medsam2_cuda_control',stage='control'))
    request = RuntimeRequest('assistant2-cuda-collective','control',root/'resources',str(directory),resources,
        ResourceDemand(fixed_bytes=2*2**30,cpu_slots=1,gpu_count=1,gpu_ids=resources['gpu_ids'],local_batches=[1]))
    began = perf_counter()
    store = directory/'gloo-store'
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        with ResourceSession(request,layout=dict(workers=[0],cache_bytes=[0],prefetch=1)) as session:
            os.environ['CUDA_VISIBLE_DEVICES'] = session.allocation.gpus[0]['uuid']
            device = torch.device('cuda',0)
            backend = distributed_backend(device,'gloo')
            verify_device(device,dict(gpus=session.allocation.gpus),0)
            values = read_json(root/'course/med-offline.json')['storage']['cases']
            expected = torch.tensor([values[name]['array_bytes'] for name in ('ct_1001','ct_1012')],dtype=torch.float64)
            try:
                dist.init_process_group(backend,init_method=store.as_uri(),rank=0,world_size=1,
                    timeout=timedelta(seconds=30))
                with attach(session.handle).reserve(expected.numel()*expected.element_size(),purpose='cuda_control'):
                    actual = expected.to(device)
                    dist.all_reduce(actual)
                    torch.cuda.synchronize(device)
                    if not torch.equal(actual.cpu(),expected):
                        raise AssertionError('真实CUDA/Gloo collective结果不同')
                    measurement = rank_measurement(device,dict(gpus=session.allocation.gpus),0)
                    del actual
                write_json(reports/'cuda-collective.json',dict(state='completed',seconds=perf_counter()-began,
                    backend=backend,world=1,gpus=session.allocation.gpus,measurement=measurement,
                    optimizer_updates=0,model_initialized=False,capacity_evidence_written=False,
                    scope='单卡CUDA/Gloo最小通信与显存可观测性，不证明完整模型容量'))
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
    finally:
        removed = []
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous
        if store.exists():
            if store.is_symlink() or store.resolve().parent!=directory.resolve():
                raise ValueError('CUDA rendezvous不属于本轮检查')
            removed.append(dict(path=str(store.resolve()),bytes=store.stat().st_size))
            store.unlink()
        for path in directory.iterdir():
            if path.is_symlink() or path.name not in ('events.jsonl','events.lock'):
                raise ValueError('CUDA检查目录包含未知产物')
            removed.append(dict(path=str(path.resolve()),bytes=path.stat().st_size))
            path.unlink()
        directory.rmdir()
        write_json(reports/'cuda-collective-cleanup.json',dict(files=len(removed),
            released_bytes=sum(row['bytes'] for row in removed),manifest=removed,directory_absent=not directory.exists()))
