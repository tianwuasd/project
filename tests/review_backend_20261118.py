from datetime import timedelta
import os
from pathlib import Path
from time import perf_counter


def check(args):
    from review_models_20261118 import setup
    engine,root,reports = setup(args)
    import torch
    import torch.distributed as dist
    from chd_segmentation.acceleration.distributed import distributed_backend
    from chd_segmentation.acceleration.measurements import process_vram_observation
    from chd_segmentation.acceleration.registry import inventory
    from chd_segmentation.artifacts.store import read_json,write_json
    from chd_segmentation.models.workflow import options
    from chd_segmentation.models.medsam2.official import effective_config
    from chd_segmentation.preprocessing.prepare import load_prepared
    if torch.cuda.is_initialized():
        raise RuntimeError('参数检查前CUDA已初始化')
    began = perf_counter()
    selected = distributed_backend('cpu')
    if selected!='gloo':
        raise AssertionError('CPU默认后端不同')
    config = read_json(root/'course/med.json')
    config.update(device='cuda',distributed_backend='gloo',run_dir=str(root/'course/runs/Med-GPU'))
    config['execution'].update(device='cuda',gpu_count=1,cpu_processes=1,min_gpus=None,batch_layout=None)
    candidate = root/'course/med-gpu.json'
    write_json(candidate,config)
    parsed = options(candidate)
    prepared = load_prepared(root/'course/med-offline.json')
    recipe = effective_config(parsed,prepared,root/'course/med-offline.json',
        Path(config['run_dir'])/'sequence',['ct_1001'])
    if recipe['trainer']['distributed']['backend']!='gloo' or recipe['scratch']['num_frames']!=8 or recipe['scratch']['resolution']!=512:
        raise AssertionError('显式Gloo改变了官方序列设置')
    directory = root/'checks/backend-parameters'
    directory.mkdir(parents=True,exist_ok=False)
    rendezvous = directory/'gloo-store'
    sample = torch.tensor([prepared['storage']['cases'][name]['array_bytes']
        for name in ('ct_1001','ct_1012')],dtype=torch.float64)
    expected = sample.clone()
    try:
        dist.init_process_group(selected,init_method=rendezvous.as_uri(),rank=0,world_size=1,
            timeout=timedelta(seconds=15))
        dist.all_reduce(sample)
        if not torch.equal(sample,expected):
            raise AssertionError('真实CPU/Gloo collective结果不同')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous.exists():
            if rendezvous.is_symlink() or rendezvous.resolve().parent!=directory.resolve():
                raise ValueError('CPU rendezvous不属于本轮检查')
            rendezvous.unlink()
        directory.rmdir()
    observed = inventory()
    external = process_vram_observation(observed[0]['uuid'],os.getpid()) if observed else dict(bytes=None,reason='no_gpu')
    if torch.cuda.is_initialized():
        raise AssertionError('CPU/参数检查意外初始化CUDA')
    write_json(reports/'backend-parameters.json',dict(state='completed',seconds=perf_counter()-began,
        torch_version=torch.__version__,gloo_available=dist.is_gloo_available(),nccl_available=dist.is_nccl_available(),
        cuda_available=torch.cuda.is_available(),cuda_initialized=False,cuda_tensors_created=0,model_updates=0,
        cpu_collective='passed',gpu_collective='not_executed',gloo_declared_devices=dist.Backend.backend_capability['gloo'],
        gpu_candidate=str(candidate),official_frames=8,official_resolution=512,batch=1,precision='fp32',
        external_observation_of_cpu_probe=external,store_directory_removed=not directory.exists(),
        limitations='现有构建声明支持CUDA/Gloo；未证明真实CUDA collective、DDP或完整模型可运行，显存未实测'))
