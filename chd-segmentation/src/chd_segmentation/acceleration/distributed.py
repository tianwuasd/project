import torch
import torch.distributed as dist


def distributed_backend(device,requested=None):
    kind = torch.device(device).type
    if kind not in ('cpu','cuda') or requested not in (None,'gloo','nccl'):
        raise ValueError('分布式设备或显式后端无效')
    selected = requested if requested is not None else 'nccl' if kind=='cuda' else 'gloo'
    if kind=='cpu' and selected!='gloo':
        raise ValueError('CPU执行需要Gloo后端')
    if selected=='nccl' and not dist.is_nccl_available():
        raise RuntimeError('当前Torch未提供NCCL；需要显式选择真实可用的后端')
    if selected=='gloo' and not dist.is_gloo_available():
        raise RuntimeError('当前Torch未提供Gloo')
    if kind=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前Torch没有可用的CUDA设备')
    if kind not in dist.Backend.backend_capability[selected]:
        raise RuntimeError('所选后端未声明支持当前设备')
    return selected
