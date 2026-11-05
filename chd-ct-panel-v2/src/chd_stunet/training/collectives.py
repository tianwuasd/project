"""Differentiable sufficient-statistic sums, including unequal/ignored shards.

Every rank evaluates the same global objective. Backward sums its adjoints;
DDP subsequently averages parameter gradients, cancelling that world factor.
Do not replace this with an ordinary detached all_reduce or rank-mean loss.
"""
import torch
import torch.distributed as dist


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        result=value.contiguous().clone()
        dist.all_reduce(result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        result=gradient.contiguous().clone()
        dist.all_reduce(result)
        return result


def differentiable_sum(value):
    return _Sum.apply(value) if world_size()>1 else value


def sum_detached(value):
    result=value.detach().clone()
    if world_size()>1:dist.all_reduce(result)
    return result


def any_rank(value, device):
    flag=torch.tensor(int(bool(value)),device=device,dtype=torch.int32)
    if world_size()>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag.item())


def all_objects(value):
    if world_size()==1:return [value]
    values=[None]*world_size()
    dist.all_gather_object(values,value)
    return values


def synchronize_overflow(optimizer, scaler, device):
    """Make inf evidence identical BEFORE GradScaler.step/update on every rank.

    Inspecting gradients explicitly also supports optimizers with no local
    non-finite value when another rank reported one. GradScaler's per-optimizer
    inf tensors are the audited PyTorch adapter; refuse an API change rather
    than accidentally advancing one rank. Normal update resets growth tracker.
    """
    bad=any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for group in optimizer.param_groups for p in group['params'])
    overflow=any_rank(bad,device)
    if scaler.is_enabled():
        evidence=scaler._found_inf_per_device(optimizer)
        if not evidence:raise RuntimeError('GradScaler unscale evidence unavailable')
        for tensor in evidence.values():tensor.fill_(float(overflow))
    elif overflow:
        raise ValueError('Non-finite FP32 gradient; all ranks refused the update')
    return overflow
