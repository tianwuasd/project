GIB = 1024**3


def batch_bytes(largest_source_voxels, batch, side):
    if min(largest_source_voxels,batch,side) <= 0:
        raise ValueError('病例体素、batch 和网络边长必须为正整数')
    return 64*1024**2 + 32*largest_source_voxels + 48*batch*side**3


def stage_minimum(args, name):
    if not getattr(args,'prefetch',False) or name not in ('crop64','crop128','all64','all128','init64','init128'):
        return 0
    side = 128 if name.endswith('128') else 64
    batch = 4 if args.profile == 'paper' else 1
    return batch_bytes(max(side**3,getattr(args,'max_source_voxels',0)),batch,side)


def prefetch_budget(host, model, minimum, requested):
    from .xu_resources import HostResourcesChanged
    available = max(0,host-model-max(GIB,host//5))
    if minimum > available:
        raise HostResourcesChanged(f'单批供数需要 {minimum} 字节；本阶段可分配 {available} 字节')
    return min(max(requested,minimum),available)
