def rank_indices(step, batch, shares, rank):
    if sum(shares) != batch or not 0 <= rank < len(shares) or min(shares) < 1 or step < 0:
        raise ValueError('全局批量、份额或rank无效')
    start = step*batch+sum(shares[:rank])
    return range(start,start+shares[rank])


def topology(config):
    from ...acceleration.registry import layout
    resources = config['resources']
    world = resources['gpu_count'] if config['device'] == 'cuda' else 1
    batch = config['training']['batch'] if 'training' in config else world
    return dict(world=world,batch=batch,shares=layout(batch,world,resources['batch_layout'] if 'training' in config else None))


def require_topology(saved, expected):
    if saved != expected:
        raise ValueError('恢复需要相同的卡数、全局批量及逐rank份额')


def claim_prediction_slice(handle, limit):
    if limit is None:
        return True
    with handle.condition:
        count = handle.state['medsam3_prediction_claimed']
        if count >= limit:
            return False
        handle.state['medsam3_prediction_claimed'] = count+1
        return True


def validation_rounds(items, world, rank):
    for start in range(0,len(items),world):
        index = start+rank
        yield items[index] if index < len(items) else None


def rng_state():
    import random
    import numpy as np
    import torch
    rng = np.random.get_state()
    return dict(python=random.getstate(),numpy=(rng[0],rng[1].tolist(),*rng[2:]),
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(value):
    import random
    import numpy as np
    import torch
    random.setstate(value['python'])
    rng = value['numpy']
    np.random.set_state((rng[0],np.asarray(rng[1],dtype=np.uint32),*rng[2:]))
    torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        torch.cuda.set_rng_state(value['cuda'])
