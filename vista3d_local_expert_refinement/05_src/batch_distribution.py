"""Pure allocation of a capped global batch; no CUDA or environment changes."""
MAX_GLOBAL_BATCH = 8


def layout(local_capacity, world_size, *, global_batch=None):
    for name,value in [('local capacity',local_capacity),('world size',world_size)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(name+' must be a positive integer')
    if world_size>MAX_GLOBAL_BATCH:
        raise ValueError('at most eight nonempty ranks per capped batch')
    if global_batch is None:global_batch=min(MAX_GLOBAL_BATCH,local_capacity*world_size)
    if (isinstance(global_batch,bool) or not isinstance(global_batch,int)
            or not world_size<=global_batch<=min(MAX_GLOBAL_BATCH,local_capacity*world_size)):
        raise ValueError('global batch must fit all nonempty ranks and the eight-patch cap')
    return [global_batch//world_size+int(rank<global_batch%world_size) for rank in range(world_size)]


def weighted_shares(total, weights, minimum=0):
    """Integer largest-remainder shares, reserving one main CPU per rank first."""
    if total<minimum*len(weights) or not weights or any(w<=0 for w in weights):
        raise ValueError('insufficient budget or invalid allocation weights')
    remaining=total-minimum*len(weights);denominator=sum(weights)
    values=[minimum+remaining*w//denominator for w in weights]
    order=sorted(range(len(weights)),key=lambda i:(-(remaining*weights[i]%denominator),i))
    for i in order[:total-sum(values)]:values[i]+=1
    return values
