import numpy as np


def validate_probability(probability):
    if probability.ndim != 4 or probability.shape[0] != 11 or not np.isfinite(probability).all():
        raise ValueError('基础概率需要有限的11通道三维数组')
    if np.any(probability < 0) or not np.allclose(probability.sum(0),1,rtol=0,atol=2e-6):
        raise ValueError('基础概率未归一化')


def apply_corruption(probability, region, kind, c, d=None, parameter=None):
    validate_probability(probability)
    if region.shape != probability.shape[1:] or region.dtype != bool or not 1 <= c <= 10:
        raise ValueError('概率扰动区域或类别不合法')
    output = probability.copy()
    if kind == 'deletion':
        if parameter is None or not 0 <= parameter <= .2:
            raise ValueError('删除扰动保留比例应为0到0.2')
        removed = (1-parameter)*probability[c,region]
        output[c,region] = probability[c,region]*parameter
        output[0,region] = probability[0,region]+removed
    elif kind == 'bridge':
        if parameter is None or not .7 <= parameter <= .95:
            raise ValueError('桥接扰动比例应为0.7到0.95')
        output[:,region] = probability[:,region]*(1-parameter)
        output[c,region] += parameter
    elif kind == 'exchange':
        if d is None or not 1 <= d <= 10 or c == d:
            raise ValueError('类别交换需要两个不同前景类别')
        output[c,region],output[d,region] = probability[d,region],probability[c,region]
    else:
        raise ValueError('未知概率扰动方式')
    validate_probability(output)
    return output


def _region(shape, center, radius):
    axes = np.ogrid[tuple(slice(0,n) for n in shape)]
    return sum((grid-c)**2 for grid,c in zip(axes,center)) <= radius**2


def _corridor(shape, start, end, radius):
    direction = end-start
    length_squared = float(direction @ direction)
    if length_squared == 0:
        raise ValueError('桥接走廊的端点必须不同')
    grids = np.ogrid[tuple(slice(0,n) for n in shape)]
    projection = sum((grid-a)*d for grid,a,d in zip(grids,start,direction))/length_squared
    fraction = np.clip(projection,0,1)
    distance = sum((grid-(a+fraction*d))**2 for grid,a,d in zip(grids,start,direction))
    return (projection >= 0) & (projection <= 1) & (distance <= radius**2)


def controlled_corruption(probability, allowed, editable, rng, chance=.2, proposals=8):
    validate_probability(probability)
    if allowed.shape != probability.shape or allowed.dtype != bool or editable.shape != probability.shape[1:]:
        raise ValueError('扰动监督合同形状或类型不一致')
    if not 0 <= chance <= 1 or not 1 <= proposals <= 8:
        raise ValueError('扰动概率或提案预算不合法')
    exact = (allowed.sum(0) == 1) & editable
    truth = allowed.argmax(0)
    eligible = np.argwhere(exact & (truth > 0))
    if not len(eligible):
        return probability,dict(applied=False,reason='empty_corruption_set',proposals=0)
    if rng.random() >= chance:
        return probability,dict(applied=False,reason='real_probability',proposals=0)
    before = (probability.argmax(0) != truth) & exact
    accepted = []
    for _ in range(proposals):
        kind = ('deletion','bridge','exchange')[rng.integers(3)]
        center = eligible[rng.integers(len(eligible))]
        c = int(truth[tuple(center)])
        radius = int(rng.integers(2,6))
        region = _region(editable.shape,center,radius)
        parameter,d = None,None
        if kind == 'bridge':
            endpoints = np.argwhere(exact & (truth == c))
            squared = ((endpoints-center)**2).sum(1)
            nearby = endpoints[(squared >= (2*radius)**2) & (squared <= (4*radius)**2)]
            if not len(nearby):
                continue
            end = nearby[rng.integers(len(nearby))]
            region = _corridor(editable.shape,center,end,radius) & exact & ~allowed[c]
            parameter = float(rng.uniform(.7,.95))
        else:
            region &= exact & (truth == c)
            if kind == 'deletion':
                parameter = float(rng.uniform(0,.2))
            else:
                choices = [k for k in range(1,11) if k != c]
                d = int(choices[rng.integers(len(choices))])
        if not region.any():
            continue
        output = apply_corruption(probability,region,kind,c,d,parameter)
        after = (output.argmax(0) != truth) & exact
        increase = int(after.sum())-int(before.sum())
        if increase > 0:
            accepted.append((kind,c,d,parameter,radius,region,increase))
    if not accepted:
        return probability,dict(applied=False,reason='no_valid_corruption',proposals=proposals)
    kind,c,d,parameter,radius,region,increase = accepted[rng.integers(len(accepted))]
    output = apply_corruption(probability,region,kind,c,d,parameter)
    return output,dict(applied=True,kind=kind,c=c,d=d,parameter=parameter,radius_voxels=radius,
                       changed_region_voxels=int(region.sum()),added_errors=increase,
                       proposals=proposals,accepted_proposals=len(accepted))

