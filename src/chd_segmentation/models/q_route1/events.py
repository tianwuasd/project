import torch
import torch.nn.functional as F

CLASSES = ('background', 'LV', 'RV', 'LA', 'RA', 'Myo', 'AO', 'PA', 'PV', 'SVC', 'IVC')
PAIRS = ((6,1),(6,2),(7,1),(7,2),(6,7),(6,6),(7,7),(8,8),(8,3),(8,4),(8,9),(8,10))


def endpoints(x, direction):
    left, right = [slice(None)] * 5, [slice(None)] * 5
    left[direction + 2], right[direction + 2] = slice(None,-1), slice(1,None)
    return x[tuple(left)], x[tuple(right)]


def pad_edge(x, direction, leading=False):
    padding = [0] * 6
    padding[2*(2-direction) + (0 if leading else 1)] = 1
    return F.pad(x, padding)


def event_probabilities(left, right, active=None):
    selected = []
    for c,d in PAIRS:
        value = left[:,c:c+1] * right[:,d:d+1]
        if c != d:
            value = value + left[:,d:d+1] * right[:,c:c+1]
        selected.append(value)
    selected = torch.cat(selected,1)
    if active is not None:
        selected = selected * active.to(selected)[None,:,None,None,None]
    return torch.cat((1-selected.sum(1,keepdim=True),selected),1)


def edge_masks(editable):
    return torch.stack([pad_edge(a & b, d)[:,0] for d in range(3)
                        for a,b in [endpoints(editable,d)]],1)


def allowed_edge_events(left, right):
    if left.dtype != torch.bool or right.dtype != torch.bool:
        raise TypeError('允许类别集合需要bool类型')
    mapping = torch.zeros(11,11,dtype=torch.long,device=left.device)
    for j,(c,d) in enumerate(PAIRS,1):
        mapping[c,d] = mapping[d,c] = j
    result = torch.zeros((left.shape[0],13,*left.shape[2:]),dtype=torch.bool,device=left.device)
    for c in range(11):
        for d in range(11):
            result[:,mapping[c,d]] |= left[:,c] & right[:,d]
    informative = ~left.all(1) & ~right.all(1) & ~result.all(1)
    return result, informative

