import torch
from torch import nn
import torch.nn.functional as F
from .events import PAIRS,edge_masks
from .solver import solve_local
from .vendor.cldice_skeleton import SoftSkeletonize

TRAINABLE_METHODS = ('M01','M02','M03','M04','M05','M06','M07','M08','M09')


class ConditionalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(77,24,3,padding=1,bias=False),nn.GroupNorm(8,24),nn.SiLU(),
                                  nn.Conv3d(24,24,3,padding=1,bias=False),nn.GroupNorm(8,24),nn.SiLU(),nn.Conv3d(24,11,1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features, z0, q, editable):
        valid = edge_masks(editable)
        condition = (q.detach()*valid[:,:,None]).flatten(1,2)
        inputs = torch.cat((features,z0,condition,valid.float()),1)
        return z0+self.body(inputs).float()*editable


def scnp_logits(logits, allowed, editable):
    if logits.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError('SCNP需要与logits一致的允许类别集合')
    exact = (allowed.sum(1,keepdim=True)==1) & editable
    positive = allowed & exact
    negative = ~allowed & exact
    minimum = -F.max_pool3d(-logits.masked_fill(~positive,torch.inf),3,1,1)
    maximum = F.max_pool3d(logits.masked_fill(~negative,-torch.inf),3,1,1)
    penalized = torch.where(allowed,minimum,maximum)
    return torch.where(exact,penalized,logits)


def partial_cldice(probability, allowed, editable, iterations=10, classes=(6,7,8), smooth=1.):
    if iterations not in (10,20):
        raise ValueError('clDice只接受预注册的10或20次软骨架迭代')
    exact = allowed.sum(1,keepdim=True)==1
    margin = iterations+2
    interior = torch.zeros_like(editable)
    if all(n > 2*margin for n in probability.shape[2:]):
        interior[:,:,margin:-margin,margin:-margin,margin:-margin] = True
    domain = interior & editable
    skeleton = SoftSkeletonize(iterations)
    terms = []
    for batch in range(probability.shape[0]):
        for c in classes:
            known = exact[batch,0] | ~allowed[batch,c]
            if not known.all() or not editable[batch].all() or not domain[batch].any():
                continue
            truth = (allowed[batch:batch+1,c:c+1] & exact[batch:batch+1]).float()
            if not (truth*domain[batch:batch+1]).any():
                continue
            predicted = probability[batch:batch+1,c:c+1]
            skel_pred = skeleton(predicted)
            skel_true = skeleton(truth)
            valid = domain[batch:batch+1]
            precision = ((skel_pred*truth*valid).sum()+smooth)/((skel_pred*valid).sum()+smooth)
            recall = ((skel_true*predicted*valid).sum()+smooth)/((skel_true*valid).sum()+smooth)
            terms.append(1-2*precision*recall/(precision+recall))
    return (torch.stack(terms).mean() if terms else probability.sum()*0),len(terms)


def infer_local(outputs, editable, method, solver_config, checkpointing=False, steps=None):
    if method not in TRAINABLE_METHODS:
        raise ValueError('局部模型方法编号不合法')
    core = (slice(None),slice(None),*outputs['core'])
    z0,mask = outputs['z0'][core],editable[core]
    if method == 'M03':
        return outputs['conditional_logits'],None
    if method == 'M01':
        return z0,None
    config = dict(solver_config)
    if steps is not None:
        config['steps'] = steps
    if method in ('M02','M04','M05','M09'):
        config['steps'] = 0
    active = None
    if method in ('M06','M08'):
        active = torch.tensor([c!=d if method=='M06' else c==d for c,d in PAIRS],device=z0.device)
    trace = solve_local(z0,outputs['q'],mask,active=active,checkpointing=checkpointing,**config)
    return trace.logits,trace

