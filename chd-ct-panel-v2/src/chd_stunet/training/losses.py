"""Partial supervision with a merged background and available-class Dice.

For partial MYO annotations, a source 0 is compatible with class 0 OR 5.
Its CE uses logsumexp for that set. Dice excludes background and masks each
case's unavailable class, so missing MYO never becomes a negative example.
With fully annotated inputs this reduces to nnU-Net CE + foreground Dice.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PartialDiceCE(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5):
        super().__init__()
        self.batch_dice, self.smooth = batch_dice, smooth

    def components(self, logits, target, available):
        # Sum/softmax reductions stay float32 under AMP; double is retained for
        # independent numerical gradient checks. Never reduce half logits here.
        values = logits if logits.dtype == torch.float64 else logits.float()
        if target.is_floating_point() and not bool(torch.equal(target,target.round())):
            raise ValueError('Segmentation targets must be discrete')
        target = target[:,0].long() if target.ndim == logits.ndim else target.long()
        available = available.to(device=logits.device,dtype=torch.bool)
        if logits.shape[1] != 8 or available.shape != (logits.shape[0],7):
            raise ValueError('Expected 8 logits and seven class-availability flags per case')
        if target.shape != logits.shape[:1]+logits.shape[2:]:
            raise ValueError('Target/logit shape mismatch')
        if not bool(((target>=-1)&(target<=8)).all()):
            raise ValueError('Unknown segmentation labels')
        complete = available.all(1)
        missing_myo = available[:,[0,1,2,3,5,6]].all(1) & ~available[:,4]
        if not bool((complete | missing_myo).all()):
            raise ValueError('Only complete or missing-MYO supervision is supported')
        valid = (target>=0)&(target<8)
        safe = target.clamp(0,7)
        if bool(((target==5)&missing_myo.view(-1,*([1]*(target.ndim-1)))).any()):
            raise ValueError('Partial-MYO input contains an invented MYO target')
        logp = F.log_softmax(values,dim=1)
        selected = logp.gather(1,safe[:,None]).squeeze(1)
        ambiguous = (target==0)&missing_myo.view(-1,*([1]*(target.ndim-1)))
        merged = torch.logsumexp(logp[:,[0,5]],dim=1)
        ce_voxels = -torch.where(ambiguous,merged,selected)
        ce = (ce_voxels*valid).sum()/valid.sum().clamp_min(1)

        # Multiplying availability BEFORE spatial/batch sums is essential: a
        # partial case contributes neither MYO TP nor MYO denominator at all.
        probabilities = logp[:,1:].exp()
        truth = torch.zeros(values.shape,device=values.device,dtype=torch.bool)
        truth.scatter_(1,safe[:,None],True)
        truth = truth[:,1:]
        axes = tuple(range(2,logits.ndim))
        intersect = (probabilities*truth*valid[:,None]).sum(axes)*available
        predicted = (probabilities*valid[:,None]).sum(axes)*available
        reference = (truth & valid[:,None]).sum(axes)*available
        active = available & valid.flatten(1).any(1)[:,None]
        if self.batch_dice:
            intersect,predicted,reference = (x.sum(0) for x in (intersect,predicted,reference))
            active = active.any(0)
        dice = (2*intersect+self.smooth)/(predicted+reference+self.smooth).clamp_min(1e-8)
        if self.batch_dice:
            dc = -(dice*active).sum()/active.sum().clamp_min(1)
        else:
            dc = -((dice*active).sum(1)/active.sum(1).clamp_min(1)).mean()
        return ce,dc

    def forward(self, logits, target, available):
        ce,dc = self.components(logits,target,available)
        return ce+dc


class DeepSupervisionLoss(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5):
        super().__init__()
        self.base = PartialDiceCE(batch_dice,smooth)

    def forward(self, outputs, targets, available):
        if not isinstance(outputs,(tuple,list)):
            return self.base(outputs,targets,available)
        if not isinstance(targets,(tuple,list)) or len(outputs)!=len(targets):
            raise ValueError('Deep-supervision outputs/targets differ')
        weights = [1/(2**i) for i in range(len(outputs))]
        weights[-1] = 0
        total = sum(weights)
        return sum(w/total*self.base(output,target,available)
                   for w,output,target in zip(weights,outputs,targets) if w)
