"""Marginal cross entropy and Dice on each case's observable label partition."""
import torch
from torch import nn


def marginal_loss(logits, target, unknown_classes, dice_weight=1., smooth=1e-5):
    """Mean per-case CE + Dice; target -1 is ignored, no dense allowed-set tensor.

    Background and unknown classes form one observable group. Remaining classes
    stay separate, including reviewed true absences. Dice includes the merged
    background group; this is explicitly not nnU-Net's unmodified Dice loss.
    """
    if target.ndim==logits.ndim:
        if target.shape[1]!=1:
            raise ValueError('Expected one integer target channel')
        target=target[:,0]
    if logits.ndim<3 or logits.shape[0]!=len(unknown_classes) or target.shape!=(logits.shape[0],*logits.shape[2:]):
        raise ValueError('Partial-label batch metadata/shape mismatch')
    result=[]
    for i, unknown in enumerate(unknown_classes):
        unknown=set(unknown)
        if not unknown.issubset(set(range(1,logits.shape[1]))):
            raise ValueError('Unknown class index outside foreground channels')
        y=target[i].long().flatten()
        valid=y!=-1
        if not valid.any():
            continue
        y=y[valid]
        if y.min()<0 or y.max()>=logits.shape[1] or any((y==c).any() for c in unknown):
            raise ValueError('Target conflicts with case-level annotation metadata')
        x=logits[i].float().flatten(1)[:,valid]
        logden=torch.logsumexp(x,dim=0)
        lognum=x.gather(0,y[None])[0]
        bg=torch.logsumexp(x[[0]+sorted(unknown)],dim=0)
        ce=(logden-torch.where(y==0,bg,lognum)).mean()
        if dice_weight:
            p=x.softmax(0)
            groups=[[0]+sorted(unknown)]+[[c] for c in range(1,x.shape[0]) if c not in unknown]
            dice=[]
            for group in groups:
                probability=p[group].sum(0)
                truth=(y==group[0]).to(probability.dtype)
                dice.append((2*(probability*truth).sum()+smooth)/(probability.sum()+truth.sum()+smooth))
            ce=ce+dice_weight*(1-torch.stack(dice).mean())
        result.append(ce)
    return torch.stack(result).mean() if result else logits.sum()*0


class MarginalBatchLoss(nn.Module):
    """Bind batch keys before each train/validation call, including deep supervision."""
    def __init__(self):
        super().__init__()
        self.unknown=None

    def bind(self, keys, metadata):
        if keys is None or not len(keys) or any(k not in metadata for k in keys):
            raise ValueError('Missing case keys or partial-supervision metadata')
        self.unknown=[metadata[k]['unknown_classes'] for k in keys]

    def forward(self, output, target):
        if self.unknown is None:
            raise ValueError('Loss requires batch-key supervision binding')
        outputs=output if isinstance(output,(tuple,list)) else [output]
        targets=target if isinstance(target,(tuple,list)) else [target]
        if len(outputs)!=len(targets):
            raise ValueError('Deep-supervision outputs/targets differ')
        weights=[1/(2**i) for i in range(len(outputs))]
        if len(weights)>1:
            weights[-1]=0
        return sum(w*marginal_loss(x,y,self.unknown) for w,x,y in zip(weights,outputs,targets) if w)/sum(weights)


def observed_confusion(output, target, unknown_classes):
    """Per-class counts only from cases where that class is annotated/known absent."""
    if target.ndim==output.ndim:
        target=target[:,0]
    pred=output.argmax(1)
    tp,fp,fn=[torch.zeros(output.shape[1]-1,device=output.device) for _ in range(3)]
    for i,unknown in enumerate(unknown_classes):
        valid=target[i]!=-1
        for c in range(1,output.shape[1]):
            if c in unknown:
                continue
            truth=target[i]==c
            found=pred[i]==c
            tp[c-1]+=(truth & found & valid).sum()
            fp[c-1]+=(~truth & found & valid).sum()
            fn[c-1]+=(truth & ~found & valid).sum()
    return [v.detach().cpu().numpy() for v in (tp,fp,fn)]
