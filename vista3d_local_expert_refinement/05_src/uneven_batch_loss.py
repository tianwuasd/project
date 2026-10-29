"""Pinned B0 loss reductions for unequal, nonempty DDP batch partitions.

The Dice sufficient statistics follow nnU-Net 2.8.1's memory-efficient Dice.
Sum the local batch before AllGatherGrad so every rank communicates [classes]
regardless of its local batch. CE still averages patches, not rank means.
"""
import torch
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class GlobalBatchDice(MemoryEfficientSoftDiceLoss):
    def forward(self,x,y,loss_mask=None):
        if self.apply_nonlin is not None:x=self.apply_nonlin(x)
        axes=tuple(range(2,x.ndim))
        with torch.no_grad():
            if x.ndim!=y.ndim:y=y.view((y.shape[0],1,*y.shape[1:]))
            if x.shape==y.shape:onehot=y.to(torch.float32)
            else:
                onehot=torch.zeros(x.shape,device=x.device,dtype=torch.float32)
                onehot.scatter_(1,y.long(),1)
            if not self.do_bg:onehot=onehot[:,1:]
            ground=(onehot if loss_mask is None else onehot*loss_mask).sum(axes,dtype=torch.float32).sum(0)
        if not self.do_bg:x=x[:,1:]
        intersect=(x*onehot if loss_mask is None else x*onehot*loss_mask).sum(axes,dtype=torch.float32).sum(0)
        predicted=(x if loss_mask is None else x*loss_mask).sum(axes,dtype=torch.float32).sum(0)
        if self.ddp:
            intersect=AllGatherGrad.apply(intersect).sum(0,dtype=torch.float32)
            predicted=AllGatherGrad.apply(predicted).sum(0,dtype=torch.float32)
            ground=AllGatherGrad.apply(ground).sum(0,dtype=torch.float32)
        return -((2*intersect+self.smooth)/(ground+predicted+float(self.smooth)).clamp_min(1e-8)).mean()


def configure(loss,*,global_batch,local_batch,world_size):
    """Retain original eager DS weights and original loss for equal partitions."""
    if global_batch%world_size==0:return loss
    leaf=loss.loss if hasattr(loss,'loss') else loss
    if not isinstance(leaf,DC_and_CE_loss) or leaf.ignore_label is not None:
        raise ValueError('unequal batches currently require registered B0 class labels without ignore regions')
    scale=world_size*local_batch/global_batch
    leaf.weight_ce*=scale
    dc=leaf.dc
    if dc.batch_dice:
        leaf.dc=GlobalBatchDice(apply_nonlin=dc.apply_nonlin,batch_dice=True,do_bg=dc.do_bg,
            smooth=dc.smooth,ddp=dc.ddp)
    else:leaf.weight_dice*=scale
    return loss
