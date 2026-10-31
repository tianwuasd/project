"""Actual CPU Gloo comparison with one unsplit eight-patch B0 loss update."""
def compare(rank,world,args):
    import copy
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    from batch_distribution import layout
    from uneven_batch_loss import configure
    torch.set_num_threads(1);torch.manual_seed(17)
    batch=args.get('global_batch',8)
    sizes=layout(batch,world,global_batch=batch);n=sizes[rank];offset=sum(sizes[:rank])
    native=SimpleNamespace(is_ddp=True,configuration_manager=SimpleNamespace(batch_size=batch),oversample_foreground_percent=.33)
    nnUNetTrainer._set_batch_size_and_oversample(native)
    assert native.batch_size==n
    foreground=n-round(n*(1-native.oversample_foreground_percent))
    flags=[None]*world;dist.all_gather_object(flags,foreground)
    assert sum(flags)==batch-round(batch*(1-.33)),flags
    model=torch.nn.Conv3d(1,3,1,bias=True);reference=copy.deepcopy(model)
    network=DDP(model)
    generator=torch.Generator().manual_seed(200)
    x=torch.randn(batch,1,4,4,4,generator=generator)
    y=torch.randint(0,3,(batch,1,4,4,4),generator=generator)
    def criterion(ddp):
        leaf=DC_and_CE_loss(dict(batch_dice=args.get('batch_dice',True),smooth=1e-5,do_bg=False,ddp=ddp),{},dice_class=MemoryEfficientSoftDiceLoss)
        return DeepSupervisionWrapper(leaf,[1-1e-6,1e-6])
    local_loss=configure(criterion(True),global_batch=batch,local_batch=n,world_size=world)
    prediction=network(x[offset:offset+n])
    actual=local_loss([prediction,prediction[:,:,::2,::2,::2]],[y[offset:offset+n],y[offset:offset+n,:,::2,::2,::2]])
    actual.backward()
    prediction=reference(x)
    expected=criterion(False)([prediction,prediction[:,:,::2,::2,::2]],[y,y[:,:,::2,::2,::2]])
    expected.backward()
    error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(model.parameters(),reference.parameters()))
    assert error<1e-6,error
    mean_loss=actual.detach().clone();dist.all_reduce(mean_loss);mean_loss/=world
    assert float((mean_loss-expected.detach()).abs())<1e-6
    # A real parameter update must retain the same sample weighting.
    torch.optim.SGD(model.parameters(),lr=.01).step()
    torch.optim.SGD(reference.parameters(),lr=.01).step()
    update_error=max(float((p-q).abs().max().detach()) for p,q in zip(model.parameters(),reference.parameters()))
    assert update_error<1e-6,update_error
    return dict(status='passed',rank=rank,world_size=world,global_batch=batch,local_batch=n,batch_dice=args.get('batch_dice',True),
        layout=sizes,global_foreground=sum(flags),gradient_max_error=error,
        update_max_error=update_error,deep_supervision=True,device='cpu',real_ct=False,nccl=False)
