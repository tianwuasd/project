"""Compare true two-rank Gloo gradients to one global ignored-label batch."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


def worker(rank,port,output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from uneven_batch_loss import configure
    torch.set_num_threads(1)
    from datetime import timedelta
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,timeout=timedelta(seconds=45))
    try:
        results=[]
        for batch_dice in (False,True):
            for ignored_rank in (False,True):
                torch.manual_seed(171)
                net=torch.nn.Conv3d(1,8,1,bias=True)
                reference=torch.nn.Conv3d(1,8,1,bias=True);reference.load_state_dict(net.state_dict())
                x=torch.randn(3,1,5,6,7)
                y=torch.randint(0,8,(3,1,5,6,7));y[0,:,:2]=8;y[1,:,:1]=8
                if ignored_rank:y[2]=8
                def objective(ddp):
                    return DC_and_CE_loss(dict(batch_dice=batch_dice,smooth=1e-5,do_bg=False,ddp=ddp),{},
                        weight_ce=1,weight_dice=1,ignore_label=8,dice_class=MemoryEfficientSoftDiceLoss)
                expected=objective(False)(reference(x),y);expected.backward()
                model=DDP(net)
                local=slice(0,2) if rank==0 else slice(2,3)
                loss=configure(objective(True),global_batch=3,local_batch=2 if rank==0 else 1,world_size=2)
                value=loss(model(x[local]),y[local]);value.backward()
                error=max((a.grad-b.grad).abs().max().item() for a,b in zip(model.module.parameters(),reference.parameters()))
                assert error<3e-6,(batch_dice,ignored_rank,error)
                mean=value.detach().clone();dist.all_reduce(mean);mean/=2
                assert abs(mean.item()-expected.item())<3e-6
                results.append(dict(batch_dice=batch_dice,entire_rank_ignored=ignored_rank,max_gradient_error=error,
                    global_loss=mean.item(),single_batch_loss=expected.item()))
        if rank==0:
            Path(output).write_text(json.dumps(dict(status='passed',backend='gloo',ranks=2,
                global_batch=3,local_batches=[2,1],comparisons=results),indent=2),encoding='utf-8')
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    import torch.multiprocessing as mp
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    mp.spawn(worker,args=(port,args.output),nprocs=2,join=True)
    print(Path(args.output).read_text(encoding='utf-8'))
