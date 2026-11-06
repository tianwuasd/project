"""Real CPU/Gloo B4 partial-epoch relocation, synthetic; not CUDA/NCCL proof."""
import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from distributed_smoke import TinyDS,inputs,seed
from chd_stunet.training.losses import DeepSupervisionLoss
from chd_stunet.training.parallel_policy import slots_for
from chd_stunet.training.rank_runtime import commit,restore_rank_rng,train_attempt,validate_attempt
from chd_stunet.training.checkpoints import read_checkpoint,restore_checkpoint


def worker(rank,world,root,generation,start,end):
    root=Path(root);device=torch.device('cpu');seed(932)
    dist.init_process_group('gloo',init_method=(root/f'store-{generation}').as_uri(),rank=rank,world_size=world,
                            timeout=timedelta(seconds=90))
    try:
        core=TinyDS().double();optimizer=torch.optim.SGD(core.parameters(),lr=.02,momentum=.99)
        scaler=torch.amp.GradScaler('cpu',init_scale=16.,growth_interval=100)
        members=[f'group-{generation}-rank-{r}' for r in range(world)];signature={'identity':'portable-global-batch'}
        state=dict(epoch=0,phase='train',step=0,attempts=0,updates=0)
        if start:
            payload=read_checkpoint(root/'checkpoint',signature)
            state=restore_checkpoint(payload,core,optimizer,scaler,device)
            restore_rank_rng(payload,members,rank,device,42,allow_partial=True)
            assert state['attempts']==start,'Cursor lost across relocation'
        model=DDP(core,find_unused_parameters=True);criterion=DeepSupervisionLoss(True)
        x,y,a=inputs('mixed');indices=slots_for(4,world,rank)
        if start and state['phase']=='validation':
            assert state['step']==1 and state['val_batches']==1,'Partial validation cursor lost'
            batch=dict(data=(x[indices]+start*.01).numpy(),target=[y[indices].numpy()]*5,available=a[indices].numpy())
            checked=validate_attempt(model,criterion,batch,dict(amp=False),device)
            assert math.isclose(checked['loss'],state['val_loss_sum'],rel_tol=1e-12,abs_tol=1e-12)
            for key in ('tp','fp','fn'):assert checked[key]==state[key],key+' statistics lost'
        for step in range(start,end):
            batch=dict(data=(x[indices]+step*.01).numpy(),target=[y[indices].numpy()]*5,available=a[indices].numpy())
            # Inject one global overflow independently of the rank layout.
            handle=None
            if step==2 and rank==world-1:
                handle=next(core.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
            result=train_attempt(model,optimizer,scaler,criterion,batch,dict(amp=False,gradient_clip=12.),device)
            if handle:handle.remove()
            state.update(step=step+1,attempts=step+1,updates=state['updates']+int(result['update_applied']))
        if end in (4,7):state.update(phase='train',step=end)
        else:
            batch=dict(data=(x[indices]+end*.01).numpy(),target=[y[indices].numpy()]*5,available=a[indices].numpy())
            checked=validate_attempt(model,criterion,batch,dict(amp=False),device)
            state.update(phase='validation',step=1,val_batches=1,val_loss_sum=checked['loss'],
                         **{key:checked[key] for key in ('tp','fp','fn')})
        commit(root/'checkpoint',model,optimizer,scaler,state,signature,device,members)
        if rank==0:
            torch.save(dict(model=core.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),state=state),root/f'end-{generation}.pt')
    finally:dist.destroy_process_group()


def assert_close(left,right,path='root'):
    if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12,msg=path)
    elif isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left:assert_close(left[key],right[key],path+'.'+str(key))
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right),path
        for i,(a,b) in enumerate(zip(left,right)):assert_close(a,b,path+'.'+str(i))
    elif isinstance(left,float):assert math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12),path
    else:assert left==right,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reference=args.output/'reference';reference.mkdir(parents=True)
    mp.spawn(worker,args=(1,str(reference),0,0,10),nprocs=1,join=True)
    moved=args.output/'moved';moved.mkdir();schedule=[(4,0,4),(2,4,6),(1,6,7),(3,7,10)]
    for generation,(world,start,end) in enumerate(schedule):
        print(f'Gloo relocation generation {generation}: world={world}, attempts={start}..{end}',flush=True)
        mp.spawn(worker,args=(world,str(moved),generation,start,end),nprocs=world,join=True)
    expected=torch.load(reference/'end-0.pt',weights_only=True)
    actual=torch.load(moved/'end-3.pt',weights_only=True)
    assert_close(expected,actual)
    receipt=dict(status='passed',scope='real CPU/Gloo, synthetic, not multi-CUDA/NCCL',schedule=schedule,
                 global_batch=4,attempts=actual['state']['attempts'],updates=actual['state']['updates'],
                 model_optimizer_scaler_cursor_close=True,tolerance=1e-12,partial_validation_cursor_preserved=True)
    (args.output/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
