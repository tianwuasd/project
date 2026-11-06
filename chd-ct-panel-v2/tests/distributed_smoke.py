"""Real CPU/Gloo DDP numerical, overflow and checkpoint/replay acceptance.

Standalone (not silently run by ordinary unit discovery):
  python tests/distributed_smoke.py --output <private-test-directory>
NCCL/CT throughput is deliberately outside this synthetic evidence.
"""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chd_stunet.training.losses import DeepSupervisionLoss
from chd_stunet.training.parallel_policy import slots_for,layout
from chd_stunet.training.collectives import synchronize_overflow
from chd_stunet.training.rank_runtime import commit,restore_rank_rng
from chd_stunet.training.checkpoints import read_checkpoint,restore_checkpoint,rng_state
from chd_stunet.training.supply_feedback import RankSupplyWindow, SupplyTrials, agree_on_layout


class TinyDS(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv3d(2,8,1) for _ in range(5)])
    def forward(self,x):return [layer(x) for layer in self.layers]


def inputs(mode):
    generator=torch.Generator().manual_seed(389)
    x=torch.randn(4,2,2,3,4,generator=generator,dtype=torch.float64)
    y=torch.randint(0,9,(4,1,2,3,4),generator=generator)
    a=torch.ones(4,7,dtype=torch.bool)
    if mode in ('mixed','partial'):
        a[[1,3] if mode=='mixed' else [0,1,2,3],4]=False
        for i in range(4):
            if not a[i,4]:y[i][y[i]==5]=0
    if mode=='mixed':y[2]=8;y[0,:,:,0]=8
    if mode=='ignore':y.fill_(8)
    return x,y,a


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.set_num_threads(1)


def reference(mode,batch_dice):
    seed(932);model=TinyDS().double();x,y,a=inputs(mode)
    criterion=DeepSupervisionLoss(batch_dice)
    value=criterion(model(x),[y]*5,a);value.backward()
    grads={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    torch.optim.SGD(model.parameters(),lr=.02,momentum=.99).step()
    return dict(value=float(value.detach()),grads=grads,model=model.state_dict())


def worker(rank,world,folder):
    folder=Path(folder);seed(932)
    dist.init_process_group('gloo',init_method=(folder/'store').as_uri(),rank=rank,world_size=world,
                            timeout=timedelta(seconds=90))
    rows=[]
    try:
        # All ranks must enact the same feedback decision even when only the
        # final rank has a slow supplier. Inject host limits, not DDP messages.
        from unittest.mock import patch
        from chd_stunet.acceleration.preprocess_cpu import GIB
        window=RankSupplyWindow()
        for _ in range(40):
            window.observe(.4 if rank==world-1 else .01,.2,.5)
        current=[2]*(world-1)+[1]
        host=dict(cpu_permitted=64,cpu_budget=30,available_memory_bytes=60*GIB,memory_budget_bytes=48*GIB)
        with patch('chd_stunet.training.supply_feedback.snapshot',return_value=host):
            decision=agree_on_layout(rank,window,0,SupplyTrials(),current,
                dict(stage={'reserve_bytes':4*GIB},worker_estimate_bytes=[GIB]*world),
                None,250,169)
        assert decision['workers']==[2]*world and len(decision['windows'])==world
        for mode in ('complete','mixed','partial','ignore'):
            for batch_dice in (True,False):
                expected=torch.load(folder/f'ref-{mode}-{batch_dice}.pt',weights_only=True)
                seed(932);model=DDP(TinyDS().double(),find_unused_parameters=True)
                criterion=DeepSupervisionLoss(batch_dice);indices=slots_for(4,world,rank)
                x,y,a=inputs(mode)
                optimizer=torch.optim.SGD(model.parameters(),lr=.02,momentum=.99)
                # Two consecutive DDP forwards also exercise the zero-weight
                # final head: it must not leave a pending reduction next step.
                for repeat in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    value=criterion(model(x[indices]),[y[indices]]*5,a[indices]);value.backward()
                    if repeat==0:
                        if abs(value.item()-expected['value'])>1e-10:raise AssertionError('Global objective differs')
                        for name,p in model.module.named_parameters():
                            if name in expected['grads']:
                                torch.testing.assert_close(p.grad,expected['grads'][name],rtol=1e-10,atol=1e-10)
                    optimizer.step()
                    if repeat==0:
                        for name,value_ in model.module.state_dict().items():
                            torch.testing.assert_close(value_,expected['model'][name],rtol=1e-10,atol=1e-10)
                rows.append(dict(mode=mode,batch_dice=batch_dice,value_and_gradient_and_update=True))
        # Simulate a gradient overflow on ONE rank with a real enabled CPU
        # scaler. All ranks must skip and reset growth identically, then resume.
        seed(17);model=DDP(nn.Linear(2,1).double());optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.99)
        scaler=torch.amp.GradScaler('cpu',init_scale=16,growth_interval=2)
        before={k:v.clone() for k,v in model.module.state_dict().items()}
        optimizer.zero_grad();scaler.scale(model(torch.ones(1,2,dtype=torch.float64)).sum()).backward()
        if rank==world-1:next(model.parameters()).grad.fill_(float('inf'))
        scaler.unscale_(optimizer)
        assert synchronize_overflow(optimizer,scaler,torch.device('cpu'))
        scaler.step(optimizer);scaler.update()
        for k,v in model.module.state_dict().items():assert torch.equal(v,before[k])
        assert scaler.get_scale()==8 and scaler.state_dict()['_growth_tracker']==0
        signature={'identity':'synthetic-ddp-only'}
        state=dict(epoch=0,phase='train',step=1,attempts=1)
        members=[f'synthetic-{i}' for i in range(world)]
        seed(100+rank)
        commit(folder/'checkpoint',model,optimizer,scaler,state,signature,torch.device('cpu'),members)
        expected_rng=(random.random(),float(np.random.rand()),torch.rand(4))
        # Continuous next successful update and its momentum/scaler.
        optimizer.zero_grad();scaler.scale(model(torch.ones(1,2,dtype=torch.float64)).sum()).backward()
        scaler.unscale_(optimizer);assert not synchronize_overflow(optimizer,scaler,torch.device('cpu'))
        scaler.step(optimizer);scaler.update()
        expected_model={k:v.clone() for k,v in model.module.state_dict().items()}
        expected_scaler=scaler.state_dict()
        expected_momentum=[s['momentum_buffer'].clone() for s in optimizer.state.values()]
        payload=read_checkpoint(folder/'checkpoint',signature)
        seed(999)
        restore_checkpoint(payload,model.module,optimizer,scaler,torch.device('cpu'))
        restore_rank_rng(payload,members,rank,torch.device('cpu'),42)
        actual_rng=(random.random(),float(np.random.rand()),torch.rand(4))
        assert expected_rng[:2]==actual_rng[:2] and torch.equal(expected_rng[2],actual_rng[2])
        try:restore_rank_rng(payload,members+['new'],rank,torch.device('cpu'),42)
        except ValueError:pass
        else:raise AssertionError('Partial-epoch membership change accepted')
        optimizer.zero_grad();scaler.scale(model(torch.ones(1,2,dtype=torch.float64)).sum()).backward()
        scaler.unscale_(optimizer);assert not synchronize_overflow(optimizer,scaler,torch.device('cpu'))
        scaler.step(optimizer);scaler.update()
        for k,v in model.module.state_dict().items():assert torch.equal(v,expected_model[k])
        assert scaler.state_dict()==expected_scaler
        for value,wanted in zip(optimizer.state.values(),expected_momentum):assert torch.equal(value['momentum_buffer'],wanted)
        (folder/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,layout=layout(4,world),
            numerical=rows,overflow_consensus=True,supply_layout_agreed=decision['workers'],
            per_rank_rng_exact=True,resume_update_exact=True,
            partial_membership_refused=True)),encoding='utf-8')
    finally:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--world-sizes',default='1,2,3,4')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for world in map(int,args.world_sizes.split(',')):
        folder=args.output/f'world-{world}';folder.mkdir()
        for mode in ('complete','mixed','partial','ignore'):
            for batch_dice in (True,False):torch.save(reference(mode,batch_dice),folder/f'ref-{mode}-{batch_dice}.pt')
        mp.spawn(worker,args=(world,str(folder)),nprocs=world,join=True)
        ranks=[json.loads((folder/f'rank-{rank}.json').read_text()) for rank in range(world)]
        results.append(dict(world_size=world,layout=layout(4,world),ranks=ranks))
        print(json.dumps(dict(world_size=world,status='passed')),flush=True)
    receipt=dict(scope='CPU Gloo real DDP, synthetic; not Linux CUDA/NCCL throughput',results=results)
    (args.output/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__':main()
