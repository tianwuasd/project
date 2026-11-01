"""Actual project/upstream checkpoint methods on CPU DDP, no trainer CUDA ctor."""
import os
from pathlib import Path
import random


class Logger:
    def get_checkpoint(self):return {'tested':'actual_cpu_ddp'}
    def load_checkpoint(self,state):self.value=state


def run(rank,world,args):
    os.environ['WHOLE_HEART_WORKSPACE']=args['workspace']
    for name in ['nnUNet_raw','nnUNet_preprocessed','nnUNet_results']:
        os.environ[name]=str(Path(args['workspace'])/name)
    os.environ['VISTA_RANK_GPU_UUID']=f'GPU-cpu-test-{rank}'
    import numpy as np
    import torch
    import torch.distributed as dist
    from project_trainers import BaselineNNUNet
    from batch_pool import EpochBatchIterator
    from batch_transfer import BatchTransfer
    torch.set_num_threads(1)
    # The pinned upstream constructor forces CUDA whenever a group exists.
    # Bypass that constructor only for this explicit CPU checkpoint API test.
    trainer=BaselineNNUNet.__new__(BaselineNNUNet)
    trainer.device=torch.device('cpu');trainer.is_ddp=True;trainer.local_rank=rank
    trainer.was_initialized=True;trainer.disable_checkpointing=False;trainer.grad_scaler=None
    trainer._vista_world_size=world;trainer._vista_global_batch=world*2;trainer._vista_run_kind='assessment'
    trainer.current_epoch=0;trainer.my_init_kwargs={};trainer._best_ema=.2
    trainer.inference_allowed_mirroring_axes=None;trainer.logger=Logger()
    torch.manual_seed(123)
    trainer.network=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4,2))
    trainer.optimizer=torch.optim.SGD(trainer.network.parameters(),lr=.02,momentum=.9)
    trainer.loss=torch.nn.MSELoss()
    trainer.dataloader_train=BatchTransfer(EpochBatchIterator(CheckpointBatches(),workers=0,
        rank=rank,phase_steps=3),trainer.device)
    trainer.dataloader_val=None
    def step():
        trainer.optimizer.zero_grad()
        x=torch.ones(2,4)*(rank+1)
        loss=trainer.network(x).square().mean();loss.backward();trainer.optimizer.step()
    def draws():return [random.random(),float(np.random.rand()),torch.rand(4).tolist()]
    random.seed(30+rank);np.random.seed(40+rank);torch.manual_seed(50+rank)
    trainer.train_step(next(trainer.dataloader_train))
    unconsumed=next(trainer.dataloader_train)['data'].clone()
    checkpoint=Path(args['workspace'])/'atomic.pth'
    from unittest.mock import patch
    with patch.object(torch,'save',wraps=torch.save) as save:
        trainer.save_checkpoint(str(checkpoint))
        # Torch object collectives also serialize tensor storages into BytesIO;
        # count the actual checkpoint file writes, not those transport packets.
        writes=[call for call in save.call_args_list
                if getattr(call.args[1],'name',None)==str(checkpoint)+'.tmp']
        assert len(writes)==(1 if rank==0 else 0)
    expected_rng=draws()
    expected={k:v.clone() for k,v in trainer.network.module.state_dict().items()}
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['vista_rank_states']['world_size']==world
    assert len(saved['vista_rank_states']['states'])==world
    saved_source=saved['vista_rank_states']['states'][rank]['loaders']['dataloader_train']
    assert saved_source['next_index']==1
    assert trainer.dataloader_train.source.state_dict()['next_index']==1
    with torch.no_grad():next(trainer.network.parameters()).add_(rank+2)
    trainer.optimizer.state.clear()
    trainer.load_checkpoint(str(checkpoint))
    assert trainer.rng_state_restored and trainer.current_epoch==1
    assert trainer.optimizer.state and expected_rng==draws()
    assert all(torch.equal(expected[k],v) for k,v in trainer.network.module.state_dict().items())
    assert saved['vista_rank_states']['states'][rank]['gpu_uuid']==f'GPU-cpu-test-{rank}'
    trainer.dataloader_train.load_state_dict(saved_source)
    replay=next(trainer.dataloader_train)
    assert torch.equal(unconsumed,replay['data'])
    trainer.train_step(replay)
    trainer.dataloader_train.close()
    step()
    assert all(torch.isfinite(p).all() for p in trainer.network.parameters())
    assert not torch.cuda.is_initialized()
    return dict(status='passed',rank=rank,world_size=world,actual_project_checkpoint=True,
        optimizer_restored=True,rank_rng_equal=True,weights_equal=True,resumed_update=True,cuda_initialized=False,
        atomic_save_calls=1 if rank==0 else 0,logical_source_cursor=1,unconsumed_batch_replayed=True)


class CheckpointBatches:
    def __iter__(self):return self
    def __next__(self):
        import torch
        return dict(data=torch.rand(2,4),target=torch.zeros(2,2))


def compile_fallback(rank,world,args):
    """One lazy compile failure, peer's real autograd, followed by real DDP update."""
    import types
    import torch
    from unittest.mock import patch
    from training_execution import apply_execution,preflight_compile
    from checkpoint_rng import capture
    torch.set_num_threads(1);torch.manual_seed(123)
    network=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4,2))
    trainer=types.SimpleNamespace(network=network,is_ddp=True,device=torch.device('cpu'),
        print_to_log_file=lambda *args:None)
    optimizer=torch.optim.SGD(network.parameters(),lr=.02,momentum=.9)
    state={k:v.clone() for k,v in network.state_dict().items()};rng=capture()
    reports=[]
    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx,value):return value.clone()
        @staticmethod
        def backward(ctx,gradient):return gradient*2
    for fault in ('exception','bias','nan','structure','gradient'):
        report=apply_execution(torch,{'mode':'performance','compile':True},main_threads=1)
        def compiler(module,**kwargs):
            if rank==0:
                def failure(data):
                    if fault=='exception':
                        torch.rand(2)
                        raise RuntimeError('asymmetric lazy compile failure')
                    value=module(data)
                    if fault=='bias':return value*2
                    if fault=='nan':return value*float('nan')
                    if fault=='structure':return [value]
                    return WrongGradient.apply(value)
                return failure
            return module
        with patch.object(torch,'compile',side_effect=compiler):
            preflight_compile(trainer,dict(data=torch.ones(2,4)),report)
        assert not report['compile_applied'] and 'rank 0' in report['fallback_reason']
        assert report['compatibility']['local_passed']==(rank==1)
        assert not report['compatibility']['passed']
        assert torch.equal(torch.get_rng_state(),rng['torch'])
        assert all(torch.equal(network.state_dict()[k],v) for k,v in state.items())
        assert not optimizer.state and all(p.grad is None for p in network.parameters())
        reports.append(dict(fault=fault,execution=report))
    optimizer.zero_grad();loss=network(torch.ones(2,4)*(rank+1)).square().mean()
    loss.backward();optimizer.step()
    assert optimizer.state and not torch.cuda.is_initialized()
    return dict(status='passed',rank=rank,asymmetric_failure_group_eager=True,
        real_ddp_autograd_update=True,preflight_weights_rng_optimizer_preserved=True,
        cuda_initialized=False,compile_is_injected_control_test=True,compatibility_cases=reports)


class MembershipBatches:
    """Tiny deterministic data source for the actual project's epoch iterator."""
    def __iter__(self):return self
    def __next__(self):
        import torch
        return torch.rand(2,4)


def membership(rank,world,args):
    os.environ['WHOLE_HEART_WORKSPACE']=args['workspace']
    for name in ['nnUNet_raw','nnUNet_preprocessed','nnUNet_results']:
        os.environ[name]=str(Path(args['workspace'])/name)
    uuid=args['gpu_uuids'][rank];os.environ['VISTA_RANK_GPU_UUID']=uuid
    import hashlib
    import numpy as np
    import torch
    from project_trainers import BaselineNNUNet
    from batch_pool import EpochBatchIterator
    from rank_checkpoint import restore
    from ddp_runtime import restore_rank_state
    torch.set_num_threads(1)
    # Same explicit CPU checkpoint-API adaptation as run(); no ctor/GPU claim.
    trainer=BaselineNNUNet.__new__(BaselineNNUNet)
    trainer.device=torch.device('cpu');trainer.is_ddp=True;trainer.local_rank=rank
    trainer.was_initialized=True;trainer.disable_checkpointing=False;trainer.grad_scaler=None
    trainer._vista_world_size=world;trainer._vista_global_batch=world*2;trainer._vista_run_kind='formal'
    trainer.current_epoch=0;trainer.my_init_kwargs={};trainer._best_ema=.2
    trainer.inference_allowed_mirroring_axes=None;trainer.logger=Logger()
    trainer.print_to_log_file=lambda *a,**k:None
    torch.manual_seed(123)
    trainer.network=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4,2))
    trainer.optimizer=torch.optim.SGD(trainer.network.parameters(),lr=.02,momentum=.9)
    def loaders(epoch):
        return [EpochBatchIterator(MembershipBatches(),workers=0,seed=0,epoch=epoch,rank=rank,stage=stage)
            for stage in ['train','validation']]
    trainer.dataloader_train,trainer.dataloader_val=loaders(0)
    def step(batch):
        trainer.optimizer.zero_grad();loss=trainer.network(batch).square().mean()
        loss.backward();trainer.optimizer.step();assert torch.isfinite(loss)
    def draws():return [random.random(),float(np.random.rand()),torch.rand(4).tolist()]
    checkpoint=Path(args['workspace'])/'membership.pth';mode='initial';seed=None
    try:
        if args['phase']=='save':
            random.seed(30+rank);np.random.seed(40+rank);torch.manual_seed(50+rank)
            for _ in range(250):step(next(trainer.dataloader_train))
            for _ in range(50):next(trainer.dataloader_val)
            trainer.save_checkpoint(str(checkpoint))
            return dict(status='passed',rank=rank,gpu_uuid=uuid,world_size=world,
                train_updates=250,validation_batches=50,checkpoint=str(checkpoint),constructor_bypassed=True)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        old=next((row for row in saved['vista_rank_states']['states'] if row['gpu_uuid']==uuid),None)
        assert saved['current_epoch']==1 and len(saved['vista_rank_states']['states'])==2
        if old:
            restore_rank_state(old['rng']);expected=draws();mode='retained_uuid_rng'
        else:
            seed=int.from_bytes(hashlib.sha256(f'{uuid}:1:vista'.encode()).digest()[:4],'big')
            expected=[random.Random(seed).random(),float(np.random.RandomState(seed).rand()),
                torch.rand(4,generator=torch.Generator().manual_seed(seed)).tolist()]
            mode='new_uuid_deterministic_seed'
        trainer.load_checkpoint(str(checkpoint))
        assert draws()==expected and trainer.current_epoch==1 and trainer.optimizer.state
        assert all(torch.equal(v,trainer.network.module.state_dict()[key]) for key,v in saved['network_weights'].items())
        for loader in [trainer.dataloader_train,trainer.dataloader_val]:loader.close()
        trainer.dataloader_train,trainer.dataloader_val=loaders(trainer.current_epoch)
        restore(trainer,saved['vista_rank_states'],loaders=True)
        assert draws()==expected
        states=[loader.state_dict() for loader in [trainer.dataloader_train,trainer.dataloader_val]]
        assert all(s['epoch']==1 and s['rank']==rank and s['next_index']==0 for s in states)
        replay=EpochBatchIterator(MembershipBatches(),workers=0,seed=0,epoch=1,rank=rank,stage='train')
        try:
            batch=next(trainer.dataloader_train);assert torch.equal(batch,next(replay))
        finally:replay.close()
        step(batch)
        assert not torch.cuda.is_initialized()
        return dict(status='passed',rank=rank,gpu_uuid=uuid,world_size=world,restore_mode=mode,seed=seed,
            rank_rng_equal=True,optimizer_restored=True,weights_equal=True,new_loader_epoch=1,new_loader_rank=rank,
            new_epoch_batch_replay_equal=True,resumed_update=True,cuda_initialized=False,constructor_bypassed=True)
    finally:
        for loader in [trainer.dataloader_train,trainer.dataloader_val]:loader.close()
