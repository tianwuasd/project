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
    trainer.dataloader_train=None;trainer.dataloader_val=None
    def step():
        trainer.optimizer.zero_grad()
        x=torch.ones(2,4)*(rank+1)
        loss=trainer.network(x).square().mean();loss.backward();trainer.optimizer.step()
    def draws():return [random.random(),float(np.random.rand()),torch.rand(4).tolist()]
    random.seed(30+rank);np.random.seed(40+rank);torch.manual_seed(50+rank)
    step()
    checkpoint=Path(args['workspace'])/'atomic.pth'
    trainer.save_checkpoint(str(checkpoint))
    expected_rng=draws()
    expected={k:v.clone() for k,v in trainer.network.module.state_dict().items()}
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['vista_rank_states']['world_size']==world
    assert len(saved['vista_rank_states']['states'])==world
    with torch.no_grad():next(trainer.network.parameters()).add_(rank+2)
    trainer.optimizer.state.clear()
    trainer.load_checkpoint(str(checkpoint))
    assert trainer.rng_state_restored and trainer.current_epoch==1
    assert trainer.optimizer.state and expected_rng==draws()
    assert all(torch.equal(expected[k],v) for k,v in trainer.network.module.state_dict().items())
    assert saved['vista_rank_states']['states'][rank]['gpu_uuid']==f'GPU-cpu-test-{rank}'
    step()
    assert all(torch.isfinite(p).all() for p in trainer.network.parameters())
    assert not torch.cuda.is_initialized()
    return dict(status='passed',rank=rank,world_size=world,actual_project_checkpoint=True,
        optimizer_restored=True,rank_rng_equal=True,weights_equal=True,resumed_update=True,cuda_initialized=False)
