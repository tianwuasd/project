"""Real CPU GradScaler regression for the pinned trainer numerical path."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import sys
import tempfile
import types
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import torch
from project_trainers import BaselineNNUNet


def trainer():
    value=object.__new__(BaselineNNUNet)
    value.network=torch.nn.Linear(1,1,bias=False)
    value.network.weight.data.fill_(1)
    value.optimizer=torch.optim.SGD(value.network.parameters(),lr=.01,momentum=.9)
    value.grad_scaler=torch.amp.GradScaler('cpu',growth_interval=2)
    value.device=torch.device('cpu');value.is_ddp=False
    value.loss=lambda output,target:output.sum()
    value.print_to_log_file=lambda *args:None
    return value


class AMPUpdates(unittest.TestCase):
    def test_overflow_consumes_attempt_backs_off_then_recovers(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        value=trainer();reference=trainer()
        for data in (1.,1e34,1.):
            batch=dict(data=torch.tensor([[data]]),target=torch.zeros(1,1))
            value._finite_train_step(batch);nnUNetTrainer.train_step(reference,batch)
            torch.testing.assert_close(value.network.weight,reference.network.weight,rtol=0,atol=0)
            self.assertEqual(value.grad_scaler.state_dict(),reference.grad_scaler.state_dict())
            if data==1e34:
                self.assertFalse(value._vista_last_optimizer_step)
                self.assertEqual(value.grad_scaler.get_scale(),32768)
                self.assertEqual(value.grad_scaler.state_dict()['_growth_tracker'],0)
                self.assertEqual(value.optimizer.state[value.network.weight]['momentum_buffer'].item(),1.)
        self.assertTrue(value._vista_last_optimizer_step)
        self.assertEqual(value._vista_update_progress,dict(attempted=3,successful=2,amp_skipped=1))
        self.assertFalse(torch.cuda.is_initialized())

    def test_overflow_cursor_checkpoint_counts_and_scaler_restore(self):
        value=trainer();consumed=[]
        value.dataloader_train=types.SimpleNamespace(mark_consumed=lambda:consumed.append(1))
        value.local_rank=0;value.disable_checkpointing=False;value.was_initialized=True
        value.logger=types.SimpleNamespace(get_checkpoint=lambda:{},load_checkpoint=lambda state:None)
        value._best_ema=None;value.current_epoch=0;value.my_init_kwargs={}
        value.inference_allowed_mirroring_axes=None
        value.train_step(dict(data=torch.tensor([[1e34]]),target=torch.zeros(1,1)))
        self.assertEqual(consumed,[1]);self.assertFalse(value.optimizer.state)
        from training_execution import memory_facts
        self.assertFalse(memory_facts(value)['optimizer_initialized'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pth';value.save_checkpoint(str(path))
            checkpoint=torch.load(path,weights_only=False)
            self.assertEqual(checkpoint['vista_update_progress'],dict(attempted=1,successful=0,amp_skipped=1))
            value._vista_update_progress={};value.grad_scaler.update(new_scale=8.)
            value.load_checkpoint(str(path))
            self.assertEqual(value.grad_scaler.get_scale(),32768)
            self.assertEqual(value._vista_update_progress,checkpoint['vista_update_progress'])
            value.train_step(dict(data=torch.ones(1,1),target=torch.zeros(1,1)))
            self.assertEqual(value._vista_update_progress,dict(attempted=2,successful=1,amp_skipped=1))
            self.assertTrue(memory_facts(value)['optimizer_initialized'])
            legacy=dict(checkpoint);legacy.pop('vista_update_progress');legacy.pop('vista_update_progress_origin_known')
            value.load_checkpoint(legacy)
            self.assertFalse(value._vista_progress_origin_known)
            value.save_checkpoint(str(path));value.load_checkpoint(str(path))
            self.assertFalse(value._vista_progress_origin_known)


def amp_rank(rank,world,args):
    """Actual Gloo: inject nonfinite gradient only AFTER DDP reduction on rank0."""
    import torch.distributed as dist
    from unittest.mock import patch
    torch.set_num_threads(1)
    value=trainer();value.network=torch.nn.parallel.DistributedDataParallel(value.network)
    value.is_ddp=True
    consumed=[];value.dataloader_train=types.SimpleNamespace(mark_consumed=lambda:consumed.append(1))
    batch=dict(data=torch.ones(1,1),target=torch.zeros(1,1))
    value.train_step(batch)
    weight=value.network.module.weight.detach().clone()
    momentum=value.optimizer.state[value.network.module.weight]['momentum_buffer'].clone()
    unscale=value.grad_scaler.unscale_
    def asymmetric(optimizer):
        unscale(optimizer)
        if rank==0:value.network.module.weight.grad.fill_(float('inf'))
    with patch.object(value.grad_scaler,'unscale_',side_effect=asymmetric):value.train_step(batch)
    assert not value._vista_last_optimizer_step and len(consumed)==2
    assert torch.equal(weight,value.network.module.weight)
    assert torch.equal(momentum,value.optimizer.state[value.network.module.weight]['momentum_buffer'])
    assert value.grad_scaler.get_scale()==32768 and value.grad_scaler.state_dict()['_growth_tracker']==0
    value.train_step(batch)
    state=dict(weight=value.network.module.weight.item(),momentum=value.optimizer.state[value.network.module.weight]['momentum_buffer'].item(),
        scaler=value.grad_scaler.state_dict(),updates=value._vista_update_progress,consumed=len(consumed))
    states=[None]*world;dist.all_gather_object(states,state)
    assert all(row==state for row in states)
    assert state['updates']==dict(attempted=3,successful=2,amp_skipped=1)
    assert not torch.cuda.is_initialized()
    return dict(status='passed',rank=rank,asymmetric_overflow=True,all_rank_state_equal=True,cuda_initialized=False,**state)


if __name__=='__main__':unittest.main()
