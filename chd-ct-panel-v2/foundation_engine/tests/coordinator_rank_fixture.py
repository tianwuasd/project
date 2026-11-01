"""Inert CPU Gloo integration for real all-rank rolling reconfiguration."""
import os
from pathlib import Path
import random

class Source:
    replayable=True
    def __iter__(self):return self
    def __next__(self):
        import torch
        return {'data':torch.tensor([[random.random()]]),'target':torch.zeros(1,1)}


def run(rank,world,args):
    import torch
    import torch.distributed as dist
    from batch_pool import EpochBatchIterator
    from batch_transfer import BatchTransfer
    from training_feedback import RollingFeedback
    torch.set_num_threads(1)
    model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1,1))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.99)
    class Trainer:pass
    trainer=Trainer();trainer.current_epoch=0;trainer._vista_gpu_uuids=args['gpu_uuids']
    trainer.dataloader_train=BatchTransfer(EpochBatchIterator(Source(),workers=1,rank=rank,phase_steps=45,prefetch=2,max_shared_bytes=256),torch.device('cpu'))
    trainer.dataloader_val=BatchTransfer(EpochBatchIterator(Source(),workers=0,rank=rank,phase_steps=0,stage='validation'),torch.device('cpu'))
    clock=[0.]
    feedback=RollingFeedback(args['resource'],rank,world,args['output'],clock=lambda:clock[0])
    try:
        for step in range(40):
            batch=next(trainer.dataloader_train)
            optimizer.zero_grad();loss=model(batch['data']).square().sum();loss.backward();optimizer.step()
            trainer.dataloader_train.mark_consumed()
            clock[0]+=2
            if args.get('fail') and rank==0 and step==19:raise ValueError('injected worker-boundary failure')
            feedback.observe(dict(batch_wait_seconds=1.5),trainer,dist)
        assert feedback.resource['rank_settings'][rank]['train_workers']==2
        assert trainer.dataloader_train.consumed==40
        actual=next(trainer.dataloader_train)['data'].clone()
        reference=EpochBatchIterator(Source(),workers=0,rank=rank,phase_steps=45)
        reference.set_epoch(0,start_index=40)
        expected=next(reference)['data'];reference.close()
        assert torch.equal(actual,expected)
        return dict(status='passed',rank=rank,workers=2,consumed=40,replayed_equal=True,
                    cuda_initialized=torch.cuda.is_initialized(),feedback_events=len(feedback.events))
    finally:
        trainer.dataloader_train.close();trainer.dataloader_val.close()
