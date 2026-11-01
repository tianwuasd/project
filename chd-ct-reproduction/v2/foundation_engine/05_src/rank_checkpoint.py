"""All-rank checkpoint state, stored inside the same atomic weight generation."""
import hashlib
import os


def loader_state(loader):
    from training_timing import TimedIterator
    if isinstance(loader,TimedIterator):loader=loader.source
    return loader.state_dict() if hasattr(loader,'state_dict') else None


def capture(trainer):
    from ddp_runtime import capture_rank_state,gather_rank_states
    state=dict(gpu_uuid=os.environ.get('VISTA_RANK_GPU_UUID'),
        rng=capture_rank_state(trainer.local_rank if trainer.device.type=='cuda' else None),
        loaders={name:loader_state(getattr(trainer,name,None))
                 for name in ['dataloader_train','dataloader_val']})
    return dict(version=1,world_size=trainer._vista_world_size,
        global_batch_size=trainer._vista_global_batch,run_kind=trainer._vista_run_kind,
        states=gather_rank_states(state))


def restore(trainer,state,*,loaders=False):
    """A membership change starts new per-UUID streams at the next full epoch.

    Existing devices retain their rank-local RNG. New devices get an explicit,
    deterministic seed; this is not advertised as a bit-identical larger-batch run.
    """
    import random
    import numpy as np
    import torch
    from ddp_runtime import restore_rank_state
    if state.get('version')!=1 or state.get('run_kind')!=trainer._vista_run_kind:
        raise ValueError('检查点版本或正式/测评身份不符')
    uuid=os.environ.get('VISTA_RANK_GPU_UUID')
    found=[s for s in state['states'] if s['gpu_uuid']==uuid]
    if len(found)>1:raise ValueError('检查点中GPU身份重复')
    if found:
        restore_rank_state(found[0]['rng'],trainer.local_rank if trainer.device.type=='cuda' else None)
        if loaders:
            for name,value in found[0]['loaders'].items():
                loader=getattr(trainer,name,None)
                # Full-epoch checkpoints move to the next epoch. Mid-epoch
                # assessment roundtrips retain their consumed-batch position.
                if value and value['epoch']==trainer.current_epoch:
                    value=dict(value,rank=trainer.local_rank)
                    loader.load_state_dict(value)
    else:
        seed=int.from_bytes(hashlib.sha256(f'{uuid}:{trainer.current_epoch}:vista'.encode()).digest()[:4],'big')
        random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
        if trainer.device.type=='cuda':torch.cuda.manual_seed(seed)
        trainer.print_to_log_file(f'新加入GPU {uuid}：epoch={trainer.current_epoch}，独立随机流seed={seed}')
