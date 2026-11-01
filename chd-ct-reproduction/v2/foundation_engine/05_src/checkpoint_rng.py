"""Checkpoint global RNG streams; serial augmentation remains the execution contract."""
import random


def capture():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore(value):
    import numpy as np
    import torch
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'].cpu())
    if value.get('cuda') is not None:torch.cuda.set_rng_state_all([s.cpu() for s in value['cuda']])
