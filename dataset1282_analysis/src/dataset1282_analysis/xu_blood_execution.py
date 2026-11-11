import os

import torch

from .xu_resources import ResourceUnavailable


def execution_policy(encoder, device, temporal):
    mode = os.environ.get('XU_BLOOD_EXECUTION', 'auto')
    if mode not in ('auto', 'legacy'):
        raise ValueError('XU_BLOOD_EXECUTION 必须为 auto 或 legacy')
    selected = torch.device(device)
    width = encoder.width
    plane = 512 * 512 * width * 4
    parameters = sum(p.numel() for p in encoder.parameters())
    budget = None
    if selected.type == 'cuda':
        import json
        free, total = torch.cuda.mem_get_info(selected)
        evidence = json.loads(os.environ.get('XU_CUDA_ADMISSION', '{}'))
        maximum = evidence.get('allocator_budget_bytes', min(int(total * .8), total - 1024**3))
        budget = min(maximum, free + torch.cuda.memory_reserved(selected))
    frames = 1
    encoder_checkpoint = True
    sequence_checkpoint = True
    if mode == 'auto':
        frames = 7 if budget is None else max(1, min(7, int((budget - 4*parameters - 7*plane) // (12*plane))))
        if budget is not None:
            encoder_checkpoint = budget < 1.15 * (16*parameters + 64*4*plane)
            sequence_checkpoint = budget < 1.15 * (4*parameters + 320*plane)
    requested_frames = int(os.environ.get('XU_BLOOD_FRAME_BATCH', '7'))
    if not 1 <= requested_frames <= 7:
        raise ValueError('XU_BLOOD_FRAME_BATCH 必须在 1 到 7 之间')
    frames = min(frames, requested_frames)
    checkpoint_request = os.environ.get('XU_BLOOD_CHECKPOINT', 'auto')
    if checkpoint_request not in ('auto', 'on', 'off'):
        raise ValueError('XU_BLOOD_CHECKPOINT 必须为 auto、on 或 off')
    if checkpoint_request == 'on':
        encoder_checkpoint = sequence_checkpoint = True
    elif checkpoint_request == 'off':
        needed = 1.15 * ((4*parameters + 320*plane) if temporal else (16*parameters + 64*4*plane))
        if budget is not None and budget < needed:
            raise ResourceUnavailable('显存预算不足以关闭 blood 激活重计算')
        encoder_checkpoint = sequence_checkpoint = False
    return dict(mode=mode, frame_batch_requested=requested_frames, frame_batch_applied=frames,
                allocator_budget_bytes=budget, encoder_checkpointing=encoder_checkpoint,
                sequence_checkpointing=sequence_checkpoint,
                development_feature_cache=mode == 'auto')
