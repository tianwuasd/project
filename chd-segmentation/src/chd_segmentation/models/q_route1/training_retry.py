from contextlib import nullcontext
from copy import deepcopy
import gc
import random
from time import perf_counter, sleep

import numpy as np
import torch

from ...observability.events import event
from .training_inputs import training_plan


def random_state(bundle, rng, device):
    return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
                data=deepcopy(rng.bit_generator.state),extra=deepcopy(bundle.extra_state()),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_random(state, bundle, rng, device):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    rng.bit_generator.state = deepcopy(state['data'])
    bundle.restore_extra(deepcopy(state['extra']))
    if device.type=='cuda':
        torch.cuda.set_rng_state(state['cuda'],device)


def backward_sample(bundle, entry, rng, update, memory, device, settings):
    reservation = nullcontext()
    if bundle.stage=='localizer':
        plan = training_plan(entry['label'].shape,entry['processed_spacing'],settings)
        reservation = memory.admit(plan['retained_bytes']+plan['backward_bytes'],label='q_localizer_training_case')
    try:
        with reservation as ticket:
            if bundle.stage=='localizer' and ticket is None:
                raise InterruptedError('定位器病例等待已取消')
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bundle.config['precision']=='bf16'):
                loss = bundle.loss(entry,rng,update)
            if not torch.isfinite(loss):
                raise FloatingPointError('模型训练损失非有限')
            (loss/bundle.config['accumulation']).backward()
            value = float(loss.detach())/bundle.config['accumulation']
            if device.type=='cuda' and bundle.stage=='localizer':
                torch.cuda.synchronize(device)
            return value, None
    except torch.cuda.OutOfMemoryError as error:
        if device.type!='cuda' or bundle.stage!='localizer':
            raise
        # 仅显式恢复CUDA显存不足；返回后释放失败调用的计算图和异常引用。
        return 0., dict(case_id=entry['case_id'],error=str(error),
                       free_bytes=torch.cuda.mem_get_info(device)[0],
                       allocated_bytes=torch.cuda.memory_allocated(device),
                       reserved_bytes=torch.cuda.memory_reserved(device))


def accumulate(bundle, optimizer, dataset, names, rng, update, rank, allocation, gate, device, memory, common):
    from ..execution import all_states

    root = gate.root
    state = random_state(bundle,rng,device) if bundle.stage=='localizer' else None
    replay, waiting_started = 0, None
    settings = bundle.config['execution']
    block = settings.get('low_vram_block_bytes',128*2**20)
    if type(block) is not int or block<1:
        raise ValueError('execution.low_vram_block_bytes需要正整数')
    bundle.low_vram_block_bytes = getattr(bundle,'low_vram_block_bytes',block)
    while True:
        if replay:
            restore_random(state,bundle,rng,device)
        gate.start(rank,f'q-update:{update}:replay:{replay}',allocation['shares'])
        dataset.mode, dataset.position = 'training', 0
        optimizer.zero_grad(set_to_none=True)
        loss_sum, failure = 0., None
        for micro in range(allocation['shares'][rank]):
            began = perf_counter()
            entry = bundle.sample(dataset,names,rng)
            if failure is None:
                value, failure = backward_sample(bundle,entry,rng,update,memory,device,common['resources'])
                loss_sum += value
            if failure is not None:
                optimizer.zero_grad(set_to_none=True)
            event(root,'model_sample_consumed',rank=rank,batch=update,position=micro,replay=replay,
                  case_id=entry['case_id'],seconds=perf_counter()-began,computed=failure is None,
                  cuda_completion_checked_at='sample_boundary' if bundle.stage=='localizer' else 'batch_boundary')
            del entry
        if failure is not None:
            gc.collect()
            torch.cuda.empty_cache()
        # 所有rank先在CPU端结束当前供数轮次，再进行小型状态通信。
        gate.finish(rank,f'q-update:{update}:replay:{replay}',device if bundle.stage!='localizer' else None)
        if bundle.stage!='localizer':
            return loss_sum
        failures = all_states(failure)
        if not any(failures):
            return loss_sum
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
        was_low = getattr(bundle,'low_vram',False)
        bundle.low_vram = True
        if was_low:
            bundle.low_vram_block_bytes = max(1,bundle.low_vram_block_bytes//2)
        replay += 1
        event(root,'q_low_vram_retry',rank=rank,update=update,replay=replay,failures=failures,
              block_bytes=bundle.low_vram_block_bytes,gradients_discarded=True,
              random_state_replayed=True,global_batch=sum(allocation['shares']),
              mode='gpu_tiled_convolution_cpu_activations_global_norm')
        if waiting_started is None:
            waiting_started = perf_counter()
        if was_low:
            poll = common['resources']['poll_seconds']
            until = perf_counter()+poll
            while perf_counter()<until:
                gate.ready.require_active(waiting_started)
                sleep(min(1.,max(0.,until-perf_counter())))
        gate.ready.wait(rank,f'q-vram-retry:{update}:{replay}')
