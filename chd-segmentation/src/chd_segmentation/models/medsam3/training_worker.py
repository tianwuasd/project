import os
from pathlib import Path
import sys


def execute(payload, handle, gpus, rank, world, coordinator, port):
    root = Path(payload['config']['training']['run_dir'])
    stream = (root/('training.log' if world == 1 else f'training.rank{rank}.log')).open('a', encoding='utf-8')
    sys.stdout = sys.stderr = stream
    cache = root/'runtime-cache'/f'rank{rank}'
    temporary = cache/'temporary'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update({key:str(cache/name) for key,name in {
        'TRITON_CACHE_DIR':'triton', 'TORCHINDUCTOR_CACHE_DIR':'inductor',
        'TORCH_HOME':'torch', 'HF_HOME':'huggingface', 'MPLCONFIGDIR':'matplotlib',
        'XDG_CACHE_HOME':'xdg', 'CUDA_CACHE_PATH':'cuda',
        'TMPDIR':'temporary', 'TMP':'temporary', 'TEMP':'temporary'}.items()})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus[rank]['uuid']
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),WORLD_SIZE=str(world))
    coordinator.register(rank)
    train(payload, handle, rank, world, coordinator, gpus)


def train(payload, handle, rank, world, coordinator, gpus):
    import gc
    import math
    import random
    from time import time, perf_counter
    import numpy as np
    import torch
    import torch.distributed as dist
    from .environment import verify_worker_environment
    from .adapter import MedSAM3Adapter
    from .training_data import datapoint, stream
    from ...acceleration.runtime import attach
    from ...artifacts.store import write_json
    from ...observability.events import event
    from ...acceleration.distributed import all_states,synchronize_gradients,distributed_backend
    from .distributed import rank_indices,require_topology,rng_state,restore_rng,validation_rounds,resumed_rng
    from ...training.checkpoint import verify_optimizer_device,tensors_equal
    from ...artifacts.store import digest

    verify_worker_environment(payload['environment'])
    config, training = payload['config'], payload['config']['training']
    root, device = Path(training['run_dir']), torch.device(config['device'])
    client = attach(handle)
    torch.set_num_threads(config['resources']['cpu_threads']//world-training['workers'])
    seed = (training['seed']+rank) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    coordinator.wait(rank,-1,'startup_ready')
    if device.type == 'cuda':
        actual = str(torch.cuda.get_device_properties(0).uuid)
        actual = actual if actual.startswith('GPU-') else 'GPU-'+actual
        if actual.lower() != gpus[rank]['uuid'].lower():
            raise ValueError('医学LoRA实际GPU与获准UUID不同')
    if world > 1:
        from datetime import timedelta
        dist.init_process_group(distributed_backend(device,config['resources'].get('distributed_backend')),
            rank=rank,world_size=world,timeout=timedelta(seconds=config['resources']['collective_timeout_seconds']),
            **({'device_id':torch.device('cuda',0)} if device.type == 'cuda' else {}))
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        reserve = max(2**30, config['resources']['vram_reserve_bytes'])
        cap = min(training['vram_limit_bytes'], free-reserve)
        if config.get('capacity_probe_bytes', 0):
            cap = min(cap, config['capacity_probe_bytes'])
        if cap <= 0 or config['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('显存余量不足或不支持请求的CUDA BF16')
        torch.cuda.set_per_process_memory_fraction(cap/total)
        torch.cuda.reset_peak_memory_stats()
    state = dict(step=0, best_loss=float('inf'), validation_pending=False)

    def combine(values, maximum=False):
        tensor = torch.tensor(values,dtype=torch.float64,device=device)
        if world > 1:
            dist.all_reduce(tensor,op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
        return tensor.tolist()

    def atomic_torch(path, value):
        pending = path.with_name(path.name+'.pending')
        with pending.open('wb') as out:
            torch.save(value, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(pending, path)

    def build(checkpoint=None):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint is not None else None
        if saved is not None:
            require_topology(saved['topology'],payload['topology'],
                saved.get('complete_update',payload['migration']['kind']=='legacy_contract_verified'))
            if saved['identity'] != payload['identity'] or len(saved['rank_rng']) != saved['topology']['world']:
                raise ValueError('检查点与来源、配方或逐rank随机状态不匹配')
            if 'contract' in saved and saved['contract'] != payload['contract']:
                raise ValueError('子进程恢复的科学合同与准入记录不同')
        engine = MedSAM3Adapter(config, training=True)
        from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
        from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks
        from sam3.train.loss.sam3_loss import Sam3LossWrapper
        matcher = BinaryHungarianMatcherV2(cost_class=2., cost_bbox=5., cost_giou=2., focal=True)
        engine.model.matcher = matcher
        criterion = Sam3LossWrapper(loss_fns_find=[
            Boxes(weight_dict={'loss_bbox':5., 'loss_giou':2.}),
            IABCEMdetr(pos_weight=10., weight_dict={'loss_ce':20., 'presence_loss':20.},
                pos_focal=False, alpha=.25, gamma=2, use_presence=True, pad_n_queries=200),
            Masks(weight_dict={'loss_mask':200., 'loss_dice':10.}, focal_alpha=.25, focal_gamma=2., compute_aux=False)],
            matcher=matcher, o2m_matcher=BinaryOneToManyMatcher(alpha=.3, threshold=.4, topk=4),
            o2m_weight=2., use_o2m_matcher_on_o2m_aux=False, normalization='local')
        names = [k for k,p in engine.model.named_parameters() if p.requires_grad]
        if not names or any(not k.endswith(('.lora_A','.lora_B')) for k in names):
            raise ValueError('仅允许已登记医学LoRA参数参与更新')
        optimizer = torch.optim.AdamW((p for p in engine.model.parameters() if p.requires_grad),
            lr=training['learning_rate'], weight_decay=.01, betas=(.9,.999), eps=1e-8)
        def scale(step):
            warmup = training['warmup_steps']
            if step < warmup:
                return (step+1)/warmup
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,training['updates']-warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        if saved is not None:
            expected = {k for k in engine.model.state_dict() if k.endswith(('.lora_A','.lora_B'))}
            if set(saved['lora']) != expected:
                raise ValueError('训练检查点LoRA键集合不同')
            loaded = engine.model.load_state_dict(saved['lora'], strict=False)
            if loaded.unexpected_keys or any(k in expected for k in loaded.missing_keys):
                raise ValueError('训练LoRA恢复不完整')
            optimizer.load_state_dict(saved['optimizer'])
            verify_optimizer_device(optimizer)
            scheduler.load_state_dict(saved['scheduler'])
            state.update(step=saved['step'], best_loss=saved['best_loss'],
                validation_pending=saved.get('validation_pending',saved['step'] > 0))
            rng = resumed_rng(saved['rank_rng'],rank,world,state['step'])
            restore_rng(rng)
            current_lora = {k:v for k,v in engine.model.state_dict().items() if k in expected}
            if (not tensors_equal(current_lora,saved['lora']) or
                not tensors_equal(optimizer.state_dict(),saved['optimizer']) or
                not tensors_equal(scheduler.state_dict(),saved['scheduler']) or
                not tensors_equal(rng_state(),rng)):
                raise ValueError('LoRA、优化器、调度器或随机状态未完整恢复')
            event(root,'medsam3_lora_restored',rank=rank,step=state['step'],optimizer_states=len(optimizer.state),
                restored_state=True,restored_exact=not payload['migration'].get('topology_changed',False)
                    and not payload['migration'].get('code_changed',False),migration=payload['migration'])
        return engine, criterion, optimizer, scheduler

    def save(name):
        states = all_states(rng_state())
        if rank == 0:
            lora = {k:v.detach().cpu() for k,v in engine.model.state_dict().items() if k.endswith(('.lora_A','.lora_B'))}
            saved = dict(identity=payload['identity'],contract=payload['contract'],contract_identity=digest(payload['contract']),
                implementation=payload['implementation'],migration=payload['migration'],complete_update=True,
                topology=payload['topology'],rank_rng=states,
                lora=lora,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                step=state['step'],best_loss=state['best_loss'],validation_pending=state['validation_pending'])
            atomic_torch(root/(name+'.pt'),saved)
            atomic_torch(root/(name+'_lora_weights.pt'),lora)
            event(root,'medsam3_lora_checkpoint',name=name,step=state['step'],world=world)
        coordinator.wait(rank,state['step'],'checkpoint_saved')

    def boundary():
        nonlocal engine, criterion, optimizer, scheduler
        coordinator.wait(rank,state['step'],'memory_boundary')
        snap = client.snapshot()
        stop,paused,low = combine([int(time() >= payload['deadline'] or snap.cancelled or snap.sample_age_seconds > 30),
            int(snap.paused),int(device.type == 'cuda' and torch.cuda.mem_get_info()[0] <
                                  max(2**30,config['resources']['vram_reserve_bytes']))],maximum=True)
        if stop:
            raise InterruptedError('医学LoRA达到截止、取消或公共采样过期')
        if low:
            raise MemoryError('显存空闲量低于声明的预留')
        if paused:
            name = 'latest' if payload['operation'] == 'train' else payload['checkpoint']
            if payload['operation'] == 'train':
                save(name)
            del engine, criterion, optimizer, scheduler
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            event(root, 'medsam3_lora_waiting', step=state['step'])
            handle.resources['max_wait'] = min(payload['deadline']-time(),handle.resources['max_wait'])
            client.wait_active()
            if time() >= payload['deadline']:
                raise TimeoutError('医学LoRA等待达到固定截止')
            engine, criterion, optimizer, scheduler = build(root/(name+'.pt'))

    def loss_for(sample):
        from sam3.model.model_misc import SAM3Output
        from sam3.train.loss.loss_fns import CORE_LOSS_KEY
        point = datapoint(config, engine.transform, sample)
        batch = engine.official.collate_fn_api([point], dict_key='input', with_seg_masks=True)['input']
        batch = engine.official.copy_data_to_device(batch, device, non_blocking=False)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and config['precision'] == 'bf16'):
            outputs = engine.model(batch)
            targets = [engine.model.back_convert(t) for t in batch.find_targets]
            with SAM3Output.iteration_mode(outputs, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as stages:
                for steps, target in zip(stages, targets, strict=True):
                    for out in steps:
                        out['indices'] = engine.model.matcher(out, target)
                        for aux in out.get('aux_outputs', []):
                            aux['indices'] = engine.model.matcher(aux, target)
            value = criterion(outputs, targets)[CORE_LOSS_KEY]
        if not torch.isfinite(value).item():
            raise ValueError('真实医学LoRA官方损失非有限值')
        return value

    def validate():
        total, count = 0., 0
        for item in validation_rounds(payload['sources']['val'],world,rank):
            boundary()
            if item is not None:
                with stream([item],config,training['workers'],training['prefetch'],payload['deadline']) as samples:
                    sample = next(iter(samples))
                engine.model.eval()
                with torch.no_grad():
                    value = loss_for(sample)
                    total += value.item()
                    count += 1
                    del value
        total,count = combine([total,count])
        result = dict(loss=total/count, samples=int(count), step=state['step'], metric='official_detection_mask_loss')
        if rank == 0:
            event(root, 'medsam3_lora_validation', **result)
        return result

    def finish_validation():
        if not state['validation_pending']:
            return
        result = validate()
        state['validation_pending'] = False
        if result['loss'] < state['best_loss']:
            state['best_loss'] = result['loss']
            save('best')
        save('latest')

    client.wait_active()
    path = root/'latest.pt' if payload['resume'] else root/(payload['checkpoint']+'.pt') if payload['operation'] == 'validate' else None
    engine, criterion, optimizer, scheduler = build(path)
    event(root, 'medsam3_lora_started', operation=payload['operation'], step=state['step'],
        rank=rank,world=world,global_batch=training['batch'],local_samples=payload['topology']['shares'][rank],
        trainable_parameters=sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
        reader_threads=training['workers'], prefetch=training['prefetch'])
    if payload['operation'] == 'validate':
        result = validate()
        if rank == 0:
            write_json(root/'validation.json', result)
        if world > 1:
            dist.destroy_process_group()
        return
    finish_validation()
    end = min(training['updates'], state['step']+payload['stop_after']) if payload['stop_after'] else training['updates']
    began = perf_counter()
    source = payload['sources']['train']
    shares = payload['topology']['shares']
    for step in range(state['step'],end):
        boundary()
        engine.model.train()
        mark = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        local_loss = 0.
        items = (source[k % len(source)] for k in rank_indices(step,training['batch'],shares,rank))
        with stream(items,config,training['workers'],training['prefetch'],payload['deadline']) as samples:
            for sample in samples:
                value = loss_for(sample)
                local_loss += value.item()
                (value/training['batch']).backward()
                del value
        synchronize_gradients(engine.model)
        norm = torch.nn.utils.clip_grad_norm_((p for p in engine.model.parameters() if p.requires_grad), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        state['step'] += 1
        loss, = combine([local_loss/training['batch']])
        if rank == 0:
            event(root,'medsam3_lora_update',step=state['step'],loss=loss,gradient_norm=float(norm),
                seconds=perf_counter()-mark,global_batch=training['batch'],world=world)
        state['validation_pending'] = state['step'] % training['validate_every'] == 0 or state['step'] == end
        save('latest')
        finish_validation()
    result = dict(step=state['step'], best_loss=state['best_loss'], seconds=perf_counter()-began,
        topology=payload['topology'],
        completed=state['step'] == training['updates'], trained_parameters='medical_lora_only',
        peaks=dict(cuda_allocated=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                   cuda_reserved=torch.cuda.max_memory_reserved() if device.type == 'cuda' else None))
    result['ranks'] = all_states(dict(rank=rank,peaks=result['peaks']))
    if rank == 0:
        write_json(root/'training.json', result)
    if world > 1:
        dist.destroy_process_group()
