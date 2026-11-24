import os
from pathlib import Path
import sys


def execute(payload, handle, gpus):
    root = Path(payload['config']['training']['run_dir'])
    stream = (root/'training.log').open('a', encoding='utf-8')
    sys.stdout = sys.stderr = stream
    cache = root/'runtime-cache'
    temporary = cache/'temporary'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update({key:str(cache/name) for key,name in {
        'TRITON_CACHE_DIR':'triton', 'TORCHINDUCTOR_CACHE_DIR':'inductor',
        'TORCH_HOME':'torch', 'HF_HOME':'huggingface', 'MPLCONFIGDIR':'matplotlib',
        'XDG_CACHE_HOME':'xdg', 'CUDA_CACHE_PATH':'cuda',
        'TMPDIR':'temporary', 'TMP':'temporary', 'TEMP':'temporary'}.items()})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]['uuid']
    train(payload, handle)


def train(payload, handle):
    import gc
    import math
    import random
    from time import time, perf_counter
    import numpy as np
    import torch
    from .environment import verify_worker_environment
    from .adapter import MedSAM3Adapter
    from .training_data import datapoint, stream
    from ...acceleration.runtime import attach
    from ...artifacts.store import write_json
    from ...observability.events import event

    verify_worker_environment(payload['environment'])
    config, training = payload['config'], payload['config']['training']
    root, device = Path(training['run_dir']), torch.device(config['device'])
    client = attach(handle)
    torch.set_num_threads(config['resources']['cpu_threads']-training['workers'])
    random.seed(training['seed'])
    np.random.seed(training['seed'])
    torch.manual_seed(training['seed'])
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
    state = dict(step=0, best_loss=float('inf'))

    def atomic_torch(path, value):
        pending = path.with_name(path.name+'.pending')
        with pending.open('wb') as out:
            torch.save(value, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(pending, path)

    def build(checkpoint=None):
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
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['identity'] != payload['identity']:
                raise ValueError('检查点与来源、划分、LoRA结构或优化设置不匹配')
            expected = {k for k in engine.model.state_dict() if k.endswith(('.lora_A','.lora_B'))}
            if set(saved['lora']) != expected:
                raise ValueError('训练检查点LoRA键集合不同')
            loaded = engine.model.load_state_dict(saved['lora'], strict=False)
            if loaded.unexpected_keys or any(k in expected for k in loaded.missing_keys):
                raise ValueError('训练LoRA恢复不完整')
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            state.update(step=saved['step'], best_loss=saved['best_loss'])
            random.setstate(saved['python_rng'])
            rng = saved['numpy_rng']
            np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
            torch.set_rng_state(saved['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(saved['cuda_rng'])
            event(root, 'medsam3_lora_restored', step=state['step'], optimizer_states=len(optimizer.state))
        return engine, criterion, optimizer, scheduler

    def save(name):
        lora = {k:v.detach().cpu() for k,v in engine.model.state_dict().items() if k.endswith(('.lora_A','.lora_B'))}
        rng = np.random.get_state()
        saved = dict(identity=payload['identity'], lora=lora, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), step=state['step'], best_loss=state['best_loss'],
            python_rng=random.getstate(), numpy_rng=(rng[0],rng[1].tolist(),*rng[2:]), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state() if device.type == 'cuda' else None)
        atomic_torch(root/(name+'.pt'), saved)
        atomic_torch(root/(name+'_lora_weights.pt'), lora)
        event(root, 'medsam3_lora_checkpoint', name=name, step=state['step'])

    def boundary():
        nonlocal engine, criterion, optimizer, scheduler
        if time() >= payload['deadline']:
            raise TimeoutError('医学LoRA达到固定截止')
        snap = client.snapshot()
        if snap.cancelled or snap.sample_age_seconds > 30:
            raise InterruptedError('公共监控取消或采样过期')
        if device.type == 'cuda' and torch.cuda.mem_get_info()[0] < max(2**30,config['resources']['vram_reserve_bytes']):
            raise MemoryError('显存空闲量低于声明的预留')
        if snap.paused:
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
        with stream(payload['sources']['val'], config, training['workers'], training['prefetch'], payload['deadline']) as samples:
            for sample in samples:
                boundary()
                engine.model.eval()
                with torch.no_grad():
                    value = loss_for(sample)
                    total += value.item()
                    count += 1
                    del value
        result = dict(loss=total/count, samples=count, step=state['step'], metric='official_detection_mask_loss')
        event(root, 'medsam3_lora_validation', **result)
        return result

    client.wait_active()
    path = root/'latest.pt' if payload['resume'] else root/(payload['checkpoint']+'.pt') if payload['operation'] == 'validate' else None
    engine, criterion, optimizer, scheduler = build(path)
    event(root, 'medsam3_lora_started', operation=payload['operation'], step=state['step'],
        trainable_parameters=sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
        reader_threads=training['workers'], prefetch=training['prefetch'])
    if payload['operation'] == 'validate':
        write_json(root/'validation.json', validate())
        return
    end = min(training['updates'], state['step']+payload['stop_after']) if payload['stop_after'] else training['updates']
    began = perf_counter()
    source = payload['sources']['train']
    items = (source[k % len(source)] for k in range(state['step'], end))
    with stream(items, config, training['workers'], training['prefetch'], payload['deadline']) as samples:
        for sample in samples:
            boundary()
            engine.model.train()
            mark = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            value = loss_for(sample)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_((p for p in engine.model.parameters() if p.requires_grad), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            state['step'] += 1
            event(root, 'medsam3_lora_update', step=state['step'], loss=value.item(), gradient_norm=float(norm), seconds=perf_counter()-mark)
            del value
            save('latest')
            if state['step'] % training['validate_every'] == 0 or state['step'] == end:
                result = validate()
                if result['loss'] < state['best_loss']:
                    state['best_loss'] = result['loss']
                    save('best')
                save('latest')
    result = dict(step=state['step'], best_loss=state['best_loss'], seconds=perf_counter()-began,
        completed=state['step'] == training['updates'], trained_parameters='medical_lora_only',
        peaks=dict(cuda_allocated=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                   cuda_reserved=torch.cuda.max_memory_reserved() if device.type == 'cuda' else None))
    write_json(root/'training.json', result)
