from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from datetime import timedelta
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import distributed as dist

from chd_stunet.acceleration.prediction import admit_prediction_device
from chd_stunet.acceleration.preprocess_cpu import snapshot
from chd_stunet.observability.runlog import RunLog, write_json

from .checkpoints import load, mirror_checkpoints, save
from .contracts import check_signature, file_sha, read_json
from .data import sample_clip, sample_seed, sequence_ids
from .inference import evaluate_case
from .io_recovery import InputReadWatch, InputUnavailable, input_path
from .journal import TrainingLog
from .losses import criterion, masked_loss
from .model import build, make_batch, versions
from .preprocessing import assert_case_files, case_metadata, eligible_starts


def make_optimizer(network, config):
    # 官方 AdamW 参数分组：编码器基础学习率、逐层衰减和无衰减 LayerNorm/bias。
    groups = {}
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = config['learning_rate']
        if name.startswith('image_encoder.'):
            lr = config['encoder_learning_rate']
            if name.startswith('image_encoder.trunk.') and 'pos_embed' not in name:
                trunk = network.image_encoder.trunk
                layer = trunk.get_layer_id(name.removeprefix('image_encoder.trunk.'))
                lr *= .9**(trunk.get_num_layers()+1-layer)
        module = network.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else network
        decay = 0. if 'bias' in name or isinstance(module, torch.nn.LayerNorm) else config['weight_decay']
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([dict(params=params, lr=lr, initial_lr=lr, weight_decay=decay)
                              for (lr, decay), params in groups.items()])


def sync_gradients(optimizer, device):
    # 分支未用到的参数保留 grad=None；另一张卡用到时则汇总其梯度。
    for group in optimizer.param_groups:
        parameters = group['params']
        used = torch.tensor([p.grad is not None for p in parameters], device=device, dtype=torch.int32)
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        active = [p for p, present in zip(parameters, used.tolist(), strict=True) if present]
        if not active:
            continue
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in active])
        dist.all_reduce(flat)
        for parameter, gradient in zip(active, flat.split([p.numel() for p in active]), strict=True):
            parameter.grad = gradient.view_as(parameter)


def update_step(network, optimizer, loss, clips, config, device, update, total_updates, world=1):
    all_objects = sum(len(clip['classes']) for _, clip in clips)
    frames = torch.zeros(config['frames'], dtype=torch.int32)
    for _, clip in clips:
        frames += clip['valid'].flatten(1).any(1).int()*len(clip['classes'])
    if world > 1:
        count = torch.cat((torch.tensor([all_objects], dtype=frames.dtype), frames)).to(device)
        dist.all_reduce(count)
        all_objects, *frame_objects = count.tolist()
    else:
        frame_objects = frames.tolist()
    optimizer.zero_grad(set_to_none=True)
    progress = update/max(1, total_updates-1)
    factor = .1+.9*(1+math.cos(math.pi*progress))/2
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr']*factor
    values = Counter()
    for seed, clip in clips:
        # 全局序列种子绑定输入与纠正点；预取和执行顺序不推进消费游标。
        torch.manual_seed(seed)
        network.rng = np.random.default_rng(seed)
        batch = make_batch(clip, device)
        valid = clip['valid'].to(device)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with autocast:
            outputs = network(batch, valid, clip['boxes'].to(device))
            losses = masked_loss(loss, outputs, batch.masks, valid, frame_objects)
        if not bool(torch.isfinite(losses['core_loss'])):
            raise FloatingPointError('MedSAM2 训练损失非有限')
        losses['core_loss'].backward()
        for key, value in losses.items():
            values[key] += float(value.detach())
    if world > 1:
        sync_gradients(optimizer, device)
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return dict(values), float(norm), all_objects


def create_clips(sequence, count, ids, root, rows, metadata, window, config, reader):
    result = []
    with reader as progress:
        for index in range(sequence, sequence+count):
            epoch, offset = divmod(index, len(ids))
            cid = sequence_ids(ids, epoch, config['seed'])[offset]
            seed = sample_seed(config['seed'], index)
            result.append((seed, sample_clip(root/'cases'/cid, metadata[cid], rows[cid],
                                             window, config, seed, progress=progress)))
    return result


def clip_prefetch_limit(path, maximum, world, batch_bytes):
    if maximum == 0:
        return 0
    live = read_json(path)
    if live['cpu_budget'] < 4*world:
        return 0
    # 预留正在消费的批次，剩余内存按进程均分。
    available = max(0, live['available_memory_bytes'] - live['reserve_bytes'])
    return min(maximum, max(0, available//(world*batch_bytes)-1))


def validate(network, config, device, root, val, metadata, window, output, state, log,
             frame_limit, checkpoint_sha, reader):
    epoch = state['epoch']
    progress = dict(epoch=epoch, update=state['update'], phase='validation', cases_total=len(val))
    predictor = build(config, training=False, device=device)
    predictor.load_state_dict(network.state_dict(), strict=True)
    scores, failures = {}, {}
    destination = output/'validation'/f'epoch_{epoch:04d}'
    for delay in (2, 5, 15, 30, 60, None):
        for cid in val:
            if cid in scores:
                continue
            log.status('running', **progress, case_id=cid, cases_completed=len(scores))
            case_dir = destination/cid
            marker = case_dir/'complete.json'
            identity = dict(checkpoint_sha256=checkpoint_sha, cache_identity=metadata[cid]['identity'],
                            case_id=cid, frame_limit=frame_limit,
                            prompt_source='simulated_gt_single_box')
            try:
                if marker.is_file():
                    record = read_json(marker)
                    if (record['identity'] != identity or record['result_sha256'] != file_sha(case_dir/'result.json')
                            or record['prompts_sha256'] != file_sha(case_dir/'prompts.json')):
                        raise ValueError('已完成验证病例的模型、来源或产物变化：'+cid)
                    score = read_json(case_dir/'result.json')
                    log.event('validation.reuse', f'复用已核验的病例 {cid}', case_id=cid, epoch=epoch)
                else:
                    if cid in failures:
                        with reader as read_progress:
                            actual = case_metadata(root/'cases'/cid,
                                                   dict(case_id=cid, source_hashes=metadata[cid]['source_hashes']),
                                                   track_files=True,
                                                   progress=lambda path, _: read_progress(path))
                        if actual['identity'] != metadata[cid]['identity']:
                            raise ValueError('验证病例缓存身份变化：'+cid)
                        metadata[cid] = actual
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type=='cuda'):
                        score = evaluate_case(predictor, root, cid, metadata[cid], window, case_dir, log,
                                              frame_limit=frame_limit, retain_raw=False, reader=reader)
                    write_json(marker, dict(identity=identity, result_sha256=file_sha(case_dir/'result.json'),
                                            prompts_sha256=file_sha(case_dir/'prompts.json')))
            except OSError as error:
                unavailable = input_path(error, root)
                if unavailable is None:
                    raise
                failures[cid] = dict(path=unavailable, error=str(error))
                continue
            scores[cid] = score
            failures.pop(cid, None)
        if not failures:
            break
        if delay is None:
            raise InputUnavailable([item['path'] for item in failures.values()],
                                   dict(cases=failures, progress=progress))
        log.status('waiting_io', **progress, cases_completed=len(scores),
                   pending=sorted(failures), next_retry_at=time.time()+delay)
        log.event('validation.wait_io', '等待重新读取未完成的验证病例', pending=failures, retry_seconds=delay)
        time.sleep(delay)
    del predictor
    scores = [scores[cid] for cid in val]
    result = dict(epoch=epoch, prompt_source='simulated_gt_single_box', cases=len(scores),
                  scope='axial_smoke_subset' if frame_limit else 'whole_volume',
                  mean_dice=sum(row['mean_dice'] for row in scores)/len(scores),
                  class_dice={key: sum(row['metrics'][key]['dice'] for row in scores)/len(scores)
                              for key in scores[0]['metrics']})
    write_json(destination/'summary.json', result)
    return result


def restore(network, optimizer, payload, device):
    network.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    torch.random.set_rng_state(payload['torch_rng'])
    if device.type == 'cuda' and payload['device_rng'] is not None:
        torch.cuda.set_rng_state(payload['device_rng'], device)
    return payload['state']


def probe_capacity(output, job, config, device, admission, root, rows, metadata, window, train,
                   world, rank, log, reader):
    candidates = [cid for cid in train if eligible_starts(
        metadata[cid]['frame_classes'], config['frames'], config['max_objects'])]
    if not candidates:
        raise ValueError('训练队列没有满足容量检查目标数的真实病例')
    local_count = config['global_batch'] if world==1 else 1
    clips = []
    with reader as progress:
        for index in range(local_count):
            number = rank*local_count+index
            cid = candidates[number%len(candidates)]
            seed = sample_seed(config['seed'], number)
            clips.append((seed, sample_clip(root/'cases'/cid, metadata[cid], rows[cid], window, config, seed,
                                            required_objects=config['max_objects'], progress=progress)))
    log.status('running', phase='capacity_probe', rank=rank)
    shared_encoding = job['shared_image_position_encoding']
    network = build(config, training=True, device=device, weights=job['weights'],
                    shared_image_position_encoding=shared_encoding)
    optimizer = make_optimizer(network, config)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    losses, gradient, objects = update_step(network, optimizer, criterion(), clips, config, device, 0, 1, world)
    if device.type=='cuda':
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_reserved(device)
        free, _ = torch.cuda.mem_get_info(device)
        if free < admission['reserve_bytes']:
            raise torch.cuda.OutOfMemoryError('正式尺寸容量检查未满足最低显存保留量')
    else:
        peak = 0
    evidence = dict(rank=rank, world_size=world, device=str(device), admission=admission,
                    image_size=config['image_size'], frames=config['frames'], targets=objects,
                    shared_image_position_encoding=shared_encoding,
                    local_targets=[len(clip['classes']) for _, clip in clips], loss=losses,
                    gradient_norm=gradient, peak_cuda_reserved_bytes=peak,
                    seconds=time.monotonic()-started, cases=[clip['case_id'] for _, clip in clips])
    log.event('training.capacity', '正式输入容量检查已完成，探测更新独立于训练检查点', **evidence)
    if world > 1:
        results = [None]*world
        dist.all_gather_object(results, evidence)
    else:
        results = [evidence]
    if rank == 0:
        write_json(output/'capacity.json', {'status': 'passed', 'ranks': results})
    del network, optimizer, clips
    if device.type=='cuda':
        torch.cuda.empty_cache()


def init_collective(device, world, rank):
    timeout = timedelta(seconds=300)
    backend = 'nccl'
    if device.type == 'cpu':
        # 本机进程之间走 IPv4 loopback，避免主机名被解析为网卡的 IPv6 地址。
        def local_gloo(store, rank, world, timeout):
            options = dist.ProcessGroupGloo._Options()
            options._devices = [dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')]
            options._timeout = timeout
            return dist.ProcessGroupGloo(store, rank, world, options)
        backend = 'medsam2_gloo'
        dist.Backend.register_backend(backend, local_gloo, devices=['cpu'])
    dist.init_process_group(backend, init_method=os.environ['CHD_MEDSAM2_RENDEZVOUS'],
                            world_size=world, rank=rank, timeout=timeout, device_id=device if device.type=='cuda' else None)


def run_worker(output, job, requested_device):
    output = Path(output)
    contract = job['contract']
    config = contract['config']
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world not in (1, config['global_batch']):
        raise ValueError('MedSAM2 计算进程数超出全局批量')
    primary = rank == 0
    recovery = job.get('recovery')
    generation = os.environ['CHD_MEDSAM2_GENERATION']
    audit = read_json(Path(recovery)/'input-audit.json')
    check_signature(audit)
    if (audit['generation'] != generation
            or audit['contract_identity'] != contract['identity'] or versions() != contract['versions']):
        raise ValueError('MedSAM2 输入校验结果与当前世代或运行环境不一致')
    root = Path(audit['root'])
    rows, metadata, train, val, window = (audit[key] for key in ('cases', 'metadata', 'train', 'val', 'window'))
    root_log = Path(recovery) if job.get('journal') else output
    train_reader = InputReadWatch(recovery, generation, rank, 'train')
    prefetch_reader = InputReadWatch(recovery, generation, rank, 'prefetch')
    validation_reader = InputReadWatch(recovery, generation, rank, 'validation')
    rank_log = root_log if primary else root_log/'ranks'/str(rank)
    log = TrainingLog(rank_log, job.get('task_id', contract['identity'])) if job.get('journal') else RunLog(rank_log)
    resources = snapshot()
    torch.set_num_threads(max(1, min(4, resources['cpu_budget']//world)))
    assigned = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if requested_device == 'cuda' else requested_device
    device, admission = admit_prediction_device(assigned, job.get('shared_desktop'))
    if device.type == 'cuda':
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('MedSAM2 CUDA 训练要求支持 BF16')
        torch.cuda.set_per_process_memory_fraction(admission['usable_bytes']/admission['total_bytes'], device)
    if world > 1:
        log.status('running', phase='process_group_init', rank=rank, world_size=world)
        init_collective(device, world, rank)
    log.event('training.device', f'MedSAM2 {device}，{config["image_size"]}×{config["image_size"]}，'
              f'{config["frames"]} 帧，{config["epochs"]} epoch', admission=admission, resources=resources,
              shared_image_position_encoding=job['shared_image_position_encoding'],
              rank=rank, world_size=world, precision='bf16' if device.type=='cuda' else 'fp32')
    if config['scope'] != 'smoke':
        probe_capacity(output, job, config, device, admission, root, rows, metadata, window,
                       train, world, rank, log, train_reader)
        if config['scope'] == 'capacity_probe':
            if primary:
                write_json(output/'summary.json', dict(status='completed', scope='capacity_probe', trained_updates=0,
                                                       capacity=read_json(output/'capacity.json')))
                write_json(output/'worker-result.json', {'state': 'completed'})
                log.status('completed', phase='capacity_probe')
            if world > 1:
                dist.destroy_process_group()
            return
    resume = os.environ.get('CHD_MEDSAM2_RESTORE') == '1'
    network = build(config, training=True, device=device, weights=None if resume else job['weights'],
                    shared_image_position_encoding=job['shared_image_position_encoding'])
    optimizer = make_optimizer(network, config)
    loss = criterion()
    if resume:
        state = restore(network, optimizer, load(recovery or output, contract), device)
        log.event('training.restore', f"从已持久化更新 {state['update']} 恢复",
                  sequence=state['sequence'], phase=state['phase'], epoch=state['epoch'])
        if primary and recovery:
            mirror_checkpoints(recovery, output)
    else:
        state = dict(update=0, sequence=0, epoch=0, phase='training',
                     best=None, history=[], classes={str(k): 0 for k in range(1, 8)})
        if primary:
            save(output, contract, network, optimizer, state, recovery=recovery)
    total_samples = config['epochs']*len(train)
    total_updates = math.ceil(total_samples/config['global_batch'])
    log.event('training.contract', '训练来源与初始检查点已验证', train_cases=len(train), val_cases=len(val),
              total_updates=total_updates, trainable_parameters=sum(p.numel() for p in network.parameters() if p.requires_grad))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    maximum_prefetch = job['prefetch_depth']
    supply_path = Path(recovery)/'supply.json'
    batch_count = config['global_batch']
    local_count = batch_count//world
    rank_offset = rank*local_count
    clip_args = (train, root, rows, metadata, window, config)
    # 估算包括张量、生成过程中的副本以及标签和图像读入空间。
    batch_bytes = local_count*(config['frames']*config['image_size']**2*(32+config['max_objects'])+16*1024**2)
    pending = deque()
    log.event('training.supply', f'每进程预取上限 {maximum_prefetch} 个批次',
              requested_depth=maximum_prefetch, estimated_batch_bytes=batch_bytes)
    with ExitStack() as stack:
        pool = ThreadPoolExecutor(max_workers=1)
        stack.callback(pool.shutdown, wait=True, cancel_futures=True)
        while state['update'] < total_updates or state['phase'] == 'validation':
            action = None
            if primary:
                if (output/'control/stop.json').is_file():
                    action = 'stopped'
                elif (output/'control/pause.json').is_file():
                    action = 'paused'
            if world > 1:
                flags = [action]
                dist.broadcast_object_list(flags, device=device)
                action = flags[0]
            if action:
                if primary:
                    save(output, contract, network, optimizer, state, recovery=recovery)
                    write_json(output/'worker-result.json', {'state': action})
                log.status(action, update=state['update'], epoch=state['epoch'])
                if world > 1:
                    dist.destroy_process_group()
                return
            if state['phase'] == 'validation':
                state['phase'] = 'training'
                ready = output/'control'/f"validation-{generation}-{state['epoch']}.json"
                if primary:
                    checkpoint = read_json(Path(recovery or output)/'checkpoints/latest.json')
                    validation = validate(network, config, device, root, val, metadata, window, output,
                                          state, log, job.get('validation_frames'), checkpoint['sha256'],
                                          validation_reader)
                    state['history'].append(validation)
                    improved = state['best'] is None or validation['mean_dice'] > state['best']['mean_dice']
                    if improved:
                        state['best'] = validation
                    save(output, contract, network, optimizer, state, best=improved, recovery=recovery)
                    log.event('validation.complete', f"epoch {state['epoch']} 全部验证病例完成", **validation)
                    if world > 1:
                        write_json(ready, {'epoch': state['epoch']})
                elif world > 1:
                    while not ready.is_file():
                        time.sleep(1)
                continue
            log.status('running', update=state['update'], epoch=state['epoch'], phase='clip_supply',
                       total_updates=total_updates)
            began = time.monotonic()
            sequence = state['sequence']
            local_start = sequence+rank_offset
            prefetched = bool(pending)
            ready_at_request = False
            if prefetched:
                prepared_sequence, future = pending.popleft()
                if prepared_sequence != sequence:
                    raise RuntimeError('MedSAM2 预取与已提交样本顺序不一致')
                ready_at_request = future.done()
                clips = future.result()
                del future
            else:
                clips = create_clips(local_start, local_count, *clip_args, train_reader)
            data_seconds = time.monotonic()-began
            next_sequence = sequence+batch_count
            limit = clip_prefetch_limit(supply_path, maximum_prefetch, world, batch_bytes)
            queue_sequence = pending[-1][0]+batch_count if pending else next_sequence
            while len(pending) < limit and queue_sequence < total_samples:
                pending.append((queue_sequence, pool.submit(
                    create_clips, queue_sequence+rank_offset, local_count, *clip_args, prefetch_reader)))
                queue_sequence += batch_count
            # 预取结果可能在后台等待，消费前重新确认源缓存文件。
            with train_reader as read_progress:
                for _, clip in clips:
                    cid = clip['case_id']
                    assert_case_files(root/'cases'/cid, metadata[cid], read_progress)
            log.status('running', update=state['update'], epoch=state['epoch'], phase='forward_backward')
            losses, gradient, objects = update_step(network, optimizer, loss, clips, config, device,
                                                    state['update'], total_updates, world)
            previous_epoch = state['epoch']
            state['sequence'] = next_sequence
            state['epoch'] = min(config['epochs'], state['sequence']//len(train))
            state['update'] += 1
            clip_rows = [dict(seed=seed, boxes=clip['boxes'].tolist(), sequence_index=index,
                              padded=index >= total_samples,
                              **{key: clip[key] for key in ('case_id','classes','indices','span_mm')})
                         for index, (seed, clip) in enumerate(clips, start=local_start)]
            if world > 1:
                combined = [None]*world
                dist.all_gather_object(combined, (losses, clip_rows))
                summed = Counter()
                clip_rows = []
                for rank_losses, rank_rows in combined:
                    summed.update(rank_losses)
                    clip_rows.extend(rank_rows)
                losses = dict(summed)
            for clip in clip_rows:
                for label in clip['classes']:
                    state['classes'][str(label)] += 1
            epoch_boundary = state['epoch'] != previous_epoch
            if epoch_boundary and (state['epoch'] % config['validation_interval'] == 0 or state['epoch']==config['epochs']):
                state['phase'] = 'validation'
            peak = torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
            log.event('training.update', f"epoch {state['epoch']}/{config['epochs']}，更新 {state['update']}/{total_updates}",
                      losses=losses, gradient_norm=gradient, objects=objects,
                      learning_rates=sorted({group['lr'] for group in optimizer.param_groups}),
                      data_wait_seconds=data_seconds, update_seconds=time.monotonic()-began,
                      first_sequence=sequence, next_sequence=next_sequence, prefetched=prefetched,
                      ready_at_request=ready_at_request, prefetch_limit=limit, queued_batches=len(pending),
                      peak_cuda_reserved_bytes=peak,
                      clips=clip_rows)
            step_stop = job.get('stop_after_steps') is not None and state['update'] >= job['stop_after_steps']
            if primary and (epoch_boundary or state['update'] % config['checkpoint_interval']==0 or step_stop):
                save(output, contract, network, optimizer, state, recovery=recovery)
            del clips
            if step_stop:
                if primary:
                    write_json(output/'worker-result.json', {'state': 'stopped'})
                log.status('stopped', update=state['update'], epoch=state['epoch'])
                if world > 1:
                    dist.destroy_process_group()
                return
    if world > 1:
        dist.destroy_process_group()
    if not primary:
        return
    result = dict(status='completed', scope=config['scope'], epoch=state['epoch'], updates=state['update'],
                  sampled_sequences=state['sequence'], padded_sequences=state['sequence']-total_samples,
                  world_size=world,
                  classes=state['classes'], best=state['best'], validation=state['history'],
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
                  config=config, contract_identity=contract['identity'], prompt_source='simulated_gt_single_box')
    write_json(output/'summary.json', result)
    write_json(output/'worker-result.json', {'state': 'completed'})
    log.status('completed', update=state['update'], epoch=state['epoch'])
