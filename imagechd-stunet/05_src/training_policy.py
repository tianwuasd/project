"""Pure resource/budget decisions. No CUDA, process creation or hidden training."""
import math
import time


def validate_gpu_count(value, maximum=None):
    if value == 'auto' or value is None:
        return None
    if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
        raise ValueError('GPU数量必须为正整数或auto')
    result = int(value)
    if maximum is not None and result > maximum:
        raise ValueError(f'GPU数量超过实际可用范围总量{maximum}')
    return result


def distribution(global_batch, gpu_uuids, requested_count=None, *, allow_batch_growth=False, per_device_batch=None):
    if isinstance(global_batch, bool) or not isinstance(global_batch, int) or global_batch < 1:
        raise ValueError('全局batch必须为正整数')
    devices = list(gpu_uuids)
    if not devices or len(devices) != len(set(devices)) or any(not isinstance(v,str) or not v.startswith('GPU-') for v in devices):
        raise ValueError('需要不重复的实际GPU UUID')
    requested = validate_gpu_count(requested_count)
    reference = global_batch
    if allow_batch_growth:
        from batch_distribution import layout,MAX_GLOBAL_BATCH
        world = min(len(devices), requested or len(devices),MAX_GLOBAL_BATCH)
        local = global_batch if per_device_batch is None else per_device_batch
        if isinstance(local,bool) or not isinstance(local,int) or local<1:
            raise ValueError('每卡batch必须为正整数')
        sizes=layout(local,world)
        global_batch=sum(sizes);local=max(sizes)
        reason = 'global batch capped at eight; further ranks partition the same patches; capacity changes recorded'
    else:
        if per_device_batch is not None:
            raise ValueError('改变每卡batch需要明确的自动batch授权')
        ceiling = min(len(devices), requested or len(devices), global_batch)
        # Equal local batches preserve the upstream mean reduction semantics.
        world = max(n for n in range(1, ceiling+1) if global_batch % n == 0)
        local = global_batch//world
        sizes=[local]*world
        reason = 'preserve global batch and balanced rank losses; no batch multiplication to fill devices'
    return dict(global_batch_size=global_batch, reference_global_batch_size=reference,world_size=world,
                local_batch_size=local,local_batch_sizes=sizes, gpu_uuids=devices[:world],
                unused_gpu_uuids=devices[world:],
                reason=reason, sampling_ratio=global_batch/reference,
                train_patches_per_epoch=global_batch*250, reference_train_patches_per_epoch=reference*250,
                formal_epochs=250, train_iterations_per_epoch=250, validation_iterations_per_epoch=50)


def cpu_candidates(snapshot, patch_size, global_batch, world_size=1):
    """Compatibility name returning exactly one direct configuration (no search)."""
    if (not isinstance(world_size,int) or world_size<1 or isinstance(world_size,bool)
            or not isinstance(global_batch,int) or global_batch<1 or global_batch%world_size):
        raise ValueError('CPU候选需要有效且可均分的全局batch与卡数')
    if len(patch_size)!=3 or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in patch_size):
        raise ValueError('需要真实3D patch尺寸')
    from live_resources import training_policy
    one_batch=math.prod(patch_size)*(global_batch//world_size)*64
    policy=training_policy(snapshot,world_size,one_batch)
    if policy['status']=='blocked':
        raise ValueError('; '.join(policy['reasons']))
    rank=policy['rank_settings'][0]
    # Historical callers consume these aliases until their stage integration is
    # updated. There is exactly one direct configuration, never speed ranking.
    value=dict(policy,threads=rank['main_threads'],augmentation_workers=rank['train_workers'],
        validation_workers=rank['val_workers'],worker_threads=rank['worker_threads'],
        val_worker_threads=rank['val_worker_threads'],prefetch=rank['prefetch'],
        val_prefetch=rank['val_prefetch'],estimated_shm_bytes=0,world_size=world_size,
        global_batch_size=global_batch,local_batch_size=global_batch//world_size,
        estimate_scope='capacity estimate including both resident pools; pipe transport; not measured peak',
        reason='one direct stage configuration; aggregate active CPU phases and resident RAM bounded')
    return [value]


def select_trial(trials, *, global_batch, tolerance=.05):
    """Legacy v1 assessment reader; v2 stages must not use speed ranking."""
    if not 0<=tolerance<=.2:
        raise ValueError('选优容差无效')
    valid=[]
    for item in trials:
        train=item.get('median_step_seconds')
        validation=item.get('validation_step_seconds')
        batch = item.get('global_batch_size')
        if (item.get('status')!='passed' or (global_batch is not None and batch!=global_batch)
                or isinstance(batch,bool) or not isinstance(batch,int) or batch<1
                or item.get('measured_training_steps',0)<10
                or not isinstance(train,(int,float)) or not math.isfinite(train) or train<=0
                or not isinstance(validation,(int,float)) or not math.isfinite(validation) or validation<0):
            continue
        # Same fixed update budget, so this score compares expected epoch wall
        # cost rather than preferring a larger global batch's samples/second.
        score=250*train+50*validation+max(0,item.get('checkpoint_seconds',0))
        valid.append((score,item))
    if not valid:
        raise ValueError('没有足够有效样本且训练预算一致的成功候选')
    fastest=min(score for score,_ in valid)
    near=[(score,item) for score,item in valid if score<=fastest*(1+tolerance)]
    if global_batch is None:
        # Authorized auto-batch mode: among similarly fast full-epoch candidates,
        # prefer actual samples/s. This is explicitly a different sampling budget.
        return min(near,key=lambda pair:(-pair[1]['global_batch_size']/pair[1]['median_step_seconds'],
                                         pair[1].get('total_cpu_slots',1),pair[0]))[1]
    _,chosen=min(near,key=lambda pair:(pair[1].get('world_size',1)*pair[0],
                                      pair[1].get('total_cpu_slots',1),pair[0]))
    # Near ties on the same GPU count prefer fewer CPU slots.
    same_world=[(score,item) for score,item in near if item.get('world_size',1)==chosen.get('world_size',1)]
    _,chosen=min(same_world,key=lambda pair:(pair[1].get('total_cpu_slots',1),pair[0]))
    return chosen


def receipt_compatibility(receipt, identity, snapshot, *, now=None):
    reasons=[]
    if receipt.get('status')!='passed' or receipt.get('committed') is not True:
        reasons.append('测评未完整通过或尚未提交')
    if receipt.get('identity')!=identity:
        reasons.append('模型/数据/计划/代码/环境身份不匹配')
    current=time.time() if now is None else now
    created,ttl=receipt.get('created_at'),receipt.get('ttl_seconds')
    if (not isinstance(created,(int,float)) or not math.isfinite(created)
            or not isinstance(ttl,(int,float)) or not math.isfinite(ttl) or ttl<=0
            or current<created or current-created>ttl):
        reasons.append('测评配置已过期或时间信息无效')
    for required,available,label in [
        ('required_cpu_slots','cpu_slots','CPU'),
        ('required_ram_bytes','available_memory_bytes','RAM'),
        ('required_shm_bytes','shared_memory_available_bytes','共享内存')]:
        needed=receipt.get(required,0)
        actual=snapshot.get(available)
        if not isinstance(needed,(int,float)) or not math.isfinite(needed) or needed<0:
            reasons.append(label+'需求记录无效')
        elif needed>0 and (not isinstance(actual,(int,float)) or actual<needed):
            reasons.append(label+'当前可用量不足或未知')
    return reasons
