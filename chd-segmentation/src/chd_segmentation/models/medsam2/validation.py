from copy import deepcopy
import random
from time import perf_counter

import numpy as np
import torch


def validation_rounds(names,world):
    if world<1 or len(set(names))!=len(names):
        raise ValueError('验证进程数或病例名单无效')
    return [list(names[index:index+world])+[None]*max(0,index+world-len(names))
            for index in range(0,len(names),world)]


def aggregate_validation(names,rows,rank_cases):
    cases = [item for group in rank_cases for item in group]
    by_id = {item['case_id']:item for item in cases}
    if len(cases)!=len(names) or set(by_id)!=set(names):
        raise ValueError('分布式验证病例重复或缺失')
    unit = 'patient' if all(rows[name].get('patient') for name in names) else 'case'
    groups = {}
    for name in names:
        key = rows[name]['patient'] if unit=='patient' else name
        groups.setdefault(key,[]).append(by_id[name]['loss'])
    return dict(metric='validation_sequence_loss',value=float(np.mean([np.mean(values) for values in groups.values()])),
        cases=[by_id[name] for name in names],aggregation_unit=unit,aggregation_groups=len(groups))


def validate_sequences(trainer, recipe, prepared, names, seed, *, gate, rank):
    from hydra.utils import instantiate
    from training.utils.data_utils import collate_fn
    if not names:
        return None
    started = perf_counter()
    rows = {row['case_id']: row for row in prepared['manifest']['cases']}
    leaf = deepcopy(recipe['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0])
    leaf['video_dataset']['names'] = list(names)
    leaf['training'] = False
    # 验证固定缩放与标准化，训练增强配置保持官方定义。
    transforms = leaf['transforms'][0]['transforms']
    leaf['transforms'][0]['transforms'] = [value for value in transforms if value['_target_'].rsplit('.', 1)[-1]
        in ('RandomResizeAPI', 'ToTensorAPI', 'NormalizeAPI')]
    model = trainer.model.module
    saved = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 prompt=deepcopy(model.rng.bit_generator.state), training=model.training)
    if trainer.device.type == 'cuda':
        saved['cuda'] = torch.cuda.get_rng_state(trainer.device)
    cases = []
    try:
        model.eval()
        dataset = instantiate(leaf)
        for round_index,assigned in enumerate(validation_rounds(names,gate.world)):
            phase = f'med-validation:{trainer.epoch}:{round_index}'
            gate.start(rank,phase,[int(name is not None) for name in assigned],trainer.device)
            name = assigned[rank]
            if name is not None:
                index = round_index*gate.world+rank
                random.seed(seed+index)
                np.random.seed(seed+index)
                torch.manual_seed(seed+index)
                model.rng = np.random.default_rng(seed+index)
                with gate.build(rank,0,0):
                    batch = collate_fn([dataset[index]], dict_key='all').to(trainer.device)
                gate.mark_ready(rank)
                gate.deliver(rank)
            gate.ready.wait(rank,phase+':input-ready',trainer.device)
            if name is not None:
                with torch.no_grad(), torch.autocast(trainer.device.type,dtype=torch.bfloat16,
                        enabled=trainer.device.type=='cuda' and trainer.optim_conf.amp.enabled):
                    losses,_,_ = trainer._step(batch,model,'val')
                value = float(next(iter(losses.values())).detach())
                if not np.isfinite(value):
                    raise FloatingPointError('MedSAM2验证损失不是有限数值：'+name)
                cases.append(dict(case_id=name,loss=value))
                del batch,losses
            gate.finish(rank,phase,trainer.device)
    finally:
        random.setstate(saved['python'])
        np.random.set_state(saved['numpy'])
        torch.set_rng_state(saved['torch'])
        model.rng.bit_generator.state = saved['prompt']
        if trainer.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda'], trainer.device)
        model.train(saved['training'])
    from ..execution import all_states
    return dict(aggregate_validation(names,rows,all_states(cases)),seed=seed,seconds=perf_counter()-started)
