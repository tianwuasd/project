from copy import deepcopy
import random
from time import perf_counter

import numpy as np
import torch


def validate_sequences(trainer, recipe, prepared, names, seed):
    from hydra.utils import instantiate
    from training.utils.data_utils import collate_fn
    if not names:
        return None
    started = perf_counter()
    rows = {row['case_id']: row for row in prepared['manifest']['cases']}
    aggregation_unit = 'patient' if all(rows[name].get('patient') for name in names) else 'case'
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
    patients, cases = {}, []
    try:
        model.eval()
        dataset = instantiate(leaf)
        for index, name in enumerate(names):
            random.seed(seed+index)
            np.random.seed(seed+index)
            torch.manual_seed(seed+index)
            model.rng = np.random.default_rng(seed+index)
            batch = collate_fn([dataset[index]], dict_key='all').to(trainer.device)
            with torch.no_grad(), torch.autocast(trainer.device.type, dtype=torch.bfloat16,
                    enabled=trainer.device.type == 'cuda' and trainer.optim_conf.amp.enabled):
                losses, _, _ = trainer._step(batch, model, 'val')
            value = float(next(iter(losses.values())).detach())
            if not np.isfinite(value):
                raise FloatingPointError('MedSAM2验证损失不是有限数值：'+name)
            key = rows[name]['patient'] if aggregation_unit == 'patient' else name
            patients.setdefault(key, []).append(value)
            cases.append(dict(case_id=name, loss=value))
            del batch, losses
    finally:
        random.setstate(saved['python'])
        np.random.set_state(saved['numpy'])
        torch.set_rng_state(saved['torch'])
        model.rng.bit_generator.state = saved['prompt']
        if trainer.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda'], trainer.device)
        model.train(saved['training'])
    return dict(metric='validation_sequence_loss', value=float(np.mean([
        np.mean(values) for values in patients.values()])), cases=cases,
        aggregation_unit=aggregation_unit, aggregation_groups=len(patients),
        seed=seed, seconds=perf_counter()-started)
