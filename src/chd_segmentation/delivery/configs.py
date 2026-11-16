from copy import deepcopy
from pathlib import Path

import yaml


def model_configs(root, code, assets):
    root, code = Path(root), Path(code)
    course = root/'course/chd-segmentation'
    result = {}
    run = course/'runs/CHD1282-Q/20261115-offline'
    for stage, template, updates in [('base', 'q_base', 40000),
                                      ('localizer', 'q_localizer', 5000),
                                      ('local', 'q_route1', 20000)]:
        cfg = yaml.safe_load((code/'configs/models'/f'{template}.yaml').read_text(encoding='utf-8'))
        cfg.update(run_dir=str(run), work_root=str(course), updates=updates,
                   groups=str(course/'inputs/q1282/q_groups.json'),
                   pretrained=str(assets['stunet']) if stage == 'base' else None,
                   parents={key: str(run/key/'best.pt') for key in ('base', 'localizer')} if stage == 'local' else {})
        cfg['execution'].update(share=False, wait_seconds=None)
        cfg['logging'] = dict(heartbeat_seconds=15, case_details=True)
        result[f'configs/CHD1282-Q/20261115/{stage}.yaml'] = cfg
    template = yaml.safe_load((code/'configs/models/medsam2.yaml').read_text(encoding='utf-8'))
    for dataset in ('CHD1282', 'ImageCHD'):
        cfg = deepcopy(template)
        cfg.update(run_dir=str(course/'runs'/f'{dataset}-MedSAM2/20261115-offline'),
                   work_root=str(course), official_root=str(assets['official']),
                   dependency_root=str(assets['dependencies']), pretrained=str(assets['medsam2']))
        cfg['execution'].update(share=False, wait_seconds=None)
        cfg['logging'] = dict(heartbeat_seconds=15, case_details=True)
        result[f'configs/{dataset}-MedSAM2/20261115/model.yaml'] = cfg
    return result
