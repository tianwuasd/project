from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import yaml

from ..artifacts.store import read_json
from ..config import load_config
from ..models.stunet.preprocess import CONTRACT as Q_CONTRACT
from ..preprocessing.offline import preprocess
from ..preprocessing.offline_store import recipe, storage_lease, validate_storage
from ..preprocessing.prepare import load_prepared
from .medsam2.native import CONTRACT as MED_CONTRACT, native_recipe, produce_native_case, resolve_window
from .workflow import options, validated_q_groups


def require_q_statistics(config, prepared):
    if config['technical']:
        return
    if any('train' in row['roles'] for row in prepared['manifest']['cases']):
        validated_q_groups(config, prepared)
    else:
        groups = yaml.safe_load(Path(config['groups']).read_text(encoding='utf-8'))
        if not isinstance(groups, dict) or set(groups) != set('ABCD'):
            raise ValueError('Q预测预处理需要原冻结A/B/C/D分组')
        fitted = set(prepared['plan'].get('fit_ids', []))
        if not fitted or not fitted <= set(groups['A']):
            raise ValueError('Q预测必须使用A组拟合计划')
    fitted = prepared['plan'].get('generation', {}).get('request', {}).get('training_cases', [])
    groups = yaml.safe_load(Path(config['groups']).read_text(encoding='utf-8'))
    if any(row['case_id'] not in groups['A'] for row in fitted):
        raise ValueError('Q拟合来源包含A组以外的患者')


def expected_recipe(config, prepared):
    if config['model'] == 'q_route1':
        require_q_statistics(config, prepared)
        common = dict(preprocessing=dict(statistics_source='Q-A-technical' if config['technical'] else 'Q-A'))
        return recipe(prepared, common)
    if not config['technical'] and prepared['dataset'] == 'chd1282' and (
            prepared['manifest'].get('split_source', {}).get('format') == 'q1282' or
            any('q_group' in row for row in prepared['manifest']['cases'])):
        raise ValueError('MedSAM2正式1282输入需要原冻结训练划分')
    common = dict(preprocessing=dict(statistics_source='MedSAM2-training'))
    return native_recipe(prepared, common, window=resolve_window(config, prepared),
                         derived_window=config['gray_window'] is None)


def validate_model_storage(config, prepared):
    expected = expected_recipe(config, prepared)
    root = validate_storage(prepared, expected_transform=expected['transform'])
    if read_json(root/'catalog.json')['recipe'] != expected:
        raise ValueError('模型灰度窗、统计来源或完整处理配方与离线数据不同')
    return expected


@contextmanager
def model_storage_lease(config, prepared, stage):
    expected = validate_model_storage(config, prepared)
    reference = dict(config, run_dir=str(Path(config['run_dir'])/stage))
    with storage_lease(prepared, reference, expected_transform=expected['transform'], expected_recipe=expected):
        yield


def preprocess_model(model_path, prepare_config_path, prepared_path, root, output):
    model = options(model_path)
    config = load_config(prepare_config_path)
    source = load_prepared(prepared_path)
    config = deepcopy(config)
    config.update(model=model['model'], model_stage='preprocess')
    config['resources']['device'] = 'cpu'
    config['preprocessing']['root'] = str(Path(root).resolve())
    config['run_dir'] = str(Path(model['run_dir'])/'preprocess')
    expected = expected_recipe(model, source)
    if model['model'] == 'q_route1':
        config['preprocessing']['statistics_source'] = expected['statistics_source']
        result = preprocess(config, prepared_path, output, method_label='Q')
    else:
        config['preprocessing']['statistics_source'] = 'MedSAM2-training'
        window = resolve_window(model, source)
        result = preprocess(config, prepared_path, output, method_label='MedSAM2',
            producer=partial(produce_native_case, window=window),
            recipe_builder=partial(native_recipe, window=window, derived_window=model['gray_window'] is None))
    validate_model_storage(model, result)
    return result
