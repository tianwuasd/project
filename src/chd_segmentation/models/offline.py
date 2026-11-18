from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import yaml

from ..artifacts.store import read_json, write_json, require_output
from ..config import load_config
from ..models.stunet.preprocess import CONTRACT as Q_CONTRACT
from ..preprocessing.offline import preprocess
from ..preprocessing.offline_store import recipe, storage_lease, validate_storage
from ..preprocessing.prepare import load_prepared, validate_prepared_config
from ..observability.events import event
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
    root = validate_storage(prepared, expected_transform=None)
    actual = read_json(root/'catalog.json')['recipe']
    permitted = (Q_CONTRACT,) if config['model']=='q_route1' else (Q_CONTRACT,MED_CONTRACT)
    if (actual['transform'] not in permitted or actual['dataset']!=prepared['dataset'] or
            actual['labels']!=prepared['dataset_json']):
        raise ValueError('现有预处理的数据集或实际输入形式不能由当前模型消费')
    return actual


@contextmanager
def model_storage_lease(config, prepared, stage):
    expected = validate_model_storage(config, prepared)
    reference = dict(config, run_dir=str(Path(config['run_dir'])/stage))
    with storage_lease(prepared, reference, expected_transform=expected['transform'], expected_recipe=expected):
        yield


def quick_check_model(model, prepared, stage, prepare_config=None):
    from ..preflight import _path_check
    from .execution import execution_config
    common = execution_config(model)
    if model['technical'] != (prepared['scope'] == 'technical'):
        raise ValueError('模型与准备描述的正式或技术范围不同')
    expected = validate_model_storage(model, prepared) if prepared.get('storage') else expected_recipe(model, prepared)
    checks = dict(dataset=prepared['dataset'],cases=[row['case_id'] for row in prepared['manifest']['cases']])
    checks['recipe'] = dict(transform=expected['transform'], statistics=expected.get('statistics_source', expected.get('window_source')))
    checks['paths'] = {name: _path_check(common[name], writable=True) for name in ('run_dir', 'work_root')}
    if prepare_config is not None:
        from ..preflight import preprocessing_disk_check
        validate_prepared_config(prepared, prepare_config)
        checks['disk_plan'] = preprocessing_disk_check(prepare_config, prepared, expected)
    return dict(ok=True, stage=stage, model=model['model'], dataset=prepared['dataset'],
                scope=prepared['scope'], prepared_identity=prepared['identity'], checks=checks,
                limits=['未读取CT或标签体素', '未初始化模型', '离线数组仅核对元数据及文件stamp'])


def preprocess_model(model_path, prepare_config_path, prepared_path, root, output):
    model = options(model_path)
    config = load_config(prepare_config_path)
    source = load_prepared(prepared_path)
    config = deepcopy(config)
    config.update(model=model['model'], model_stage='preprocess')
    config['resources']['device'] = 'cpu'
    config['preprocessing']['root'] = str(Path(root).resolve())
    config['run_dir'] = str(Path(model['run_dir'])/'preprocess')
    if source.get('storage'):
        actual = validate_model_storage(model,source)
        if source['dataset'] != config['dataset'] or model['technical'] != (source['scope']=='technical'):
            raise ValueError('现有预处理的数据集或任务范围不同')
        target = require_output(output,config['work_root'],[source['storage']['root'],
            source['manifest']['data_root'],prepared_path,Path(__file__).resolve().parents[3]])
        if target.exists() and read_json(target)!=source:
            raise FileExistsError('已有输出描述对应其他预处理来源')
        if not target.exists():
            write_json(target,source)
        event(config['run_dir'],'preprocess_completed',cases=len(source['manifest']['cases']),
            reused=len(source['manifest']['cases']),new=0,output=str(target),
            data=source['storage']['root'],transform=actual['transform'])
        return source
    expected = expected_recipe(model, source)
    event(config['run_dir'], 'quick_check', report=quick_check_model(model, source, 'preprocess', config))
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
