"""Standard-library model/range contract shared by the panel and compute workers."""
from pathlib import Path
from server_data import read_json, write_json

MODELS = {
    'B0': {'label': 'B0 基线 nnU-Net', 'trainer': 'Project1B0'},
    'B1': {'label': 'B1 外观增强 nnU-Net', 'trainer': 'Project1B1'},
}
FOLDS = ['ct_holdG', 'ct_holdA', 'ct_holdB', 'mr_holdE', 'mr_holdCD']
SCOPES = {'ct-first': ['ct_holdG'], 'mr-first': ['mr_holdE'], 'all': FOLDS}


def parse_models(value):
    if value is None:
        return list(MODELS)
    values = value if isinstance(value, (list, tuple)) else str(value).replace('，', ',').replace(',', ' ').split()
    aliases = {str(i): name for i, name in enumerate(MODELS, 1)}
    values = [aliases.get(str(x).upper(), str(x).upper()) for x in values]
    if values == ['ALL']:
        return list(MODELS)
    if not values or any(x not in MODELS for x in values):
        raise ValueError('至少选择一个现有模型：B0、B1，可逗号分隔或填写 all')
    return [name for name in MODELS if name in values]


def parse_folds(value):
    if isinstance(value, str) and value in SCOPES:
        return list(SCOPES[value])
    values = value if isinstance(value, (list, tuple)) else str(value).replace('，', ',').replace(',', ' ').split()
    if not values or any(x not in FOLDS for x in values):
        raise ValueError('实验范围必须为：' + ','.join(FOLDS))
    return [fold for fold in FOLDS if fold in values]


def load_selection(work, default_folds=None):
    path = Path(work) / '00_admin/model_selection.json'
    if not path.exists():
        return {'folds': list(default_folds or FOLDS), 'methods': list(MODELS)}
    value = read_json(path)
    return {'folds': parse_folds(value['folds']), 'methods': parse_models(value['methods'])}


def save_selection(work, folds, methods):
    """Selection is frozen before model work; never narrow an old full experiment."""
    work = Path(work)
    value = {'folds': parse_folds(folds), 'methods': parse_models(methods)}
    path = work / '00_admin/model_selection.json'
    if path.exists():
        if load_selection(work) != value:
            raise ValueError('本任务的模型/范围已冻结；更换选择请新建任务')
        return value
    if any((work / '07_experiments/formal').glob('*/run.json')):
        if value != {'folds': FOLDS, 'methods': list(MODELS)}:
            raise ValueError('旧全量实验不能缩小冻结范围；请通过历史输入建立独立任务')
    if (work / '08_results/evaluation/frozen_models.json').exists():
        raise ValueError('已有冻结评价清单，不能追加选择文件改变范围')
    write_json(path, value)
    return value


def trainer_class(method):
    import project_trainers
    return getattr(project_trainers, MODELS[method]['trainer'])
