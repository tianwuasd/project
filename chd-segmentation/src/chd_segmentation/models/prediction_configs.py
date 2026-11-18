from pathlib import Path

import yaml

from ..artifacts.store import read_json
from .standalone_config import PREDICTION_MODELS, options


def generate(model, prepared_path, server_root, input_source=None, device='cuda'):
    if model not in PREDICTION_MODELS:
        raise ValueError('配置生成仅支持totalsegmentator与medsam3')
    prepared = read_json(prepared_path)
    dataset = prepared['dataset']
    if dataset not in ('imagechd','chd1282'):
        raise ValueError('未知数据集')
    server_root = Path(server_root).resolve()
    course = server_root/'course/chd-segmentation'
    target = course/'inputs'/dataset/(model+'.yaml')
    template = Path(__file__).parents[3]/'configs/models'/(model+'.yaml')
    config = yaml.safe_load(template.read_text(encoding='utf-8'))
    config.update(run_dir=str(course/'runs'/dataset/model),work_root=str(course),
        output=str(server_root/'result/chd-segmentation'/dataset/model),device=device)
    config['input_source']['path'] = str(Path(input_source).resolve()) if input_source else None
    prefix = '/data4/zhangruichen/'
    for key,value in config['backend'].items():
        if isinstance(value,str) and value.startswith(prefix):
            config['backend'][key] = str(server_root/value[len(prefix):])
    if model=='medsam3':
        names = prepared['dataset_json']['labels']
        config['backend']['class_prompts'] = {k:v for k,v in config['backend']['class_prompts'].items() if k in names}
    options(config,target)
    if target.exists():
        if yaml.safe_load(target.read_text(encoding='utf-8'))!=config:
            raise FileExistsError('已有正式配置不同，需选择新的输入目录')
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x',encoding='utf-8',newline='\n') as stream:
            yaml.safe_dump(config,stream,allow_unicode=True,sort_keys=False)
    return dict(config=str(target),model=model,dataset=dataset,prepared=str(Path(prepared_path).resolve()))
