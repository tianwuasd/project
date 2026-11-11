from pathlib import Path

import torch

from .contracts import sha256_file


FORMAT = 'xu-blood-reference-v1'


def selected_components(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('format') != FORMAT:
        return state
    if set(state['components']) != {'encoder','sequence'}:
        raise ValueError('血池模型组件清单不完整')
    for stage, entry in state['components'].items():
        name = stage+'-selected.pt'
        if entry['file'] != name:
            raise ValueError('血池组件文件名与阶段不一致')
        path = checkpoint.parent/name
        if path.is_symlink() or path.resolve() != path or sha256_file(path) != entry['sha256']:
            raise ValueError('血池组件权重发生变化')
        component = torch.load(path,map_location='cpu',weights_only=True)
        if (component['contract'] != state['contract'] or component['stage'] != stage
                or component['epoch'] != entry['epoch'] or sha256_file(path) != entry['sha256']):
            raise ValueError('血池组件的训练身份或轮次不一致')
        state[stage] = component['model']
    return state


def save_selected_bundle(path, kind, contract):
    from .xu_training import _save
    path = Path(path)
    components = {}
    for stage in ('encoder','sequence'):
        name = stage+'-selected.pt'
        component_path = path.parent/name
        checksum = sha256_file(component_path)
        value = torch.load(component_path,map_location='cpu',weights_only=True)
        if (value['contract'] != contract or value['stage'] != stage
                or value['epoch'] < 1 or sha256_file(component_path) != checksum):
            raise ValueError('血池最佳模型组件未完整提交')
        components[stage] = dict(file=name,sha256=checksum,epoch=value['epoch'])
        del value
    if path.exists():
        existing = selected_components(path)
        if existing['contract'] != contract or existing['kind'] != kind:
            raise ValueError('已有血池模型的身份不同')
        for stage, entry in components.items():
            selected = torch.load(path.parent/entry['file'],map_location='cpu',weights_only=True)['model']
            if (existing[stage].keys() != selected.keys()
                    or any(not torch.equal(existing[stage][k],v) for k,v in selected.items())):
                raise ValueError('已有血池模型与最佳组件不同')
        return
    _save(path,dict(format=FORMAT,kind=kind,contract=contract,components=components))
