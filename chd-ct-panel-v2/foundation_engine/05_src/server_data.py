"""Bind private runtime manifests; labels are opened only at their authorized stage."""
import hashlib
import json
import os
from pathlib import Path
from cohort import PROJECT_ID, discover_cases, make_splits, label_access_ids, file_hash

digest = file_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def protocol_hash():
    return digest(Path(__file__).with_name('cohort.py'))


def resolve_cases(data_root, cases):
    current = discover_cases(data_root)
    if current != cases:
        raise ValueError('原始影像、病例身份或数据位置发生变化；请新建任务')
    return current


def bind_workspace(code_root, work_root, data_root, gpu_memory):
    code, work, data = [Path(p).expanduser().resolve() for p in [code_root, work_root, data_root]]
    if (work == code or work in code.parents or work.is_relative_to(code)
            or work.is_relative_to(data) or data.is_relative_to(work)):
        raise ValueError('结果目录必须独立于代码及原始数据')
    config = dict(project_id=PROJECT_ID, protocol_sha256=protocol_hash(), data_root=str(data),
                  work_root=str(work), gpu_memory_gb=gpu_memory)
    marker = work / '00_admin/server_config.json'
    if marker.exists() and read_json(marker) != config:
        raise ValueError('已有工作区不属于相同工程、协议、数据路径或显存规划')
    if not marker.exists() and work.exists() and any(p.name != '.run.lock' for p in work.iterdir()):
        raise ValueError('结果目录非空且未绑定到本工程')
    cases = discover_cases(data)
    splits = make_splits(cases)
    inventory = work / '04_data/manifests/case_inventory.json'
    split_path = work / '04_data/manifests/splits_v1.json'
    if inventory.exists():
        old = read_json(inventory)
        frozen = read_json(split_path)
        if old['cases'] != cases or frozen['folds'] != splits or frozen['inventory_sha256'] != digest(inventory):
            raise ValueError('已有来源清单或划分发生变化')
        return config
    write_json(marker, config)
    write_json(inventory, dict(project_id=PROJECT_ID, cases=cases, patient_id_available=False))
    write_json(work / '04_data/manifests/quarantine.json', [c for c in cases if c['status']=='quarantined'])
    write_json(split_path, dict(project_id=PROJECT_ID, inventory_sha256=digest(inventory), folds=splits,
                               patient_id_available=False))
    return config


def freeze_source_labels(work, fold):
    """Read exactly fit+cal labels. A change forbids reusing preprocessing/checkpoints."""
    work = Path(work)
    split = read_json(work / '04_data/manifests/splits_v1.json')['folds'][fold]
    cases = read_json(work / '04_data/manifests/case_inventory.json')['cases']
    from cohort import validate_split
    validate_split(split, cases)
    lookup = {c['case_id']: c for c in cases}
    labels = {cid: digest(lookup[cid]['label']) for cid in label_access_ids(split)}
    path = work / f'04_data/manifests/{fold}_source_labels.json'
    if path.exists() and read_json(path) != labels:
        raise ValueError('来源训练/验证标签已变化，禁止继续使用旧缓存和模型')
    write_json(path, labels)
    return labels
