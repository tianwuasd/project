"""Stdlib per-case commit checks, shared by coordinator and CPU/GPU workers."""
from pathlib import Path
from server_data import digest,read_json


def folder(work,model):return Path(work)/f'08_results/evaluation/{model["fold"]}_{model["method"]}'


def signature(work,model,case_id,sha):
    inventory={c['case_id']:c for c in read_json(Path(work)/'04_data/manifests/case_inventory.json')['cases']}
    case=inventory[case_id]
    if digest(case['image'])!=case['image_sha256']:raise ValueError('评价影像发生变化：'+case_id)
    return dict(frozen_sha256=sha,image_sha256=case['image_sha256'],label_sha256=digest(case['label']),case_id=case_id)


def committed(path,expected,outputs):
    path=Path(path)
    if not path.exists():return False
    value=read_json(path)
    if value.get('signature')!=expected or value.get('status')!='completed':raise ValueError('病例提交签名不符：'+str(path))
    for key in outputs:
        item=value['outputs'][key]
        if digest(item['path'])!=item['sha256']:raise ValueError('病例输出摘要不符：'+str(item['path']))
    return True


def cached(work,model,case_id,sha,phase):
    expected=signature(work,model,case_id,sha);base=folder(work,model)
    if committed(base/f'{case_id}.json',expected,['raw','lcc']):return True
    return phase=='gpu' and committed(base/'intermediate'/f'{case_id}.json',expected,['logits','properties'])
