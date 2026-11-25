import os
from pathlib import Path

from ..contracts import atomic_json, digest, read_json, sha256_file
from .heldout import verify_case
from .source import FINE, safe_path, verify_frozen
from .workflow import branch_record


SIX = (*FINE,'init64','init128')


def link_file(source, destination):
    if source.resolve()!=source or destination.resolve()!=destination:
        raise ValueError('缓存复用不接受软链接或越界路径')
    if destination.exists():
        if not os.path.samefile(source,destination):
            raise ValueError('已有缓存路径未指向所选原缓存')
    else:
        os.link(source,destination)


def reuse_branch(previous, out, branch, name, original, job):
    values=branch_record(previous,branch,name,original)
    if values is None:
        return False
    source=safe_path(previous/'branches'/branch,name+'.npy')
    record=read_json(previous/'branches'/branch/(name+'.json'))
    if sha256_file(source)!=record['sha256']:
        raise ValueError('既有分支预测内容发生变化')
    root=out/'branches'/branch
    root.mkdir(parents=True,exist_ok=True)
    link_file(source,root/source.name)
    atomic_json(root/(name+'.json'),dict(record,job=digest(job),reused_from=str(source)))
    return True


def reuse_prediction(previous, out, job):
    previous=Path(previous).resolve(strict=True)
    out=Path(out).resolve(strict=True)
    if job['mode']!='six-way' or previous==out or previous in out.parents or out in previous.parents:
        raise ValueError('缓存复用要求six-way及相互独立的新旧输出目录')
    original=read_json(previous/'run.json')
    old,new=verify_frozen(previous),verify_frozen(out)
    if (original['cases']!=job['cases'] or any(original[k]!=job[k] for k in ('plans','dataset','technical_subset'))
            or any(old['checkpoints'].get(k)!=new['checkpoints'].get(k) for k in SIX)
            or not set(SIX)<=new['checkpoints'].keys() or old['boxes']!=new['boxes']
            or old['run']['request']!=new['run']['request']):
        raise ValueError('原预测的病例、六个模型、取框或预处理计划不同')
    for relative in ('xu_fusion.py','xu_roi.py'):
        key='src/dataset1282_analysis/'+relative
        if not original['implementation'].get(key) or original['implementation'][key]!=job['implementation'].get(key):
            raise ValueError('原预测的融合或取框实现不同，不能直接复用')
    copied,prepared=0,0
    for name in job['cases']:
        source=previous/'prepared'/name
        if source.exists():
            marker=verify_case(previous,name)
            destination=out/'prepared'/name
            destination.mkdir(parents=True,exist_ok=True)
            for filename in ('image.b2nd','geometry.json'):
                link_file(safe_path(source,filename),destination/filename)
            atomic_json(destination/'complete.json',dict(marker,job=digest(job),reused_from=str(source)))
            verify_case(out,name)
            prepared+=1
        for branch in SIX:
            copied+=int(reuse_branch(previous,out,branch,name,original,job))
    atomic_json(out/'cache-reuse.json',dict(source=str(previous),prepared_cases=prepared,
        branch_cases=copied,method='same_filesystem_hardlinks',original_job=digest(original)))
    print(f'复用预处理缓存 {prepared} 例；分支预测缓存 {copied} 份；缺失分支继续推理',flush=True)
