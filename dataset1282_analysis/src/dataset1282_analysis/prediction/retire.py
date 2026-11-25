import argparse
import json
import os
from pathlib import Path
import re
import shutil

import psutil

from ..contracts import atomic_json,read_json,sha256_file
from .data4 import check_data
from .preprocessed import PreprocessedInputs
from .reuse import SIX
from .source import implementation


BASE=Path('/data4/zhangruichen')
OLD=BASE/'back/dataset1282_preprocessing'
TRIALS=('xu-train-1794040488','xu-train-1794041533','xu-train-1794044872',
        'xu-trial-1794034045','xu-trial-koLaVt')


def take(source,target):
    if not source.exists():
        return
    if source.is_symlink() or target.is_symlink():
        raise ValueError('迁移文件不能为软链接')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha256_file(source)!=sha256_file(target):
            raise ValueError('目标已存在不同内容：'+str(target))
        source.unlink()
    else:
        source.rename(target)


def remove(path):
    if path.exists():
        if path.resolve()!=path or not (path.is_relative_to(OLD) or path.is_relative_to(BASE/'result/chd1282/runs')):
            raise ValueError('拒绝删除范围外或软链接目录')
        shutil.rmtree(path)


def main(argv=None):
    p=argparse.ArgumentParser(description='正式验收后迁移唯一模型结果并清理旧CHD1282数组；默认仅预览')
    p.add_argument('--preprocessed',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--cht-code',type=Path,required=True)
    p.add_argument('--old-export',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args(argv)
    if os.name!='posix' or not BASE.is_dir() or OLD.resolve()!=OLD:
        raise ValueError('此命令仅适用于已指定的data4服务器目录')
    current=Path(args.preprocessed).resolve(strict=True)
    if not current.is_relative_to(BASE/'data/chd-segmentation/preprocessed'):
        raise ValueError('当前预处理必须保留在正式data/preprocessed目录')
    source=check_data(current,args.cht_code)
    model_root=Path(args.models).resolve(strict=True)
    if not model_root.is_relative_to(BASE/'result/chd1282'):
        raise ValueError('模型必须保留在result/chd1282')
    model=read_json(model_root/'models.json')
    receipt=read_json(args.receipt)
    names=list(receipt['evaluation']['cases'])
    if (receipt['protocol']!='xu-data4-acceptance-v1' or receipt['cases']!=1 or receipt['mode']!='six-way'
            or receipt['checkpoints']!=model['checkpoints'] or receipt['implementation']!=implementation()
            or Path(receipt['preprocessed']).resolve()!=current
            or PreprocessedInputs(current,args.cht_code,names).identity!=receipt['preprocessed_identity']):
        raise ValueError('需要当前数据和当前模型的一例真实预测及评价验收')
    for branch in SIX:
        if sha256_file(model_root/(branch+'.pt'))!=model['checkpoints'][branch]:
            raise ValueError('保留模型内容不符')
    old_export=Path(args.old_export).resolve(strict=True)
    if old_export==source.source.root or not old_export.is_relative_to(BASE/'data/chd-segmentation/preprocessed'):
        raise ValueError('旧718导出目录范围不正确')
    previous=read_json(old_export/'prepared.json')
    for name,pointer in previous['storage']['cases'].items():
        new=source.prepared['storage']['cases'][name]
        for filename in ('image.npy','label.npy','locations.npz'):
            if not os.path.samefile(old_export/pointer['key'][:16]/filename,
                                    source.source.root/new['key'][:16]/filename):
                raise ValueError('旧718数组未由新数据硬链接保留')
    runs=BASE/'result/chd1282/runs'
    if (OLD/'experiments').is_symlink() or runs.is_symlink():
        raise ValueError('实验结果根不能为软链接')
    print(json.dumps(dict(source=str(OLD),retained_runs=str(runs),delete=[str(OLD/'input'),str(OLD/'work'),str(old_export)],
        retired_trials=list(TRIALS),
        backups=[str(x) for x in (BASE/'back').glob('*1282*') if x.is_dir() and x!=OLD],
        note='其他1282目录只列出，不按名称推断可删除；未知模型和备份需逐项确认'),ensure_ascii=False),flush=True)
    if not args.apply:
        return
    ancestors={os.getpid(),*(x.pid for x in psutil.Process().parents())}
    for process in psutil.process_iter(['pid','uids','cmdline','open_files'],ad_value=None):
        info=process.info
        if info['pid'] in ancestors or not info['uids'] or info['uids'].real!=os.getuid():
            continue
        paths=[*(info['cmdline'] or []),*(x.path for x in (info['open_files'] or []))]
        if any(str(OLD) in x for x in paths):
            raise ValueError('仍有本用户进程引用旧目录：'+str(info['pid']))
    raw=Path(source.prepared['manifest']['data_root']).resolve(strict=True)
    take(OLD/'input/dicom2nii_info.xlsx',raw/'dicom2nii_info.xlsx')
    take(OLD/'input/manifest.json',raw/'manifest-original.json')
    metadata=BASE/'result/chd1282/metadata'
    for path in OLD.glob('*.json'):
        take(path,metadata/path.name)
    for path in OLD.glob('*.log'):
        take(path,BASE/'course/chd1282/history/launch'/path.name)
    for path in (OLD/'work/environment_logs').glob('*'):
        if path.is_file() and path.suffix in ('.log','.json','.txt'):
            take(path,BASE/'course/chd1282/history/environment'/path.name)
    original=OLD/'work/prepared/zmic-source-v030/hu-marginal'
    for name in ('request.json','complete.json','cohort.json','fit.json'):
        take(original/name,metadata/'preprocessing'/name)
    if (OLD/'experiments').exists():
        if runs.exists():
            raise ValueError('result/runs已存在，拒绝覆盖或合并未知历史结果')
        runs.parent.mkdir(parents=True,exist_ok=True)
        (OLD/'experiments').rename(runs)
    if runs.exists():
        logs={}
        for path in runs.rglob('*.log'):
            key=(path.parent,re.sub(r'-[0-9]{10,}$','',path.stem))
            logs.setdefault(key,[]).append(path)
        for files in logs.values():
            newest=max(files,key=lambda x:x.stat().st_mtime_ns)
            take(newest,BASE/'course/chd1282/history'/newest.relative_to(runs))
            for path in files:
                if path!=newest:
                    path.unlink()
        for name in TRIALS:
            trial=runs/name
            if trial.is_dir() and not trial.is_symlink():
                for path in trial.rglob('*.json'):
                    if not {'cases','prepared','branches','prediction-cache','.runtime'} & set(path.relative_to(trial).parts):
                        take(path,metadata/'retired-trials'/name/path.relative_to(trial))
                remove(trial)
        for run in runs.iterdir():
            if not run.is_dir() or run.is_symlink() or not run.name.startswith('xu-'):
                continue
            for name in ('prepared','branches','prediction-cache'):
                remove(run/name)
            for cached in run.rglob('*.npy'):
                if cached.parent.name=='cases' and cached.with_suffix('.nii.gz').is_file():
                    cached.unlink()
        # 只合并内容与当前六模型完全相同的权重，其他唯一检查点保留。
        for path in runs.rglob('*.pt'):
            branch=path.stem if path.stem in SIX else path.parent.name.removeprefix('auto-') if path.name=='best.pt' else None
            if branch not in SIX or path.is_symlink():
                continue
            retained=model_root/(branch+'.pt')
            if not path.samefile(retained) and sha256_file(path)==model['checkpoints'][branch]:
                pending=path.with_name(path.name+'.link-pending')
                if pending.exists():
                    raise ValueError('未知未完成链接文件')
                os.link(retained,pending)
                os.replace(pending,path)
    remove(OLD/'input')
    remove(OLD/'work')
    if old_export.resolve()!=old_export or source.source.root.is_relative_to(old_export):
        raise ValueError('旧导出目录解析改变')
    shutil.rmtree(old_export)
    for path in OLD.glob('dataset1282-zmic*'):
        if path.is_file() and not path.is_symlink() and path.stat().st_size<=20*1024**2:
            path.unlink()
    remaining=[str(x) for x in OLD.iterdir()] if OLD.exists() else []
    if OLD.exists() and not remaining:
        OLD.rmdir()
    atomic_json(BASE/'result/chd1282/migration.json',dict(state='completed',
        raw=str(raw),preprocessed=str(current),models=str(model_root),checkpoints=str(model_root.parent/'checkpoints'),
        results=str(runs),logs=str(BASE/'course/chd1282'),remaining_back_entries=remaining,
        deleted=[str(OLD/'input'),str(OLD/'work'),str(old_export)]))
    print('已迁移模型、检查点、结果；已清理旧input、work和旧718导出。其他备份目录仍须按预览精确核定。',flush=True)


if __name__=='__main__':
    main()
