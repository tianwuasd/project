"""Resolve one explicit configuration; local machine paths stay out of Git."""
import json
import getpass
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def read_config():
    public=json.loads((ROOT/"configs/analysis.json").read_text(encoding="utf-8"))
    private=ROOT/"project.local.json"
    local=json.loads(private.read_text(encoding="utf-8-sig")) if private.is_file() else {}
    return public,local


def resolve_path(value, *, base=ROOT):
    if value is None or not str(value).strip():
        raise ValueError("Path must not be empty; it cannot default to the current directory")
    path=Path(value).expanduser()
    return (base/path if not path.is_absolute() else path).resolve()


def platform_paths(local, *, platform=None, project_root=ROOT):
    platforms=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
    values=platforms.get(platform or sys.platform,{})
    if not any('{workspace_root}' in str(value) for value in values.values()):
        return values
    configured=local.get('workspace_root') or os.environ.get('CHD_WORKSPACE_ROOT')
    if configured:
        workspace=Path(configured).expanduser().resolve()
    else:
        ancestors=[p for p in (project_root,*project_root.parents) if p.name.endswith('_workspace')]
        if not ancestors:
            raise ValueError('请在本机配置设置 workspace_root，或提供 CHD_WORKSPACE_ROOT；生成目录必须位于 workspace')
        workspace=ancestors[0]
    return {name:value.format(workspace_root=str(workspace),user=getpass.getuser())
            if isinstance(value,str) else value for name,value in values.items()}


def prepare_paths(local):
    """Scientific parameters live in the method; machine config supplies paths."""
    return dict(platform_paths(local),**{k:v for k,v in local.items()
                if k in {'raw_root','preprocess_output_root','runtime_root','descriptor'}})


def training_paths(local):
    return dict(platform_paths(local),**{k:v for k,v in local.items()
                if k in {'training_prepared_root','training_output_root','runtime_root','weights_path',
                         'completed_preprocessing_default'}})


def prediction_paths(local):
    return dict(training_paths(local), **{k: v for k, v in local.items()
                if k in {'prediction_output_root'}})


def default_prepared(paths):
    if paths.get('training_prepared_root'):return resolve_path(paths['training_prepared_root'])
    pointer=ROOT/'outputs/latest_preprocessing.json'
    if pointer.is_file():
        candidate=resolve_path(json.loads(pointer.read_text(encoding='utf-8'))['output'])
        if (candidate/'manifests/imagechd_input.json').is_file():return candidate
    if paths.get('completed_preprocessing_default'):
        candidate=resolve_path(paths['completed_preprocessing_default'])
        if (candidate/'manifests/imagechd_input.json').is_file():return candidate
    raise ValueError('No prepared run selected; run prepare first or use --prepared / training_prepared_root')
