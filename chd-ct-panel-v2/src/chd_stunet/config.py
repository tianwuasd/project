"""Resolve one explicit configuration; local machine paths stay out of Git."""
import json
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


def prepare_paths(local):
    """Scientific parameters live in the method; machine config supplies paths."""
    platforms=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
    return dict(platforms.get(sys.platform,{}),**{k:v for k,v in local.items()
                if k in {'raw_root','preprocess_output_root','runtime_root','descriptor'}})


def training_paths(local):
    platforms=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
    return dict(platforms.get(sys.platform,{}),**{k:v for k,v in local.items()
                if k in {'training_prepared_root','training_output_root','runtime_root','weights_path',
                         'completed_preprocessing_default'}})


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
