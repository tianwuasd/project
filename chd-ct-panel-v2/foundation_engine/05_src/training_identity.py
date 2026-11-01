"""Shared training/evaluation identity, usable by the stdlib controller."""
import os
from pathlib import Path
from batch_distribution import validate_global_batch
from server_data import read_json,digest
from stunet_recipe import recipe


def training_signature(work,fold,method,*,global_batch=None):
    work=Path(work)
    split=read_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    prep=work/f"04_data/derived/nnUNet_preprocessed/Dataset{split['dataset_id']:03d}_{fold}"
    batch=validate_global_batch(global_batch if global_batch is not None else int(os.environ.get('WHOLE_HEART_GLOBAL_BATCH','4')))
    return dict(fold=fold,method=method,seed=0,plan_sha256=digest(prep/'WholeheartBaselinePlans.json'),
        splits_sha256=digest(work/'04_data/manifests/splits_v1.json'),
        environment_sha256=digest(work/'06_configs/pip_freeze_training.txt'),
        conda_environment_sha256=digest(work/'06_configs/conda_explicit_training.txt'),
        code_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        selected_global_batch_size=batch,training_recipe=recipe(method))
