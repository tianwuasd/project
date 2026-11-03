"""Versioned training recipe and identities; path/resource choices are separate."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
from ..config import ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True).encode()).hexdigest()


def recipe(smoke=False, overrides=None):
    settings=json.loads((ROOT/'configs/training/stunet-s.json').read_text(encoding='utf-8'))
    settings['scope']='smoke' if smoke else 'formal'
    if smoke:
        settings.update(epochs=2,train_steps_per_epoch=4,validation_steps_per_epoch=1,
                        batch_size=1,patch_size=[64]*3)
    allowed={'epochs','train_steps_per_epoch','validation_steps_per_epoch','batch_size',
             'patch_size','amp','seed','include_auxiliary'}
    for key,value in (overrides or {}).items():
        if key not in allowed:raise ValueError('Unsupported recipe override: '+key)
        if value is not None:settings[key]=value
    for key in ('epochs','train_steps_per_epoch','validation_steps_per_epoch','batch_size'):
        if type(settings[key]) is not int or settings[key]<1:raise ValueError('Invalid '+key)
    if len(settings['patch_size'])!=3 or any(type(x) is not int or x<16 for x in settings['patch_size']):
        raise ValueError('Patch must have three integer dimensions >=16')
    return settings


def scientific_sources(root=ROOT):
    """AST ignores comments/docstrings; semantic edits still invalidate resume."""
    paths=list((root/'src/chd_stunet/training').glob('*.py'))
    paths += [root/'src/chd_stunet/_vendor/stunet_architecture.py',
              root/'src/chd_stunet/acceleration/training_cpu.py',
              root/'src/chd_stunet/preprocessing/annotations.py',
              root/'src/chd_stunet/preprocessing/cohort.py']
    result={}
    for path in sorted(paths):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                    node.body.pop(0)
        result[path.relative_to(root).as_posix()]=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    return result


def make_signature(settings, dataset, weight_sha):
    versions={n:importlib.metadata.version(n) for n in
              ('torch','nnunetv2','numpy','scipy','blosc2','batchgeneratorsv2','batchgenerators',
               'acvl-utils','scikit-image','threadpoolctl')}
    if versions['nnunetv2']!=settings['nnunet_version']:
        raise ValueError('Requires audited nnunetv2 '+settings['nnunet_version'])
    body=dict(recipe=settings,data=dataset,weight_sha256=weight_sha,versions=versions,
              producer_ast_sha256=scientific_sources())
    return dict(identity=digest(body),**body)
