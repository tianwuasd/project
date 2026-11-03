"""Freeze project code for owned workers; live code updates cannot hot-swap data."""
import hashlib
import json
from pathlib import Path
import shutil
from ..config import ROOT
from ..observability.runlog import write_json
from .contracts import scientific_sources


def freeze_code(output,signature,create):
    root=Path(output)/'code';record=Path(output)/'code_sources.json'
    if create:
        if root.exists() or record.exists():raise ValueError('Code snapshot already exists; use a new run directory')
        paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').rglob('*.json'))
        paths += [ROOT/'run.py',ROOT/'smoke.py']
        files={}
        for path in paths:
            name=path.relative_to(ROOT).as_posix();target=root/name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            files[name]=hashlib.sha256(target.read_bytes()).hexdigest()
        if scientific_sources(root)!=signature['producer_ast_sha256']:
            raise ValueError('Project code changed while freezing the training snapshot')
        write_json(record,dict(files=files))
    if not record.is_file():raise ValueError('Missing frozen code record; old run preserved')
    files=json.loads(record.read_text(encoding='utf-8'))['files']
    for name,sha in files.items():
        target=(root/name).resolve()
        if not target.is_relative_to(root.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen training code checksum differs: '+name)
    if scientific_sources(root)!=signature['producer_ast_sha256']:
        raise ValueError('Frozen training science contract differs')
    return root/'run.py'
