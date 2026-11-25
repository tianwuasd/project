from dataclasses import replace
from pathlib import Path
import os
import shutil
import tempfile
from types import SimpleNamespace

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..storage_migration import canonical_path, load_relocation, resolve_relocated_path
from . import PROTOCOL


FINE = ('crop64', 'crop128', 'all64', 'all128')


def safe_path(root, relative):
    path = root / relative
    if path.resolve() != path or root not in path.parents:
        raise ValueError('Linked or escaping prediction artifact')
    return path


def implementation():
    from ..xu_pipeline import ROOT, _sources
    files = _sources(False)
    for path in (*Path(__file__).parent.glob('*.py'), ROOT/'scripts/xu_predict.py',
                 ROOT/'configs/xu-resume-compat.json'):
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return files


def _receipt(training, stage, contract_id, schema):
    path = safe_path(training, f'receipts/{stage.name}.json')
    if not path.is_file():
        raise ValueError(f'Training stage has no completed receipt: {stage.name}')
    receipt = read_json(path)
    expected = 'skipped' if stage.skip_reason else 'complete'
    if receipt.get('state') != expected:
        raise ValueError(f'Training stage is not completed or its receipt changed: {stage.name}')
    if stage.skip_reason:
        if receipt.get('reason') != stage.skip_reason:
            raise ValueError('Auxiliary skip record changed')
    else:
        files = receipt['files']
        if not set(stage.required) <= files.keys():
            raise ValueError('Incomplete training artifacts')
    return receipt


def inspect_source(training, cache, mode):
    from ..xu_pipeline_stages import iter_stages
    training = Path(training).resolve(strict=True)
    record = read_json(safe_path(training,'run.json'))
    relocation = load_relocation(training)
    if relocation is not None:
        relative_run = (training/'run.json').relative_to(Path(relocation['destination'])).as_posix()
        if relative_run not in relocation['contracts'] or canonical_path(cache.run_dir) != record['prepared']:
            raise ValueError('Prediction source is absent from the verified migration contracts')
    technical = cache.scope == 'technical_subset'
    schema = None
    if (record['request'] != cache.request_digest
            or record['scope'] != cache.scope
            or record['padding_percent'] != 20 or (record['profile']=='smoke') != technical):
        raise ValueError('Training identity differs from the prepared input or audited implementation')
    args = SimpleNamespace(prepared=cache.run_dir, technical_subset=technical,
                           profile=record['profile'], supervision=record['supervision'],
                           device=record['device']['device'], seed=record['seed'],
                           crop64=None,crop128=None,reviews=None)
    from ..xu_init_extension import bind_extension, extension_identity
    extension_file = safe_path(training,'.runtime/auto-init-extension.json')
    extension = None
    args.auto_init = extension_file.is_file()
    if args.auto_init:
        extension_record = read_json(extension_file)
        args.init_radius_mm = extension_record['radius_mm']
        args.init_margin_percent = extension_record['margin_percent']
        extension = bind_extension(args,training,record)

    def external_path(value):
        path = Path(value)
        if relocation is not None and (path == Path(relocation['source']) or Path(relocation['source']) in path.parents):
            path = resolve_relocated_path(path,Path(relocation['destination'])/'.storage-relocation.json',training/'run.json')
        return path
    if record['protocol'] == 'xu-model-run-2':
        args.device_names = [row['device'] for row in record['devices']]
    for name in ('crop64','crop128'):
        item = record['supplied_'+name]
        if item:
            args.__dict__[name] = external_path(item['path'])
    if record.get('reviews'):
        args.reviews = external_path(record['reviews']['path'])
    needed = {*FINE,'audit-roi'}
    if mode != 'four-way':
        needed.update(('targets','init64','init128'))
        if extension:
            needed.update(('auto-targets','auto-init64','auto-init128'))
    if mode in ('blood','vessels'):
        needed.add('blood')
    receipts, paths, hashes = {}, {}, {}
    for stage in iter_stages(cache,args,training):
        if stage.name in needed:
            identity_stage = stage
            if relocation is not None:
                identity_stage = replace(stage,argv=tuple(canonical_path(value) if Path(value).is_absolute() else value
                                                         for value in stage.argv))
            stage_id = extension_identity(stage,digest(record),extension)
            receipts[stage.name] = _receipt(training,identity_stage,stage_id,schema)
            if stage.name in (*FINE,'init64','init128','auto-init64','auto-init128','blood'):
                key = stage.name.removeprefix('auto-')
                if stage.skip_reason:
                    if stage.name in ('crop64','crop128'):
                        paths[key] = getattr(args,stage.name)
                        hashes[key] = record['supplied_'+key]['sha256']
                    elif stage.name == 'blood':
                        raise ValueError('Blood prediction requires completed blood training')
                else:
                    paths[key] = training/stage.name/'best.pt'
                    hashes[key] = receipts[stage.name]['files']['best.pt']
        if needed <= receipts.keys():
            break
    if needed != receipts.keys() or bool('init64' in paths) != bool('init128' in paths):
        raise ValueError('Incomplete source training stages')
    if mode == 'six-way' and not {'init64','init128'} <= paths.keys():
        raise ValueError('六网络模式要求两个已完成的init模型')
    book = read_json(training/'audit-roi/boxes.json')
    if book['padding_percent'] != 20:
        raise ValueError('Prediction requires the selected 20 percent ROI margin')
    if book['checkpoints'] != {name:hashes[name] for name in ('crop64','crop128')}:
        raise ValueError('ROI and locator checkpoints differ')
    return dict(protocol=PROTOCOL,run=record,receipts=receipts,checkpoints=hashes,
                boxes=book,prepared=str(cache.run_dir),training=str(training)), paths


def _freeze_checkpoints(paths, identity, building):
    import torch
    from ..xu_blood_checkpoint import FORMAT
    for name,path in paths.items():
        target = building/(name+'.pt')
        shutil.copyfile(path,target)
        if sha256_file(target) != identity['checkpoints'][name]:
            raise ValueError('冻结模型内容与所选训练权重不同')
        if name == 'blood':
            state = torch.load(path,map_location='cpu',weights_only=True)
            if state.get('format') == FORMAT:
                for entry in state['components'].values():
                    component = safe_path(Path(path).resolve().parent,entry['file'])
                    frozen = safe_path(building,entry['file'])
                    shutil.copyfile(component,frozen)
            del state


def freeze_source(training, cache, out, mode):
    identity, paths = inspect_source(training,cache,mode)
    destination = out/'source'
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix='.freeze-',dir=out) as scratch:
            building = Path(scratch)/'source'
            building.mkdir()
            _freeze_checkpoints(paths,identity,building)
            atomic_json(building/'manifest.json',identity)
            os.replace(building,destination)
    verify_frozen(out,identity)
    return identity


def verify_frozen(out, expected=None):
    root = out/'source'
    identity = read_json(safe_path(root,'manifest.json'))
    if identity['protocol'] != PROTOCOL or (expected is not None and identity != expected):
        raise ValueError('Frozen prediction source differs')
    for name,expected_hash in identity['checkpoints'].items():
        if name not in (*FINE,'init64','init128','blood'):
            raise ValueError('Unknown frozen model')
        safe_path(root,name+'.pt').stat()
    return identity
