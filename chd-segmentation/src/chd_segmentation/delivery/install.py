from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from filelock import FileLock
import yaml

from ..artifacts.store import digest, file_digest, read_json, write_json
from ..datasets.q_split import build_q_split, create_q_bundle
from ..datasets.transfer import bind_inputs, manifest_identity
from .assets import verify_assets
from .configs import model_configs


SOURCE_ID = 'd98440f41e27242a808694f8008a31984356dd032941f088298447007715eab3'
Q_ID = 'c427ad74beb3a720159479aaf4a1fa4c2df7c846c0e6bfad416e6167171005d3'
GROUPS_SHA = '5f2c5a82fbbb746adbbb919a93080f886b1115fd6c2d183b14a8a8b9a97dc820'


def checked_bundle(path, expected):
    path = Path(path).resolve(strict=True)
    descriptor = read_json(path/'bundle.json')
    if descriptor['identity'] != expected or digest({k: v for k, v in descriptor.items() if k != 'identity'}) != expected:
        raise ValueError('冻结输入包身份不同：'+str(path))
    for name, sha in descriptor['immutable_files'].items():
        if Path(name).name != name or file_digest(path/name) != sha:
            raise ValueError('冻结输入文件摘要不同：'+str(path/name))
    for info in descriptor['datasets'].values():
        if Path(info['manifest']).name != info['manifest'] or manifest_identity(read_json(path/info['manifest'])) != info['identity']:
            raise ValueError('病例清单事实身份不同：'+str(path))
    return descriptor


def inspect_inputs(source):
    source = Path(source)
    descriptor = checked_bundle(source, SOURCE_ID)
    info = descriptor['datasets']['chd1282']
    _, groups = build_q_split(read_json(source/info['manifest']), read_json(source/info['split']))
    encoded = json.dumps(groups, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != GROUPS_SHA or \
            {key: len(value) for key, value in groups.items()} != dict(A=393, B=197, C=128, D=456):
        raise ValueError('重建Q组合与已确认冻结名单不同')
    return dict(source_identity=descriptor['identity'], groups=groups)


def ensure_q_inputs(source, output):
    inspect_inputs(source)
    output = Path(output)
    if output.exists():
        descriptor = checked_bundle(output, Q_ID)
        if file_digest(output/'q_groups.json') != GROUPS_SHA:
            raise ValueError('已有Q名单发生变化')
        return dict(identity=descriptor['identity'], reused=True)
    source = Path(source).resolve(strict=True)
    output = output.resolve()
    base = source.parent if output.is_relative_to(source) else output.parent
    base.mkdir(parents=True, exist_ok=True)
    staging = base/('.q-build-'+uuid4().hex)
    try:
        result = create_q_bundle(source, staging)
        checked_bundle(staging, Q_ID)
        audit = read_json(staging/'q_split_audit.json')
        audit['output'] = str(output)
        write_json(staging/'q_split_audit.json', audit)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError('Q目标被其他进程创建')
        staging.rename(output)
    finally:
        if staging.exists():
            if staging.parent != base or staging.is_symlink() or any(p.is_symlink() for p in staging.rglob('*')):
                raise ValueError('Q构建目录归属异常，保留待核对')
            shutil.rmtree(staging)
    return dict(identity=result['bundle_identity'], reused=False)


def prepare_configs(root, source, bound, qdir, stu):
    course = root/'course/chd-segmentation'
    result = {}
    source_descriptor = read_json(source/'bundle.json')
    for label, dataset in [('CHD1282-MedSAM2', 'chd1282'), ('ImageCHD-MedSAM2', 'imagechd'), ('CHD1282-Q', 'chd1282')]:
        if label == 'CHD1282-Q':
            item = deepcopy(next(e for e in source_descriptor['experiments'] if e['config']['dataset'] == dataset))
            cfg = item['config']
            cfg['split'].update(path=str(qdir/'q1282_split.json'), format='q1282', unit='patient')
            cfg['training']['patch'] = [128]*3
            cfg['plan'] = str(course/'plans/CHD1282-Q/20261115/preprocessing_plan.json')
            manifest = qdir/'chd1282_manifest.json'
        else:
            item = next(e for e in source_descriptor['experiments'] if e['config']['dataset'] == dataset and e['config']['fold'] == 0)
            cfg = yaml.safe_load((source/item['filename']).read_text(encoding='utf-8'))
            split = source_descriptor['datasets'][dataset]['split']
            cfg['split']['path'] = str(bound/split)
            manifest = bound/source_descriptor['datasets'][dataset]['manifest']
            if cfg.get('plan') is not None and not Path(cfg['plan']).is_absolute():
                cfg['plan'] = str(source/cfg['plan'])
        cfg.update(manifest=str(manifest), pretrained=str(stu), work_root=str(course),
                   run_dir=str(course/'prepare'/label/'20261115'),
                   result_root=str(root/'result/chd-segmentation'/label/'20261115-offline'))
        cfg.setdefault('preprocessing', {}).update(root=str(root/'data/chd-segmentation/preprocessed'),
                                                 statistics_source='Q-A' if label == 'CHD1282-Q' else 'main-train')
        result[f'configs/{label}/20261115/prepare.yaml'] = cfg
    return result


def check_configs(course, configs):
    for name, value in configs.items():
        raw = course/name
        path = raw.resolve()
        if not path.is_relative_to(course.resolve()) or raw.is_symlink():
            raise ValueError('配置目标越界')
        if path.exists() and yaml.safe_load(path.read_text(encoding='utf-8')) != value:
            raise ValueError('已有配置内容不同，保留原件：'+str(path))


def copy_input_metadata(source, target, descriptor):
    names = {'bundle.json', *descriptor['immutable_files']}
    names.update(row['manifest'] for row in descriptor['datasets'].values())
    names.update(row['filename'] for row in descriptor['experiments'])
    for name in names:
        if Path(name).name != name or not (source/name).is_file() or (source/name).is_symlink():
            raise ValueError('原输入包含无效元数据路径')
    target.mkdir(parents=True, exist_ok=False)
    for name in sorted(names):
        shutil.copy2(source/name, target/name)


def validate_destination(path, within, protected):
    raw, anchor = Path(path), Path(within).resolve()
    resolved = raw.resolve()
    if raw.is_symlink() or resolved == anchor or not resolved.is_relative_to(anchor):
        raise ValueError('安装写入目标越界：'+str(raw))
    for original in protected:
        original = Path(original).resolve()
        if resolved.is_relative_to(original) or original.is_relative_to(resolved):
            raise ValueError('安装写入目标与受保护资料重叠：'+str(raw))


def validate_asset_destination(root, code, source, target, dataset_roots=None):
    roots = dict(chd1282=Path(root)/'data/dataset1282_CHD_CT_back', imagechd=Path(root)/'data/ImageCHD_dataset')
    roots.update(dataset_roots or {})
    validate_destination(target, root, [code, source, *roots.values()])
    validate_destination(Path(target).with_name(Path(target).name+'.pending'), root, [code, source, *roots.values()])


def install(root, code, source, assets, check=False, dataset_roots=None):
    root, code, source = [Path(p).resolve(strict=True) for p in (root, code, source)]
    info = inspect_inputs(source)
    course = root/'course/chd-segmentation'
    if course.is_relative_to(source) or source == course:
        raise ValueError('安装目录与原输入根重叠')
    qdir = course/'inputs/q1282'
    descriptor = read_json(source/'bundle.json')
    roots = dict(chd1282=root/'data/dataset1282_CHD_CT_back', imagechd=root/'data/ImageCHD_dataset')
    roots.update(dataset_roots or {})
    for protected in [code, *[Path(p) for p in roots.values()]]:
        resolved = protected.resolve()
        if course.resolve().is_relative_to(resolved) or resolved.is_relative_to(course.resolve()):
            raise ValueError('安装过程目录与源码或原始数据重叠')
    originals = [read_json(source/row['manifest']) for row in descriptor['datasets'].values()]
    bound = source if all(not row.get('binding_required', False) and Path(row['data_root']).is_dir() for row in originals) \
        else course/'inputs/CHD1282-ImageCHD'
    configs = model_configs(root, code, assets)
    configs.update(prepare_configs(root, source, bound, qdir, assets['stunet']))
    protected = [code, *roots.values(), *[path for path in source.iterdir() if path.is_file()]]
    targets = [qdir, course/'installations/CHD1282-Q-MedSAM2-20261115.json',
               course/'.model-install.lock', course/'.model-assets.lock', course/'installation-work',
               *[course/name for name in configs]]
    if bound != source:
        targets.append(bound)
    for directory in (qdir, bound if bound != source else qdir):
        validate_destination(directory, course, protected)
        if directory.exists():
            targets.extend(directory.iterdir())
            checked_bundle(directory, Q_ID if directory == qdir else SOURCE_ID)
    for target in targets:
        validate_destination(target, course, protected)
        pending = target.with_name(target.name+'.pending')
        validate_destination(pending, course, protected)
        if pending.exists():
            raise FileExistsError('安装目标存在未完成文件：'+str(pending))
    for target in assets.values():
        if target is not None and not Path(target).exists():
            validate_asset_destination(root, code, source, target, roots)
    check_configs(course, configs)
    missing = verify_assets(assets)
    missing.extend(dict(asset=dataset, path=str(path), preparation='existing_dataset')
                   for dataset, path in roots.items() if not Path(path).is_dir())
    preview = dict(state='checked' if not missing else 'missing_requirements', source_identity=info['source_identity'],
                   groups={key: len(value) for key, value in info['groups'].items()},
                   missing=missing, manual_upload_files=[], generated_configs=len(configs),
                   existing_inputs=str(source), existing_stunet=str(assets['stunet']))
    if check:
        return preview
    if missing:
        raise FileNotFoundError(json.dumps(preview, ensure_ascii=False))
    created = []
    course.mkdir(parents=True, exist_ok=True)
    with FileLock(course/'.model-install.lock', timeout=0):
        check_configs(course, configs)
        if qdir.exists():
            checked_bundle(qdir, Q_ID)
        if bound != source:
            if bound.exists():
                checked_bundle(bound, SOURCE_ID)
            else:
                copy_input_metadata(source, bound, descriptor)
                created.append(str(bound.relative_to(course)))
            bind_inputs(bound, root, roots, assets['stunet'])
        q = ensure_q_inputs(source, qdir)
        if not q['reused']:
            created.append(str(qdir.relative_to(course)))
        bind_inputs(qdir, root, roots, assets['stunet'])
        for name, value in configs.items():
            path = course/name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = path.with_name(path.name+'.pending')
                if pending.exists():
                    raise FileExistsError('存在未完成配置：'+str(pending))
                pending.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8', newline='\n')
                pending.replace(path)
                created.append(name)
        receipt = dict(state='installed', source_identity=info['source_identity'], q_identity=Q_ID,
                       groups=preview['groups'], assets={key: str(value) if value is not None else None for key, value in assets.items()},
                       code=str(code), work_root=str(course), manual_upload_files=[],
                       configs=sorted(configs))
        target = course/'installations/CHD1282-Q-MedSAM2-20261115.json'
        if not target.exists() or read_json(target) != receipt:
            write_json(target, receipt)
            created.append(str(target.relative_to(course)))
    return dict(receipt, created_files=created)
