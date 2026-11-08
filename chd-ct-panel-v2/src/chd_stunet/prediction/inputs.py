import math
from pathlib import Path
import pickle
import re

import nibabel as nib
import numpy as np

from ..observability.runlog import write_json
from ..preprocessing.cohort import TARGETS, frozen_roles
from ..preprocessing.stunet_s import corrected_affine
from ..training.contracts import digest
from .model import file_sha, read_json


def validate_test_role(row, model):
    cid = row['case_id']
    if row.get('cohort_group', 'main') != 'main' or row['roles'].get(str(model['fold'])) != 'test':
        raise ValueError('病例不是对应折的主队列留出测试病例：'+cid)
    data = model['signature']['data']
    if cid in set(data['train_ids']) | set(data['val_ids']):
        raise ValueError('测试病例进入过来源模型的训练或验证：'+cid)
    sha = row.get('source_hashes', {}).get('image')
    if sha and any(item['source_hashes']['image'] == sha for item in data['cases']):
        raise ValueError('测试影像与训练/验证来源重复：'+cid)


def select_cases(prepared, models, selected=(), split='test'):
    prepared = Path(prepared).resolve(strict=True)
    manifest = read_json(prepared/'manifests/imagechd_input.json')
    producer = read_json(prepared/'signature.json')
    if read_json(prepared/'dataset.json')['labels'] != TARGETS:
        raise ValueError('预处理类别映射不匹配')
    all_rows = manifest['cases'] + manifest['excluded']
    if len(all_rows) != 110 or len({r['case_id'] for r in all_rows}) != 110:
        raise ValueError('原队列清单不完整或有重复病例')
    complete = [r['case_id'] for r in all_rows if not r.get('missing_label_ids')]
    if len(complete) != 97:
        raise ValueError('历史完整标注队列不匹配')
    historical = frozen_roles(complete, producer['method']['split_seed'])
    wanted = set(selected)
    if len(wanted) != len(selected):
        raise ValueError('病例选择重复')
    result = []
    for model in models:
        data = model['signature']['data']
        if manifest['identity'] != data['preprocessing_identity'] or producer['identity'] != manifest['identity']:
            raise ValueError('预处理运行与来源训练不一致')
        original_plan = read_json(prepared/f"fold_{model['fold']}"/'preprocessing_plan.json')
        if original_plan != data['plan']:
            raise ValueError('预处理计划与训练快照不一致')
        for row in manifest['cases']:
            if row.get('cohort_group', 'main') != 'main':
                continue
            if row['roles'] != historical[row['case_id']]:
                raise ValueError('历史病例划分发生变化：'+row['case_id'])
            if row['roles'][str(model['fold'])] != split:
                continue
            if wanted and row['case_id'] not in wanted:
                continue
            validate_test_role(row, model)
            result.append(dict(row, fold=model['fold'], prepared_root=str(prepared),
                               evaluate=True, source_kind='heldout'))
    if wanted - {r['case_id'] for r in result}:
        raise ValueError('所选病例不属于所选模型的测试集：'+str(sorted(wanted-{r['case_id'] for r in result})))
    if not result:
        raise ValueError('没有可预测的留出测试病例')
    return sorted(result, key=lambda r: (r['fold'], r['case_id']))


def new_cases(paths, models):
    result = []
    seen = set()
    for path in paths:
        path = Path(path).resolve(strict=True)
        cid = path.name.removesuffix('.nii.gz').removesuffix('.nii')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', cid) or cid in seen:
            raise ValueError('新病例文件名须唯一，且只含英文、数字、下划线或连字符')
        seen.add(cid)
        image = nib.load(path)
        spacing = list(map(float, image.header.get_zooms()[:3]))
        if len(image.shape) != 3 or image.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('新 CT 必须为三维 NIfTI 且空间单位明确为 mm：'+str(path))
        if (not np.isfinite(image.affine).all() or not np.isfinite(spacing).all() or min(spacing) <= 0
                or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12):
            raise ValueError('新 CT 几何无效')
        if not np.allclose(spacing, nib.affines.voxel_sizes(image.affine), rtol=1e-5, atol=1e-6):
            raise ValueError('新 CT 的 header spacing 与 affine 冲突，请核实原始几何：'+str(path))
        sha = file_sha(path)
        for model in models:
            training_cases = model['signature']['data']['cases']
            if any(r['source_hashes']['image'] == sha for r in training_cases):
                raise ValueError('新病例与模型训练/验证影像相同：'+cid)
            result.append(dict(case_id=cid, fold=model['fold'], pair={'image': str(path)},
                               shape_xyz=list(image.shape), source_affine=image.affine.tolist(),
                               spacing_xyz=spacing, spacing_evidence='NIfTI mm header',
                               source_hashes={'image': sha}, source_kind='new', evaluate=False))
    return result


def estimate_case(row, model):
    axes = model['plan']['transpose_forward']
    spacing = np.asarray(row['spacing_xyz'][::-1])[axes]
    shape = np.asarray(row['shape_xyz'][::-1])[axes]
    target = model['plan']['configurations']['3d_fullres']['spacing']
    resampled = [max(1, math.ceil(n*s/t)) for n, s, t in zip(shape, spacing, target)]
    # 包含原图、插值临时空间、8类概率、预取及CPU导出；不把压缩磁盘大小当内存。
    raw = math.prod(row['shape_xyz'])
    return int((raw + math.prod(resampled))*96 + 512*1024**2)


def prepare_case(record, model, destination, threads=1):
    import blosc2
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from threadpoolctl import threadpool_limits
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    image_path = Path(record['pair']['image'])
    before = image_path.stat()
    if file_sha(image_path) != record['source_hashes']['image']:
        raise ValueError('源 CT 摘要改变：'+record['case_id'])
    native = nib.load(image_path)
    if list(native.shape) != record['shape_xyz'] or not np.allclose(native.affine, record['source_affine'], atol=1e-6):
        raise ValueError('源 CT 几何与审计不一致')
    cid = record['case_id']
    completion = None
    if record.get('prepared_root'):
        root = Path(record['prepared_root'])/f"fold_{model['fold']}"
        marker = root/'completion'/f'{cid}.json'
        if marker.is_file():
            completion = read_json(marker)
            contract = digest(dict(run=model['signature']['data']['preprocessing_identity'],
                                   plan=model['signature']['data']['plan']))
            if (not completion['success'] or completion['contract'] != contract or
                    completion['source_hashes'] != record['source_hashes'] or
                    completion['case_id'] != cid or completion['fold'] != model['fold']):
                raise ValueError('预处理完成记录与模型不一致')
            for name in (cid+'.b2nd', cid+'.pkl'):
                if file_sha(root/'data'/name) != completion['file_sha256'][name]:
                    raise ValueError('预处理缓存损坏：'+name)
            array = blosc2.open(str(root/'data'/f'{cid}.b2nd'), mode='r', dparams={'nthreads': 1})[:]
            with (root/'data'/f'{cid}.pkl').open('rb') as stream:
                properties = pickle.load(stream)
            if properties['source_hashes'] != record['source_hashes']:
                raise ValueError('缓存空间属性来源错误')
    if completion is None:
        affine = (native.affine.copy() if record['source_kind'] == 'new'
                  else corrected_affine(native.affine, record['spacing_xyz']))
        properties = dict(nibabel_stuff={'original_affine': affine},
                          spacing=record['spacing_xyz'][::-1], source_affine=native.affine,
                          source_shape_xyz=list(native.shape), source_hashes=record['source_hashes'])
        manager = PlansManager(model['plan'])
        data = native.get_fdata(dtype=np.float32).transpose(2, 1, 0)[None]
        with threadpool_limits(limits=threads):
            array, _, properties = DefaultPreprocessor(verbose=False).run_case_npy(
                data, None, properties, manager, manager.get_configuration('3d_fullres'), model['dataset_json'])
        del data
    if array.ndim != 4 or array.shape[0] != 1 or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('预测输入数组格式或数值无效')
    after = image_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('CT 在读取期间改变')
    properties.pop('class_locations', None)
    properties_path = destination/'properties.pkl'
    with properties_path.open('wb') as stream:
        pickle.dump(properties, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(destination/'image.npy', array)
    result = dict(image=str(destination/'image.npy'), properties=str(properties_path),
                  shape=list(array.shape), reused_preprocessing=completion is not None,
                  image_sha256=file_sha(destination/'image.npy'), properties_sha256=file_sha(properties_path))
    write_json(destination/'prepared.json', result)
    return result
