from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np

from ...artifacts.store import read_json
from ...preprocessing.validation import SourceReader
from .input import rgb_slice, selected_cases, slices_for


def entries(config, training):
    manifest = read_json(config['manifest'])
    if manifest.get('binding_required'):
        raise ValueError('输入清单尚未绑定；先使用工程bind-inputs登记实际数据目录')
    names = training['train_cases']+training['val_cases']
    if len(set(names)) != len(names):
        raise ValueError('训练验证病例重复')
    selected = selected_cases(config, names)
    groups = {role: {r['patient'] for r in selected if r['case_id'] in training[role+'_cases']}
              for role in ('train', 'val')}
    if not all(groups.values()) or groups['train'] & groups['val']:
        raise ValueError('训练验证必须按患者分离')
    sources = {r['case_id']: r for r in manifest['cases']}
    prepared = {}
    if training['prepared']:
        index = read_json(training['prepared'])
        if index.get('format') != 'medsam3_training_slices_v1' or not index.get('samples'):
            raise ValueError('已有预处理切片登记格式不同')
        for row in index['samples']:
            key = (row['case_id'], row['slice'])
            if key in prepared:
                raise ValueError('已有预处理切片重复')
            path = (Path(training['prepared']).parent/row['file']).resolve(strict=True)
            stat = path.stat()
            if not path.is_file():
                raise ValueError('预处理切片必须为普通文件')
            prepared[key] = dict(file=str(path), stamp=[stat.st_size, stat.st_mtime_ns],
                                 source_stamp=row['source_stamp'])
    reader = SourceReader({'technical': {'roi_xyz': None}}, config['resources']['reserve_ram_bytes'])
    result = {'train': [], 'val': []}
    for row in selected:
        role = 'train' if row['case_id'] in training['train_cases'] else 'val'
        if row['role'] != ('train' if role == 'train' else 'development'):
            raise ValueError('本配方使用冻结train/development，禁止改写独立测试角色')
        source = sources[row['case_id']]
        row.update(label=str((Path(config['data_root'])/source['label']).resolve(strict=True)),
                   geometry_policy=source['geometry_policy'])
        if not Path(row['label']).is_relative_to(Path(config['data_root'])):
            raise ValueError('标签路径越出显式数据根')
        image, label, spacing, roi = reader.read_metadata(row)
        row['source_stamp'] = {key: [Path(row[key]).stat().st_size, Path(row[key]).stat().st_mtime_ns]
                               for key in ('image', 'label')}
        classes = [k for k in training['classes'] if k not in row['unknown_classes']]
        if not classes:
            raise ValueError('所选病例没有已知标注类别')
        for k in slices_for(config, row['case_id'], image.shape[2]):
            item = dict(row=row, slice=k, classes=classes, shape_xy=list(image.shape[:2]), image_affine=image.affine.tolist(),
                        label_affine=label.affine.tolist(), spacing_xyz=spacing.tolist(), role=role)
            if prepared:
                stored = prepared[(row['case_id'], k)]
                if stored['source_stamp'] != row['source_stamp']:
                    raise ValueError('已有切片的原始文件身份不同')
                item['prepared'] = stored['file']
                item['prepared_stamp'] = stored['stamp']
            result[role].append(item)
    return result


def read_slice(item, config):
    row = item['row']
    for key, stamp in row['source_stamp'].items():
        current = Path(row[key]).stat()
        if [current.st_size, current.st_mtime_ns] != stamp:
            raise ValueError('登记后的原始影像或标签发生变化')
    if 'prepared' in item:
        stat = Path(item['prepared']).stat()
        if [stat.st_size, stat.st_mtime_ns] != item['prepared_stamp']:
            raise ValueError('登记后的预处理切片发生变化')
        with np.load(item['prepared'], allow_pickle=False) as stored:
            rgb, labels, valid = stored['rgb'].copy(), stored['labels'].copy(), stored['valid'].copy()
    else:
        image = nib.load(row['image'], keep_file_open=True)
        label = nib.load(row['label'], keep_file_open=True)
        rgb = rgb_slice(image, item['slice'], config)
        labels = np.asarray(label.dataobj[:, :, item['slice']]).T
        valid = ~np.isin(labels, row['ignore_label_ids'])
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or
            labels.shape != tuple(reversed(item['shape_xy'])) or labels.shape != rgb.shape[:2] or
            valid.shape != labels.shape or valid.dtype != np.bool_):
        raise ValueError('切片合同需要uint8 RGB、同网格整数标签和bool valid')
    if not np.isfinite(labels).all() or not np.equal(labels, np.rint(labels)).all():
        raise ValueError('标签包含非有限或非整数值')
    if not np.isin(labels, list(range(11))+row['ignore_label_ids']).all():
        raise ValueError('标签包含CHD1282合同以外的类别值')
    if np.isin(labels, row['unknown_classes']).any():
        raise ValueError('标签像素与未知类别声明矛盾')
    if not valid.all() or np.isin(labels, row['ignore_label_ids']).any():
        raise ValueError('官方实例损失尚不支持像素忽略区域；禁止将该区域作为背景监督')
    return item, rgb, labels


@contextmanager
def stream(items, config, workers, prefetch, deadline):
    if not workers:
        yield (read_slice(item, config) for item in items)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='medsam3-read') as pool:
        from time import time
        source = iter(items)
        pending = deque()
        def consume():
            for item in source:
                pending.append(pool.submit(read_slice, item, config))
                if len(pending) >= prefetch:
                    yield pending.popleft().result(timeout=max(.001, deadline-time()))
            while pending:
                yield pending.popleft().result(timeout=max(.001, deadline-time()))
        yield consume()


def datapoint(engine, sample):
    import torch
    from PIL import Image
    from scipy.ndimage import label
    from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAMImage, Object, FindQueryLoaded, InferenceMetadata

    item, rgb, labels = sample
    height, width = labels.shape
    objects, queries = [], []
    for class_id in item['classes']:
        components, count = label(labels == class_id, structure=np.ones((3, 3), dtype=bool))
        ids = []
        for component in range(1, count+1):
            if (len(objects)+1)*1008**2+1008**2*12 > engine.config['workspace_bytes']:
                raise MemoryError('真实实例掩膜构建超过声明的工作区，未丢弃实例')
            mask = components == component
            ys, xs = np.nonzero(mask)
            object_id = len(objects)
            ids.append(object_id)
            objects.append(Object(bbox=torch.tensor([xs.min(), ys.min(), xs.max()+1, ys.max()+1], dtype=torch.float32),
                area=float(mask.sum()), object_id=object_id, segment=torch.from_numpy(mask)))
        queries.append(FindQueryLoaded(query_text=engine.config['prompts'][class_id], image_id=0,
            object_ids_output=ids, is_exhaustive=True, query_processing_order=0,
            inference_metadata=InferenceMetadata(coco_image_id=0, original_image_id=0,
                original_category_id=class_id, original_size=(height, width), object_id=-1, frame_index=0)))
    image = Image.fromarray(rgb)
    point = Datapoint(find_queries=queries, images=[SAMImage(data=image, objects=objects, size=(height, width))], raw_images=[image])
    return engine.transform(point)
