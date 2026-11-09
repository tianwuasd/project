from pathlib import Path

import numpy as np
from PIL import Image
import torch

from . import PROTOCOL
from .contracts import check_signature, read_json
from .preprocessing import assert_case_files, case_metadata, eligible_starts


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def open_prepared(root, fold, *, smoke=False, train_cases=(), val_cases=()):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root/'manifest.json')
    summary = read_json(root/'summary.json')
    check_signature(manifest)
    if (manifest['protocol'] != PROTOCOL or summary['status'] != 'completed'
            or summary['manifest_identity'] != manifest['identity']):
        raise ValueError('要求完成并匹配的 MedSAM2 预处理目录')
    if not smoke and (not manifest['full_cohort'] or train_cases or val_cases):
        raise ValueError('正式 MedSAM2 训练需要完整审核队列')
    cases = {row['case_id']: row for row in manifest['cases']}
    roles = {cid: row['roles'][str(fold)] for cid, row in cases.items()}
    train = sorted(train_cases or [cid for cid, role in roles.items() if role in ('train', 'aux_train')])
    val = sorted(val_cases or [cid for cid, role in roles.items() if role == 'val'])
    if smoke and not train_cases:
        train = [cid for cid in train if roles[cid] == 'train'][:1] + [
            cid for cid in train if roles[cid] == 'aux_train'][:1]
    if smoke and not val_cases:
        val = val[:1]
    if (not train or not val or len(set(train+val)) != len(train+val)
            or any(roles.get(cid) not in ('train', 'aux_train') for cid in train)
            or any(roles.get(cid) != 'val' for cid in val)):
        raise ValueError('MedSAM2 训练/验证病例与固定折角色不一致')
    metadata = {}
    for cid in train+val:
        marker = case_metadata(root/'cases'/cid, cases[cid], track_files=True)
        if summary['completed'].get(cid) != marker['identity']:
            raise ValueError('MedSAM2 病例未在已提交清单中：'+cid)
        metadata[cid] = marker
    return root, manifest, cases, metadata, train, val, manifest['folds'][str(fold)]


def frame_tensor(frame, size, window):
    low, high = window['low'], window['high']
    # 训练与传播使用同一窗和几何转换；RGB 扩展只发生在被消费的帧。
    values = np.rint(np.clip((frame-low)/(high-low), 0, 1)*255).astype(np.uint8)
    image = Image.fromarray(values).resize((size, size), Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.array(image)).float().div_(255).expand(3, -1, -1)
    return (rgb - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def resized_mask(mask, size):
    image = Image.fromarray(mask.astype(np.uint8, copy=False))
    return np.array(image.resize((size, size), Image.Resampling.NEAREST))


def box_from_mask(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('不能从空标签生成框提示')
    height, width = mask.shape
    x0, x1, y0, y1 = int(x.min()), int(x.max()), int(y.min()), int(y.max())
    return [min(x0, width-2), min(max(x1, x0+1), width-1),
            min(y0, height-2), min(max(y1, y0+1), height-1)]


def xyxy(mask):
    x0, x1, y0, y1 = box_from_mask(mask)
    return [x0, y0, x1, y1]


def sample_clip(folder, marker, row, window, config, seed, *, required_objects=1):
    rng = np.random.default_rng(seed)
    frames, size = config['frames'], config['image_size']
    assert_case_files(folder, marker)
    image = np.load(folder/'ct.npy', mmap_mode='r', allow_pickle=False)
    label = np.load(folder/'labels.npy', mmap_mode='r', allow_pickle=False)
    if list(image.shape) != marker['shape'] or label.shape != image.shape or image.dtype != np.float32 or label.dtype != np.int8:
        raise ValueError('MedSAM2 缓存数组形状或类型错误：'+row['case_id'])
    candidates = eligible_starts(marker['frame_classes'], frames, required_objects)
    # 从合法起点的随机排列取首个合格项，对目标仍可见的起点均匀采样。
    for first in rng.permutation(candidates):
        initial = resized_mask(label[first], size)
        classes = sorted(set(map(int, np.unique(initial))) & set(marker['observed']))
        if len(classes) >= required_objects:
            break
    else:
        raise ValueError('当前尺寸没有合格的训练片段：'+row['case_id'])
    chosen = sorted(map(int, rng.choice(classes, min(len(classes), config['max_objects']), replace=False)))
    indices = list(range(first, first+frames))
    labels = np.stack([initial if offset == 0 else resized_mask(label[z], size)
                       for offset, z in enumerate(indices)])
    valid = labels != 8
    if row['missing_label_ids']:
        # 部分标注的背景没有明确类别，保守地只用有标注的前景区域监督。
        valid &= labels != 0
    masks = np.stack([labels == label_id for label_id in chosen], axis=1)
    boxes = np.asarray([xyxy(initial == label_id) for label_id in chosen], dtype=np.float32)
    images = torch.stack([frame_tensor(image[z], size, window) for z in indices])
    assert_case_files(folder, marker)
    return dict(images=images,
                masks=torch.from_numpy(masks), valid=torch.from_numpy(valid),
                boxes=torch.from_numpy(boxes), classes=chosen, indices=indices, case_id=row['case_id'],
                span_mm=(frames-1)*marker['geometry']['spacing_zyx'][0])


def sequence_ids(ids, epoch, seed):
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(ids))
    return [ids[i] for i in order]


def sample_seed(seed, sequence):
    return int(np.random.SeedSequence([seed, sequence]).generate_state(1)[0])
