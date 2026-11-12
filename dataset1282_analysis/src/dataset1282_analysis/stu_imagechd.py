from pathlib import Path
import json
import hashlib

import nibabel as nib
import numpy as np
from monai.data import Dataset
from monai.transforms import Compose


PROTOCOL = 'imagechd_stunet_s_seven_structure_v1'
V2_PROTOCOL = 'chd_v2_stunet_s_seven_structure_v1'
LABEL_SPACE = 'imagechd_0_7_ignore_minus1'


def imagechd_cross_entropy(logits, target):
    import torch.nn.functional as functional

    if logits.ndim != 5 or logits.shape[1] != 8:
        raise ValueError('ImageCHD需要8类三维网络输出')
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError('ImageCHD目标与网络输出形状不一致')
    if not (((target >= 0) & (target <= 7)) | (target == -1)).all():
        raise ValueError('ImageCHD目标标签超出0..7/-1')
    if not (target != -1).any():
        raise ValueError('ImageCHD目标块没有有效监督体素')
    if target.is_floating_point() and not (target == target.round()).all():
        raise ValueError('ImageCHD目标标签必须为整数')
    return functional.cross_entropy(logits, target.long(), ignore_index=-1)


class STUImageCHDDataset:
    # 独立只读入口：不创建缓存、不修改冻结折数及源文件。
    label_space = LABEL_SPACE
    ignore_label = -1
    num_classes = 8

    def __init__(self, descriptor, *, fold, role):
        if fold not in (0, 1, 2) or role not in ('train', 'val', 'test'):
            raise ValueError('需要明确的三折编号和train/val/test角色')
        self.descriptor = Path(descriptor).resolve(strict=True)
        record = json.loads(self.descriptor.read_text(encoding='utf-8-sig'))
        self.protocol = record.get('protocol')
        if self.protocol not in (PROTOCOL, V2_PROTOCOL):
            raise ValueError('需要独立ImageCHD STU输入描述文件')
        if self.protocol == V2_PROTOCOL:
            source = record.get('source_provenance', {})
            if (source.get('builder') != 'imagechd_raw_v1'
                    or source.get('split_algorithm') != 'original_case_level_threefold_v1'):
                raise ValueError('ImageCHD v2来源或病例划分不符合约定')
        cases = record['cases']
        ids = [c['case_id'] for c in cases]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('病例清单为空或ID重复')
        for case in cases:
            if case['group'] != 'ImageCHD' or case['modality'] != 'CT':
                raise ValueError('需要ImageCHD CT数据来源')
            if case['status'] not in ('eligible', 'quarantined'):
                raise ValueError('未知病例状态')
            if case['status'] == 'eligible' and any(
                    case['roles'].get(str(i)) not in ('train', 'val', 'test')
                    for i in range(3)):
                raise ValueError('缺少冻结三折角色')
        self.cases = tuple(c for c in cases if c['status'] == 'eligible'
                           and c['roles'][str(fold)] == role)
        if not self.cases:
            raise ValueError('指定折及角色没有纳入病例')
        self.reader = Dataset(self.cases, transform=Compose([self._load_case]))

    def __len__(self):
        return len(self.cases)

    def _path(self, value):
        path = Path(value)
        if not path.is_absolute():
            path = self.descriptor.parent / path
        return path.resolve(strict=True)

    def __getitem__(self, index):
        return self.reader[index]

    def _load_case(self, case):
        image_path, label_path = self._path(case['image']), self._path(case['label'])
        with image_path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != case['image_sha256']:
            raise ValueError('ImageCHD源影像SHA256不符')
        if self.protocol == V2_PROTOCOL:
            with label_path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != case['label_sha256']:
                raise ValueError('ImageCHD源标签SHA256不符')
        image, label = nib.load(image_path), nib.load(label_path)
        if len(image.shape) != 3 or image.shape != label.shape or not np.allclose(
                image.affine, label.affine, rtol=1e-6, atol=1e-7):
            raise ValueError('ImageCHD影像与标签空间不一致')
        values = image.get_fdata(dtype=np.float32, caching='unchanged')
        labels = label.get_fdata(dtype=np.float32, caching='unchanged')
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError('ImageCHD存在非有限数值')
        if (labels.min() < 0 or labels.max() > 255
                or not np.equal(labels, np.floor(labels)).all()):
            raise ValueError('ImageCHD原始标签必须是0..255有限整数')
        present = {int(v) for v in np.unique(labels)}
        extras = sorted(v for v in present if v > 7)
        if extras != sorted(case['extra_label_ids']):
            raise ValueError('ImageCHD附加标签与冻结描述不一致')
        if self.protocol == V2_PROTOCOL:
            missing = sorted(set(range(1, 8)) - present)
            if missing != sorted(case['missing_label_ids']):
                raise ValueError('ImageCHD缺少标签记录与源标签不一致')
        spacing = np.asarray(case['spacing_mm'], dtype=np.float64)
        if (case['spacing_source'] != 'imagechd_author_excel' or spacing.shape != (3,)
                or not np.isfinite(spacing).all() or not ((spacing > 0) & (spacing < 100)).all()):
            raise ValueError('缺少有效作者间距')
        source_affine = image.affine.copy()
        norms = np.linalg.norm(source_affine[:3, :3], axis=0)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('ImageCHD源方向矩阵无效')
        working_affine = source_affine.copy()
        working_affine[:3, :3] = source_affine[:3, :3] / norms * spacing
        mapped = np.where(labels <= 7, labels, -1).astype(np.int16)
        return dict(case_id=case['case_id'], image=values[None],
                    label=mapped[None], spacing_xyz=spacing,
                    source_affine=source_affine, working_affine=working_affine, label_space=LABEL_SPACE,
                    ignore_label=-1)
