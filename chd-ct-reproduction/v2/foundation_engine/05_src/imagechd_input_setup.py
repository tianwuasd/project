"""Create a private, deterministic ImageCHD descriptor from read-only source pairs."""
import argparse
import json
import os
from pathlib import Path
import random
import tempfile

from imagechd_cohort import PROJECT_ID, file_hash, make_splits
from imagechd_excel import read_spacing_excel


def frozen_roles(ids, seed=42):
    if seed != 42:
        raise ValueError('自动输入沿用原实验 seed42；其他划分须单独审核')
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    roles = {cid: {} for cid in order}
    for fold in range(3):
        test = set(order[fold::3])
        remainder = [cid for cid in order if cid not in test]
        random.Random(seed + 1009 * (fold + 1)).shuffle(remainder)
        count = max(1, round(len(order) * .1))
        if count >= len(remainder):
            raise ValueError('纳入病例不足以建立非空三折 train/val/test')
        val = set(remainder[:count])
        for cid in order:
            roles[cid][str(fold)] = 'test' if cid in test else 'val' if cid in val else 'train'
    return roles


def build_auto(raw, output, seed=42):
    import nibabel as nib
    import numpy as np
    raw, output = Path(raw).expanduser().resolve(strict=True), Path(output).expanduser().resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError('描述目录须与原始数据分离')
    target = output / 'imagechd_input.json'
    if output.exists() and not target.is_file():
        raise FileExistsError('已有目录不含冻结输入描述，拒绝写入')
    excel = raw / 'imagechd_dataset_image_info.xlsx'
    spacings = read_spacing_excel(excel)
    images = sorted(raw.glob('*_image.nii.gz'))
    labels = sorted(raw.glob('*_label.nii.gz'))
    if not images or len(images) != len(labels):
        raise ValueError('原始目录没有完整的 ImageCHD image/label 文件对')
    cases, seen = [], set()
    for image in images:
        cid = image.name.removesuffix('_image.nii.gz')
        label = raw / (cid + '_label.nii.gz')
        if not label.is_file() or cid not in spacings:
            raise ValueError('文件对或作者间距缺失: ' + cid)
        source, reference = nib.load(image), nib.load(label)
        if (len(source.shape) != 3 or source.shape != reference.shape
                or not np.allclose(source.affine, reference.affine, rtol=0, atol=1e-5)):
            raise ValueError('原始影像/标签网格不一致，须先核查: ' + cid)
        values = np.unique(np.asanyarray(reference.dataobj))
        if (not np.isfinite(values).all() or not np.equal(values, np.round(values)).all()
                or (values < 0).any() or (values > 255).any()):
            raise ValueError('非有限、负值或非整数标签: ' + cid)
        ids = set(map(int, values))
        missing = sorted(set(range(1, 8)) - ids)
        sha = file_hash(image)
        if sha in seen:
            raise ValueError('重复影像SHA256，须先核实身份: ' + cid)
        seen.add(sha)
        cases.append(dict(case_id=cid, group='ImageCHD', modality='CT', patient_id=None,
            image=str(image), label=str(label), image_sha256=sha, label_sha256=file_hash(label),
            status='quarantined' if missing else 'eligible',
            exclusion=('missing_MYO_annotation' if missing == [5] else 'missing_target_annotations') if missing else None,
            missing_label_ids=missing, extra_label_ids=sorted(ids - set(range(8))),
            spacing_mm=spacings[cid]['spacing_mm'], spacing_source='imagechd_author_excel',
            spacing_excel_row=spacings[cid]['excel_row'], roles={}))
    roles = frozen_roles([c['case_id'] for c in cases if c['status'] == 'eligible'], seed)
    for case in cases:
        case['roles'] = roles.get(case['case_id'], {})
    make_splits(cases)
    record = dict(protocol=PROJECT_ID, patient_identity_verified=False, cases=cases,
        source_provenance=dict(raw_root=str(raw), spacing_excel_sha256=file_hash(excel),
            builder='imagechd_raw_v1', split_algorithm='original_case_level_threefold_v1', split_seed=seed))
    if target.is_file():
        if json.loads(target.read_text(encoding='utf-8-sig')) != record:
            raise ValueError('冻结输入与当前源文件/QC/划分不符；保留旧描述，另建输入任务')
        return record
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete nonempty directory. A killed writer leaves no partial
    # descriptor at the configured target; later attempts use their own staging.
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    temporary = staging / 'imagechd_input.json'
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.rename(staging, output)
        except OSError:
            # A simultaneous builder may have committed the same frozen input.
            if not target.is_file() or json.loads(target.read_text(encoding='utf-8-sig')) != record:
                raise
    finally:
        if staging.exists():
            temporary.unlink(missing_ok=True)
            staging.rmdir()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = build_auto(args.raw, args.output, args.seed)
    print(json.dumps(dict(source_pairs=len(result['cases']),
        eligible=sum(c['status'] == 'eligible' for c in result['cases']), patient_identity_verified=False)))
