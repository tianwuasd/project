"""Runtime-only CT inventory. Discovery never opens a label file."""
import hashlib
from pathlib import Path

PROJECT_ID = 'vista3d_wholeheart_nnunet_v1'
GROUPS = {'A': 'A ct_train', 'B': 'B ct_train', 'G': 'G ct_train'}
FOLDS = {'ct_holdA': (811, 'CT', 'A'), 'ct_holdB': (812, 'CT', 'B'), 'ct_holdG': (813, 'CT', 'G')}
QUARANTINE = {'Case2009', 'Case2017'}


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def discover_cases(data_root):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError('请显式指定 Wholeheart_Train_Dataset 目录')
    result = []
    for group, prefix in [('A', 1000), ('B', 2000), ('G', 7000)]:
        folder = root / GROUPS[group]
        expected = {f'Case{prefix+n}_{kind}.nii.gz' for n in range(1, 21) for kind in ['image', 'label']}
        actual = {p.name for p in folder.glob('Case*.nii.gz')}
        if actual != expected:
            raise ValueError(f'{GROUPS[group]} 文件不完整或含未知病例: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
        for n in range(1, 21):
            cid = f'Case{prefix+n}'
            image, label = folder / f'{cid}_image.nii.gz', folder / f'{cid}_label.nii.gz'
            if not image.is_file() or not label.is_file():
                raise ValueError(f'病例文件缺失: {cid}')
            result.append(dict(case_id=cid, group=GROUPS[group], modality='CT',
                               image=str(image.resolve()), label=str(label.resolve()),
                               image_sha256=file_hash(image), patient_id=None,
                               status='quarantined' if cid in QUARANTINE else 'eligible'))
    _eligible(result)
    return result


def _eligible(cases):
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('重复病例编号')
    eligible = [c for c in cases if c['status'] == 'eligible']
    for field in ['image_sha256', 'patient_id']:
        values = [c[field] for c in eligible if c.get(field)]
        if len(values) != len(set(values)):
            raise ValueError(f'重复 {field}，需先核实患者并按患者分组')
    for group, count in [('A', 20), ('B', 18), ('G', 20)]:
        if sum(c['group'] == GROUPS[group] and c['modality'] == 'CT' for c in eligible) != count:
            raise ValueError(f'CT 中心 {group} 的合格数量应为 {count}')
    if any(c['case_id'] in QUARANTINE for c in eligible):
        raise ValueError('隔离病例不得进入实验')
    return eligible


def make_splits(cases, seed=20260919):
    eligible = _eligible(cases)
    result = {}
    for fold, (dataset_id, modality, holdout) in FOLDS.items():
        parts = {p: [] for p in ['train', 'val', 'selector', 'unlabeled', 'target']}
        for group in GROUPS:
            ids = [c['case_id'] for c in eligible if c['group'] == GROUPS[group]]
            ids.sort(key=lambda cid: hashlib.sha256(f'{seed}:{cid}'.encode()).hexdigest())
            if group == holdout:
                parts['target'] += ids
            else:
                # Fixed per-center development pool; selector/cal disjoint from fit.
                parts['selector'] += ids[:2]
                parts['val'] += ids[2:4]
                parts['train'] += ids[4:8]
                parts['unlabeled'] += ids[8:]
        result[fold] = dict(project_id=PROJECT_ID, dataset_id=dataset_id, modality=modality,
                            holdout=holdout, split_seed=seed, label_budget_baseline=12,
                            label_budget_research_reserved=16,
                            **{k: sorted(v) for k, v in parts.items()})
        result[fold]['fingerprint_fit_ids'] = result[fold]['train'][:]
    return result


def validate_split(split, cases):
    fold = f"ct_hold{split['holdout']}"
    if fold not in FOLDS or split != make_splits(cases, split['split_seed'])[fold]:
        raise ValueError('划分与冻结的少标注协议不一致')


def label_access_ids(split):
    return split['train'] + split['val']
