"""Private ImageCHD input descriptor. Discovery does not read reference labels."""
import hashlib
import json
from pathlib import Path

PROJECT_ID = 'chd_v2_stunet_s_seven_structure_v1'
INPUT_PROTOCOLS={PROJECT_ID,'imagechd_stunet_s_seven_structure_v1'}
GROUPS = {'ImageCHD': 'ImageCHD'}
FOLDS = {f'imagechd_fold{i}': (921+i, 'CT', str(i)) for i in range(3)}
QUARANTINE = set()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def discover_cases(data_root):
    root = Path(data_root).expanduser().resolve()
    descriptor = json.loads((root/'imagechd_input.json').read_text(encoding='utf-8-sig'))
    if descriptor.get('protocol') not in INPUT_PROTOCOLS:
        raise ValueError('需要独立ImageCHD输入描述文件，禁止读取Wholeheart或旧64立方缓存')
    cases = descriptor['cases']
    ids = [c['case_id'] for c in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError('病例清单为空或ID重复')
    seen = set()
    for case in cases:
        if case['group'] != 'ImageCHD' or case['modality'] != 'CT':
            raise ValueError('来源不是ImageCHD CT')
        if case['status'] not in ('eligible','quarantined'):
            raise ValueError('未知纳入状态')
        if file_hash(case['image']) != case['image_sha256']:
            raise ValueError('源影像SHA256已变化')
        if case['image_sha256'] in seen:
            raise ValueError('重复影像，需要先核实病例身份')
        seen.add(case['image_sha256'])
        spacing = case['spacing_mm']
        if len(spacing) != 3 or any(not isinstance(x,(int,float)) or not 0 < x < 100 for x in spacing):
            raise ValueError('缺少有效作者间距')
        if case['spacing_source'] != 'imagechd_author_excel':
            raise ValueError('没有作者间距来源')
    make_splits(cases)
    return cases


def make_splits(cases, seed=20261029):
    eligible = [c for c in cases if c['status']=='eligible']
    ids = {c['case_id'] for c in eligible}
    if not ids or len(ids) != len(eligible):
        raise ValueError('纳入清单为空或重复')
    result = {}
    for index, (fold, (dataset, modality, holdout)) in enumerate(FOLDS.items()):
        parts = {k:sorted(c['case_id'] for c in eligible if c['roles'][str(index)]==role)
                 for k,role in [('train','train'),('val','val'),('target','test')]}
        sets = [set(v) for v in parts.values()]
        if any(not x for x in sets) or set.union(*sets)!=ids or any(sets[i]&sets[j] for i in range(3) for j in range(i)):
            raise ValueError('冻结三折划分缺失、重叠或含未知角色')
        result[fold] = dict(project_id=PROJECT_ID,dataset_id=dataset,modality=modality,holdout=holdout,
            split_seed=seed,label_budget_baseline=len(parts['train'])+len(parts['val']),
            label_budget_research_reserved=0,data_protocol='imagechd_frozen_oof_v1',
            evaluation_scope='case_level_OOF_patient_identity_unverified',
            selector=[],unlabeled=[],fingerprint_fit_ids=parts['train'][:],**parts)
    targets = [set(s['target']) for s in result.values()]
    if sum(map(len,targets))!=len(ids) or set.union(*targets)!=ids:
        raise ValueError('三折目标未恰好覆盖每个纳入病例一次')
    return result


def validate_split(split, cases):
    fold = 'imagechd_fold'+str(split['holdout'])
    if fold not in FOLDS or split != make_splits(cases,split['split_seed'])[fold]:
        raise ValueError('划分与冻结ImageCHD输入不同')


def label_access_ids(split):
    return split['train']+split['val']
