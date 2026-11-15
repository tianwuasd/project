from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

import yaml

from ..artifacts.store import digest, file_digest, read_json, write_json
from .splits import apply_split
from .transfer import manifest_identity


PROTOCOL = 'chd1282-q-scheme-a-v1'
SOURCE_PROTOCOL = 'dataset1282-scheme-a-preprocess-1'


def build_q_split(manifest, cohort, seed=20261104):
    if manifest['dataset'] != 'chd1282' or cohort['protocol'] != SOURCE_PROTOCOL:
        raise ValueError('Q组合需要1282 Scheme A冻结来源')
    if type(seed) is not int or seed < 0:
        raise ValueError('划分种子必须为非负整数')
    source_roles = {'train', 'development', 'source_test', 'target_primary', 'target_secondary'}
    available = {row['case_id']: row for row in manifest['cases']}
    if len(available) != len(manifest['cases']):
        raise ValueError('来源清单存在重复病例')
    selected = {key: deepcopy(row) for key, row in cohort['cases'].items()
                if row['role'] in source_roles and row.get('included', True) and row.get('index_scan', True)}
    if not selected or not set(selected) <= available.keys():
        raise ValueError('冻结病例在来源清单中缺失')
    patient_roles, patient_strata = {}, {}
    strata = defaultdict(list)
    for key, row in selected.items():
        patient = row['patient']
        if not patient or available[key].get('patient') not in (None, patient):
            raise ValueError('患者身份缺失或与冻结来源不符')
        if patient in patient_roles and patient_roles[patient] != row['role']:
            raise ValueError('同一患者跨越原冻结角色')
        patient_roles[patient] = row['role']
        if row['role'] == 'train':
            stratum = (str(row['device']), str(row['age_band']))
            if patient in patient_strata and patient_strata[patient] != stratum:
                raise ValueError('同一训练患者的设备或年龄层不一致')
            patient_strata[patient] = stratum
    for patient, stratum in patient_strata.items():
        strata[stratum].append(patient)
    count = len(patient_strata)
    if count < 2:
        raise ValueError('原训练患者不足以生成非空A/B组')
    # 最大余数分配使分层后的全局人数保持最接近2:1。
    quotas = {key: len(values) * 2 // 3 for key, values in strata.items()}
    remainder = (2 * count + 1) // 3 - sum(quotas.values())
    order = sorted(strata, key=lambda key: (-(len(strata[key]) * 2 % 3), key))
    for key in order[:remainder]:
        quotas[key] += 1
    patients_a = set()
    stratum_audit = []
    for key in sorted(strata):
        patients = sorted(strata[key], key=lambda patient: (hashlib.sha256(
            f'{seed}:{patient}'.encode('utf-8')).hexdigest(), patient))
        patients_a.update(patients[:quotas[key]])
        stratum_audit.append(dict(device=key[0], age_band=key[1], patients=len(patients),
                                  A=quotas[key], B=len(patients)-quotas[key]))
    groups = {key: [] for key in 'ABCD'}
    for key, row in selected.items():
        role = row['role']
        group = ('A' if row['patient'] in patients_a else 'B') if role == 'train' else (
            'C' if role == 'development' else 'D')
        row['q_group'], row['source_role'] = group, role
        groups[group].append(key)
    groups = {key: sorted(values) for key, values in groups.items()}
    if any(not values for values in groups.values()):
        raise ValueError('Q的A/B/C/D组必须全部非空')
    audit = dict(seed=seed, training_patient_ratio=[2, 1], strata=stratum_audit,
                 cases={key: len(values) for key, values in groups.items()},
                 patients={key: len({selected[name]['patient'] for name in values})
                           for key, values in groups.items()},
                 source_roles=dict(Counter(row['source_role'] for row in selected.values())),
                 policy='仅原训练患者细分A/B；验证、测试、排除和索引扫描沿用冻结来源；统计仅A拟合')
    split = dict(protocol=PROTOCOL, source_protocol=SOURCE_PROTOCOL, seed=seed,
                 groups=groups, cases={key: selected[key] for key in sorted(selected)}, audit=audit)
    return split, groups


def create_q_bundle(source, output, seed=20261104):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Q交付目录必须独立且尚不存在')
    descriptor = read_json(source/'bundle.json')
    if digest({k: v for k, v in descriptor.items() if k != 'identity'}) != descriptor['identity']:
        raise ValueError('来源迁移包身份无效')
    for name, expected in descriptor['immutable_files'].items():
        if Path(name).name != name or file_digest(source/name) != expected:
            raise ValueError('来源冻结文件摘要无效')
    info = descriptor['datasets']['chd1282']
    manifest = read_json(source/info['manifest'])
    if manifest_identity(manifest) != info['identity']:
        raise ValueError('来源1282清单身份无效')
    cohort = read_json(source/info['split'])
    split, groups = build_q_split(manifest, cohort, seed)
    legacy = apply_split(manifest, dict(fold=0, split=dict(path=str(source/info['split']),
        format='chd1282_cohort', id_field='case_id', unit='patient')))
    if set(groups['A']+groups['B']) != {r['case_id'] for r in legacy['cases'] if 'train' in r['roles']}:
        raise ValueError('Q训练组合与原冻结训练角色不一致')
    split['provenance'] = dict(bundle_identity=descriptor['identity'],
        cohort_sha256=file_digest(source/info['split']), manifest_identity=info['identity'])
    output.mkdir(parents=True)
    frozen = {}
    for name in (info['split'], 'chd1282_original_cohort.json', 'chd1282_policy.json'):
        shutil.copy2(source/name, output/name)
        frozen[name] = file_digest(output/name)
    # 保留路径可迁移的清单，不复制影像；机器绑定由bind-inputs执行。
    manifest.update(data_root='.', binding_required=True)
    for row in manifest['cases']:
        row.pop('source_stamp', None)
        for kind in ('image', 'label'):
            row[kind] = row['preferred_files'][kind]
    write_json(output/info['manifest'], manifest)
    for name, value in (('q1282_split.json', split), ('q_groups.json', groups)):
        write_json(output/name, value)
        frozen[name] = file_digest(output/name)
    experiment = deepcopy(next(e for e in descriptor['experiments'] if e['config']['dataset'] == 'chd1282'))
    experiment.update(filename='q1282_prepare.yaml', run_name='q1282_prepare')
    experiment['config']['split'].update(path='q1282_split.json', format='q1282', unit='patient')
    experiment['config']['training']['patch'] = [128]*3
    model_configs = []
    for stage, updates in (('base', 40000), ('localizer', 5000), ('local', 20000)):
        config = dict(model='q_route1', device='cuda', precision='fp32', technical=False, seed=20261111,
            run_dir='BIND_INPUTS_REQUIRED', work_root='BIND_INPUTS_REQUIRED', groups='q_groups.json',
            updates=updates, accumulation=4, patch=128, local_size=128, learning_rate=.0002,
            validate_every=1000, method='M07', execution=dict(gpu_count=2, workers=0))
        model_configs.append(dict(filename=f'q1282_{stage}.yaml', stage=stage, config=config))
    new_descriptor = dict(datasets={'chd1282': dict(info, split='q1282_split.json')},
        experiments=[experiment], model_configs=model_configs, immutable_files=frozen,
        weight_relative=descriptor['weight_relative'], source_bundle_identity=descriptor['identity'])
    new_descriptor['identity'] = digest(new_descriptor)
    write_json(output/'bundle.json', new_descriptor)
    for item in [experiment]+model_configs:
        (output/item['filename']).write_text(yaml.safe_dump(item['config'], allow_unicode=True, sort_keys=False), encoding='utf-8')
    report = dict(bundle_identity=new_descriptor['identity'], source_bundle_identity=descriptor['identity'],
                  output=str(output), **split['audit'])
    write_json(output/'q_split_audit.json', report)
    return report
