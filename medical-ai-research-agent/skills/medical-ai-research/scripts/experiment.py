"""Patient-level, numeric-feature binary baseline and ablation runner. No image training."""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(m):
    for key in ['data_kind', 'patient_column', 'target', 'prediction_time', 'outcome', 'features', 'feature_sets']:
        if not m.get(key):
            raise ValueError(f'manifest 缺少 {key}')
    if m['data_kind'] not in ['synthetic', 'real']:
        raise ValueError('data_kind 必须为 synthetic 或 real')
    if m['data_kind'] == 'real' and not m.get('deidentified_and_authorized'):
        raise ValueError('真实数据须声明已脱敏且获准使用')
    blocked = {m['patient_column'], m['target'], m.get('site_column')}
    names = []
    for feature in m['features']:
        name = feature['name']
        if name in blocked or name in names:
            raise ValueError('特征不得含 ID、结局、中心或重复名称')
        if feature.get('available') not in ['pre_index', 'at_index']:
            raise ValueError(f'{name}: available/时点未经确认，拒绝运行')
        if not feature.get('source'):
            raise ValueError(f'{name}: 缺少特征来源')
        names.append(name)
    for label, columns in m['feature_sets'].items():
        if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(names):
            raise ValueError(f'{label}: 特征集合不在白名单或重复')
    if m.get('external_site') is not None and not m.get('site_column'):
        raise ValueError('external_site 需要 site_column')
    if not 20 <= m.get('bootstrap', 200) <= 10000:
        raise ValueError('bootstrap 必须在 20 到 10000 之间')
    if len(m['feature_sets']) > 10:
        raise ValueError('首版最多 10 个特征组合，修改研究协议后另建批次')


def make_split(ids, y, sites, external_site, seed):
    patients = np.unique(ids)
    labels, centers = [], []
    for patient in patients:
        mask = ids == patient
        if len(set(y[mask])) != 1:
            raise ValueError('同患者结局 outcome 不一致；该工具只支持单一患者级二分类终点')
        if len(set(sites[mask])) != 1:
            raise ValueError('同一患者跨中心 site；先合并/排除跨院重叠，再冻结划分')
        labels.append(y[mask][0]); centers.append(sites[mask][0])
    labels, centers = np.array(labels), np.array(centers)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('结局必须有 0/1 两类')
    if external_site is not None:
        test = patients[centers == str(external_site)]
        dev = patients[centers != str(external_site)]
    else:
        dev, test = train_test_split(patients, test_size=0.2, random_state=seed, stratify=labels)
    lookup = dict(zip(patients.tolist(), labels.tolist()))
    if len(dev) < 10 or len(test) < 4:
        raise ValueError('患者数不足以执行训练/验证/测试分离')
    train, val = train_test_split(dev, test_size=0.25, random_state=seed,
                                 stratify=[lookup[p] for p in dev])
    split = {'train': train.tolist(), 'validation': val.tolist(), 'test': test.tolist()}
    for name, selected in split.items():
        if set(lookup[p] for p in selected) != {0, 1}:
            raise ValueError(f'{name} 仅一个类别，指标不可估计；需重新设计研究划分')
    return split


def aggregate(ids, y, probabilities):
    patients = np.unique(ids)
    return patients, np.array([y[ids == p][0] for p in patients]), np.array([
        probabilities[ids == p].mean() for p in patients])


def metric_values(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {'auroc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)), 'sensitivity': float(tp / (tp + fn)),
            'specificity': float(tn / (tn + fp)), 'prevalence': float(y.mean()),
            'patients': len(y)}


def metrics(y, p, threshold, seed, bootstrap):
    value = metric_values(y, p, threshold)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        index = rng.integers(0, len(y), len(y))
        if len(set(y[index])) == 2:
            estimates.append(roc_auc_score(y[index], p[index]))
    value['auroc_ci95'] = np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else None
    value['bootstrap_valid'] = len(estimates)
    value['bootstrap_requested'] = bootstrap
    return value


def pipeline(family, param, seed):
    model = {'dummy': lambda: DummyClassifier(strategy='prior'),
             'logistic': lambda: LogisticRegression(C=param, max_iter=1500, random_state=seed),
             'random_forest': lambda: RandomForestClassifier(
                 n_estimators=80, max_depth=param, min_samples_leaf=3, n_jobs=1, random_state=seed)}[family]()
    return Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                     ('scaler', StandardScaler()), ('model', model)])


def run_experiment(csv_path, manifest_path, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'结果目录已存在，使用新目录: {output_dir}')
    m = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    validate_manifest(m)
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header or any(not name.strip() for name in header):
            raise ValueError('CSV 表头缺失或包含空列名')
        if len(header) != len(set(header)):
            raise ValueError('CSV 存在重复 duplicate 表头，会使患者或特征列被静默替换')
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError('CSV 行字段数与表头不一致')
    if not rows:
        raise ValueError('CSV 为空')
    columns = [feature['name'] for feature in m['features']]
    required = columns + [m['patient_column'], m['target']] + ([m['site_column']] if m.get('site_column') else [])
    if not set(required) <= set(rows[0]):
        raise ValueError('CSV 缺少已声明的列')
    ids = np.array([row[m['patient_column']].strip() for row in rows])
    if np.any(ids == ''):
        raise ValueError('患者 ID 不能为空')
    labels = [row[m['target']].strip() for row in rows]
    if not set(labels) <= {'0', '1'}:
        raise ValueError('二分类标签只允许 0/1')
    y = np.array(labels, dtype=int)
    sites = np.array([row[m['site_column']].strip() if m.get('site_column') else 'internal' for row in rows])
    if np.any(sites == ''):
        raise ValueError('中心字段不能为空')
    def number(value):
        if value.strip().lower() in {'', 'na', 'nan', 'null'}:
            return np.nan
        val = float(value)
        if not np.isfinite(val):
            raise ValueError('特征出现无穷值')
        return val
    x = np.array([[number(row[column]) for column in columns] for row in rows])
    seed = int(m.get('seed', 42))
    split = make_split(ids, y, sites, m.get('external_site'), seed)
    masks = {key: np.isin(ids, value) for key, value in split.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    save(output_dir / 'state.json', {'status': 'running', 'data_kind': m['data_kind']})
    save(output_dir / 'manifest.json', m)
    save(output_dir / 'split.json', split)
    started = time.monotonic()
    try:
        results, predictions = [], []
        # Equal total training weight per patient, irrespective of repeated scans.
        train_ids = ids[masks['train']]
        weights = np.array([1 / np.sum(train_ids == pid) for pid in train_ids])
        weights *= len(weights) / weights.sum()
        grids = {'dummy': [None], 'logistic': [0.1, 1.0], 'random_forest': [3, None]}
        for feature_set, selected in m['feature_sets'].items():
            xx = x[:, [columns.index(column) for column in selected]]
            for family, params in grids.items():
                candidates = []
                for param in params:
                    model = pipeline(family, param, seed)
                    model.fit(xx[masks['train']], y[masks['train']], model__sample_weight=weights)
                    _, vy, vp = aggregate(ids[masks['validation']], y[masks['validation']],
                                          model.predict_proba(xx[masks['validation']])[:, 1])
                    candidates.append((float(roc_auc_score(vy, vp)), model, param, vy, vp))
                best = max(candidates, key=lambda item: item[0])
                auc, model, param, vy, vp = best
                thresholds = np.unique(np.r_[0.0, 0.5, vp, 1.0])
                threshold = float(max(thresholds, key=lambda t: balanced_accuracy_score(vy, vp >= t)))
                patient_ids, ty, tp = aggregate(ids[masks['test']], y[masks['test']],
                                               model.predict_proba(xx[masks['test']])[:, 1])
                test_metrics = metrics(ty, tp, threshold, seed, m.get('bootstrap', 200))
                results.append({'feature_set': feature_set, 'family': family, 'selected_parameter': param,
                                'validation_auroc': auc, 'threshold_from_validation': threshold,
                                'candidate_validation_auroc': [{'parameter': item[2], 'auroc': item[0]} for item in candidates],
                                'test': test_metrics})
                predictions.extend({'feature_set': feature_set, 'family': family, 'patient_id': str(pid),
                                    'label': int(label), 'probability': float(prob)}
                                   for pid, label, prob in zip(patient_ids, ty, tp))
        winner = max(results, key=lambda row: row['validation_auroc'])
        result = {'status': 'completed', 'data_kind': m['data_kind'], 'seed': seed,
                  'test_patients': len(split['test']), 'row_count': len(rows), 'patient_count': len(set(ids)),
                  'split_type': 'external_site_holdout' if m.get('external_site') is not None else 'internal_patient_holdout',
                  'models': results, 'selected_by_validation': {k: winner[k] for k in ['feature_set', 'family']},
                  'provenance': {'csv_sha256': sha(csv_path), 'manifest_sha256': sha(manifest_path),
                                 'code_sha256': sha(__file__), 'python': platform.python_version(),
                                 'numpy': np.__version__, 'sklearn': sklearn.__version__},
                  'runtime_seconds': round(time.monotonic() - started, 3),
                  'limitations': ['一次固定划分；探索性基线，非正式临床验证。',
                                  '重复扫描均须在预测时点前；测试按患者均值聚合，预处理按训练扫描拟合。',
                                  'CI仅重采样测试患者，不包含重新训练的方差；测试集不可用于后续选择。',
                                  '字段可用性依据manifest声明，不能替代原始时间戳与数据来源审计。']}
        save(output_dir / 'results.json', result)
        with (output_dir / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['feature_set', 'family', 'patient_id', 'label', 'probability'])
            w.writeheader(); w.writerows(predictions)
        save(output_dir / 'state.json', {'status': 'completed', 'data_kind': m['data_kind']})
        return result
    except BaseException as exc:
        save(output_dir / 'state.json', {'status': 'failed', 'error': str(exc), 'data_kind': m['data_kind']})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = run_experiment(a.data, a.manifest, a.output)
    print(json.dumps({'status': result['status'], 'output': str(a.output),
                      'models': len(result['models']), 'data_kind': result['data_kind']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
