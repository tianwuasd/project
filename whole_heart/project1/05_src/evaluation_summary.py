"""从已校验的逐病例记录生成CSV与中文报告，不重新训练或选择模型。"""
import csv
import math
from pathlib import Path
import numpy as np
from server_data import digest, read_json, write_json


def paired_summary(rows, samples=2000):
    """每个固定目标组内部按病例成对抽样；最终各目标组等权。

    区间仅描述当前这些训练模型与目标组内的病例不确定性，不代表新中心或新种子。
    """
    paired = {}
    for row in rows:
        key = row['fold'], row['case_id']
        values = paired.setdefault(key, {})
        if row['method'] in values or row['macro_dice'] is None:
            raise ValueError('重复记录或未定义病例Dice，无法生成完整配对汇总')
        values[row['method']] = row['macro_dice']
    groups = {}
    for (fold, _), values in paired.items():
        if set(values) != {'B0', 'B1'}:
            raise ValueError('B0/B1病例未完全配对')
        groups.setdefault(fold, []).append(values['B1'] - values['B0'])
    if not groups:
        raise ValueError('没有成对病例')
    rng = np.random.default_rng(20260922)
    simulated = np.zeros(samples)
    for values in groups.values():
        simulated += rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1) / len(groups)
    return dict(mean_delta=float(np.mean([np.mean(v) for v in groups.values()])),
                ci95=np.percentile(simulated, [2.5, 97.5]).tolist(),
                cases=len(paired), groups=len(groups), bootstrap_samples=samples)


def distance_summary(values):
    defined = [float(v) for v in values if v is not None]
    value = float(np.mean(defined)) if defined else None
    return dict(mean='inf' if value is not None and math.isinf(value) else value,
                defined=len(defined), undefined=len(values)-len(defined),
                infinite=sum(math.isinf(v) for v in defined))


def summarize(root):
    root = Path(root)
    folder = root / '08_results/evaluation'
    frozen = read_json(folder / 'frozen_models.json')
    sha = digest(folder / 'frozen_models.json')
    case_rows, class_rows = [], []
    for model in frozen['models']:
        for case_id in model['targets']:
            record = read_json(folder / f"{model['fold']}_{model['method']}" / f'{case_id}.json')
            if record['status'] != 'completed' or record['signature']['frozen_sha256'] != sha:
                raise ValueError('病例评价缺失或与当前冻结模型不匹配')
            for output in record['outputs'].values():
                if digest(output['path']) != output['sha256']:
                    raise ValueError('预测文件校验不符')
            for variant, scores in record['results'].items():
                common = dict(fold=model['fold'], modality=record['modality'], method=model['method'],
                              case_id=case_id, variant=variant)
                case_rows.append(dict(common, **{k: v for k, v in scores.items() if k != 'classes'}))
                for name, score in scores['classes'].items():
                    class_rows.append(dict(common, structure=name, **score))
    report = root / '09_reports/final_evaluation'
    report.mkdir(parents=True, exist_ok=True)
    for name, rows in [('case_metrics.csv', case_rows), ('class_metrics.csv', class_rows)]:
        with (report / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {}
    lines = ['# B0/B1 完整中心留出评价', '',
             '模型全部完成并冻结后才评价目标中心；来源开发集 EMA Dice 选择 best 检查点。',
             'raw 为主要结果；lcc 是预先指定的逐类最大连通块（26邻域）次要分析，两组同规则。',
             '完整标注范围，无官方血管截断；HD(max)与HD95均为毫米，不能冒充官方排行榜。', '',
             '| 模态 | 输出 | B0 中心等权Dice | B1 中心等权Dice | B1−B0 [病例bootstrap 95%区间] |',
             '|---|---|---:|---:|---|']
    for modality in ['CT', 'MRI']:
        for variant in ['raw', 'lcc']:
            rows = [r for r in case_rows if r['modality'] == modality and r['variant'] == variant]
            pair = paired_summary(rows)
            methods = {}
            for method in ['B0', 'B1']:
                subset = [r for r in rows if r['method'] == method]
                folds = {fold: float(np.mean([r['macro_dice'] for r in subset if r['fold'] == fold]))
                         for fold in sorted({r['fold'] for r in subset})}
                distances = [r for r in class_rows if r['modality'] == modality and r['variant'] == variant and r['method'] == method]
                methods[method] = dict(center_equal_dice=float(np.mean(list(folds.values()))),
                                       worst_center_dice=min(folds.values()), per_center=folds,
                                       both_empty_class_cases=sum(r['both_empty_classes'] for r in subset),
                                       one_empty_class_cases=sum(r['one_empty_classes'] for r in subset),
                                       pooled_class_hd_mm=distance_summary([r['hd_mm'] for r in distances]),
                                       pooled_class_hd95_mm=distance_summary([r['hd95_mm'] for r in distances]))
            result[f'{modality}_{variant}'] = dict(methods=methods, paired=pair)
            lines.append(f"| {modality} | {variant} | {methods['B0']['center_equal_dice']:.4f} | {methods['B1']['center_equal_dice']:.4f} | {pair['mean_delta']:+.4f} [{pair['ci95'][0]:+.4f}, {pair['ci95'][1]:+.4f}] |")
    lines += ['', '## 如何看结果', '',
              'Dice越高越好；B1−B0为正表示增强组在本次实验中更好。逐中心与最差中心结果见summary.json；逐病例和七结构明细见两个CSV。',
              '单侧结构缺失计Dice=0、HD/HD95=inf，绝不删掉失败病例；双侧缺失记空并报告数量。病例Dice对有定义结构取均值，CSV保留实际分母。',
              '距离汇总为病例×结构的描述性均值，不是官方体积加权WHS。存在inf时均值保持inf。',
              'bootstrap在每个固定目标组内成对抽取病例，再对目标组等权。只有seed0；区间不覆盖训练随机性，也不能证明对新中心普遍有效。',
              'MRI的C&D是合并组，不能拆成两个独立中心。不要根据这些目标结果反复修改方法后覆盖本次实验。']
    write_json(report / 'summary.json', dict(frozen_sha256=sha, results=result))
    (report / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'评价及配对汇总完成：{report}', flush=True)


if __name__ == '__main__':
    from data_pipeline import ROOT
    summarize(ROOT)
