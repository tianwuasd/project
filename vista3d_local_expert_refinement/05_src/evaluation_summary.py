"""从已校验的逐病例记录生成CSV与中文报告，不重新训练或选择模型。"""
import csv
import math
from pathlib import Path
from server_data import digest, read_json, write_json


def defined_mean(values):
    values = [v for v in values if v is not None]
    return float(sum(values)/len(values)) if values else None


def distance_summary(values):
    defined = [float(v) for v in values if v is not None]
    value = float(sum(defined)/len(defined)) if defined else None
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
    lines = ['# 所选模型的中心留出评价', '',
             '模型全部完成并冻结后才评价目标中心；来源开发集 EMA Dice 选择 best 检查点。',
             'raw 为主要结果；lcc 是预先指定的逐类最大连通块（26邻域）次要分析，不用于选择模型。',
             '完整标注范围，无官方血管截断；HD(max)与HD95均为毫米，不能冒充官方排行榜。', '',
             '| 模态 | 输出 | B0 中心等权 Dice |',
             '|---|---|---:|']
    for modality in ['CT']:
        for variant in ['raw', 'lcc']:
            rows = [r for r in case_rows if r['modality'] == modality and r['variant'] == variant]
            if not rows:
                continue
            selected_methods = sorted({r['method'] for r in rows})
            pair = None
            methods = {}
            for method in selected_methods:
                subset = [r for r in rows if r['method'] == method]
                folds = {fold: defined_mean([r['macro_dice'] for r in subset if r['fold'] == fold])
                         for fold in sorted({r['fold'] for r in subset})}
                distances = [r for r in class_rows if r['modality'] == modality and r['variant'] == variant and r['method'] == method]
                defined_centers = [v for v in folds.values() if v is not None]
                methods[method] = dict(center_equal_dice=defined_mean(folds.values()),
                                       worst_center_dice=min(defined_centers) if defined_centers else None, per_center=folds,
                                       defined_centers=len(defined_centers), total_centers=len(folds),
                                       undefined_case_dice=sum(r['macro_dice'] is None for r in subset),
                                       both_empty_class_cases=sum(r['both_empty_classes'] for r in subset),
                                       one_empty_class_cases=sum(r['one_empty_classes'] for r in subset),
                                       pooled_class_hd_mm=distance_summary([r['hd_mm'] for r in distances]),
                                       pooled_class_hd95_mm=distance_summary([r['hd95_mm'] for r in distances]))
            result[f'{modality}_{variant}'] = dict(methods=methods, paired=pair)
            value = methods['B0']['center_equal_dice']
            values = [f'{value:.4f}' if value is not None else '无定义（见计数）']
            lines.append(f"| {modality} | {variant} | {values[0]} |")
    lines += ['', '## 如何看结果', '',
              'Dice越高越好。逐中心与最差中心结果见summary.json；逐病例和七结构明细见两个CSV。',
              '单侧结构缺失计Dice=0、HD/HD95=inf，绝不删掉失败病例；双侧缺失记空并报告数量。病例Dice对有定义结构取均值，CSV保留实际分母。',
              '距离汇总为病例×结构的描述性均值，不是官方体积加权WHS。存在inf时均值保持inf。',
              '当前只有B0和seed0；不报告方法差异或训练随机性的置信区间。不能证明对新中心普遍有效。',
              '不要根据这些目标结果反复修改方法后覆盖本次实验。']
    write_json(report / 'summary.json', dict(frozen_sha256=sha, results=result))
    (report / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'基线评价汇总完成：{report}', flush=True)


if __name__ == '__main__':
    from data_pipeline import ROOT
    summarize(ROOT)
