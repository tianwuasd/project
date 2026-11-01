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
    lines = ['# ImageCHD 七结构外折评价', '',
             '模型完成并冻结后才评价外折；内部验证集 EMA Dice 选择 best 检查点。',
             'raw 为主要结果；lcc 是预先指定的逐类最大连通块（26邻域）次要分析，不用于选择模型。',
             '额外源标签区域忽略；间距来自作者Excel，原点/方向沿用源头信息；距离值不等于官方排行榜成绩。', '',
             '| 方法 | 输出 | 逐例平均Dice | 折等权Dice | 有定义病例数 |',
             '|---|---|---:|---:|---:|']
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
                methods[method] = dict(case_mean_dice=defined_mean(r['macro_dice'] for r in subset),
                                       fold_equal_dice=defined_mean(folds.values()),
                                       worst_fold_dice=min(defined_centers) if defined_centers else None, per_fold=folds,
                                       defined_folds=len(defined_centers), total_folds=len(folds),
                                       undefined_case_dice=sum(r['macro_dice'] is None for r in subset),
                                       both_empty_class_cases=sum(r['both_empty_classes'] for r in subset),
                                       one_empty_class_cases=sum(r['one_empty_classes'] for r in subset),
                                       pooled_class_hd_mm=distance_summary([r['hd_mm'] for r in distances]),
                                       pooled_class_hd95_mm=distance_summary([r['hd95_mm'] for r in distances]))
            result[f'{modality}_{variant}'] = dict(methods=methods, paired=pair)
            for method,value in methods.items():
                mean=value['case_mean_dice'];equal=value['fold_equal_dice']
                fmt=lambda v:f'{v:.4f}' if v is not None else '无定义'
                valid=sum(r['macro_dice'] is not None for r in rows if r['method']==method)
                lines.append(f"| {method} | {variant} | {fmt(mean)} | {fmt(equal)} | {valid} |")
    lines += ['', '## 如何看结果', '',
              'Dice越高越好。逐折与最差折结果见summary.json；逐病例和七结构明细见两个CSV。各类别零分和空预测需单独复核。',
              '单侧结构缺失计Dice=0、HD/HD95=inf，绝不删掉失败病例；双侧缺失记空并报告数量。病例Dice对有定义结构取均值，CSV保留实际分母。',
              '距离汇总为病例×结构的描述性均值，不是官方体积加权WHS。存在inf时均值保持inf。',
              'S1为预训练微调，S0为同架构随机初始化。只汇总实际选定的方法；当前seed0不能量化训练随机性。病例身份尚未完成患者层面去重。',
              '不要根据这些目标结果反复修改方法后覆盖本次实验。']
    class_failure_summary={}
    for method in sorted({r['method'] for r in class_rows}):
        for variant in ['raw','lcc']:
            for structure in sorted({r['structure'] for r in class_rows}):
                rows=[r for r in class_rows if r['method']==method and r['variant']==variant and r['structure']==structure]
                scores=[r['dice'] for r in rows]
                class_failure_summary[f'{method}/{variant}/{structure}']=dict(mean_dice=defined_mean(scores),
                    zero_overlap=[r['case_id'] for r in rows if r['dice']==0],
                    below_0_1=[r['case_id'] for r in rows if r['dice'] is not None and r['dice']<.1],
                    empty_prediction=[r['case_id'] for r in rows if r.get('predicted_voxels')==0],
                    defined=sum(v is not None for v in scores),total=len(rows))
    write_json(report / 'summary.json', dict(frozen_sha256=sha, results=result,classes=class_failure_summary))
    (report / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'基线评价汇总完成：{report}', flush=True)


if __name__ == '__main__':
    from data_pipeline import ROOT
    summarize(ROOT)
