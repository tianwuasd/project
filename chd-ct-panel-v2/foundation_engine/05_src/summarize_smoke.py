"""汇总成功/失败短测并估算正式预算，绝不从短测 Dice 推断方法效果。"""
from pathlib import Path
from statistics import mean
from data_pipeline import ROOT, load_json, save_json
from model_catalog import load_selection


def selected_summary():
    selected = load_selection(ROOT)
    expected = {(f, m) for f in selected['folds'] for m in selected['methods']}
    suites = sorted((ROOT / '09_reports').glob('smoke_suite_*/suite.json'))
    valid = [p for p in suites if {(r['fold'], r['method']) for r in load_json(p)} == expected
             and all(r['status'] == 'passed' for r in load_json(p))]
    if not valid:
        raise RuntimeError('尚没有覆盖所选模型与方向且全部成功的短测')
    runs, rows = {}, []
    for entry in load_json(valid[-1]):
        reports = [s[8:].strip() for s in Path(entry['log']).read_text(encoding='utf-8').splitlines() if s.startswith('REPORT: ')]
        if len(reports) != 1:
            raise ValueError('短测运行报告无法唯一定位')
        run = load_json(reports[0])
        if run['status'] != 'passed' or not run['smoke_only']:
            raise ValueError('短测报告状态不符')
        runs[(entry['fold'], entry['method'])] = run
        epochs = 250 if entry['fold'].startswith('ct_') else 300
        seconds = mean(run['train_seconds_excluding_first_two'])
        rows.append(dict(fold=entry['fold'], method=entry['method'], train_step_seconds=seconds,
                         peak_allocated_gib=run['peak_allocated_gib'],
                         one_training_hours_extrapolated=epochs * (250*seconds + 50*run['validation_step_seconds'])/3600))
    for fold in selected['folds']:
        group = [runs[fold, m] for m in selected['methods']]
        if len({r['initial_weight_sha256'] for r in group}) != 1 or len({r['plan_sha256'] for r in group}) != 1:
            raise ValueError('同方向所选模型的初始化或计划不同')
    value = dict(status='passed', selection=selected, rows=rows,
                 selected_runs_core_hours_extrapolated=sum(r['one_training_hours_extrapolated'] for r in rows),
                 limitation='短测吞吐外推，不包括预处理、完整推理和IO；不证明正式效果或实际完工时间。')
    save_json(ROOT / '09_reports/smoke_summary.json', value)
    lines = ['# 所选模型短测', '', '工程验证，不是方法性能结论。', '']
    lines += [f"- {r['fold']} / {r['method']}：每训练步 {r['train_step_seconds']:.3f} 秒，峰值显存 {r['peak_allocated_gib']:.2f} GiB。" for r in rows]
    lines += ['', value['limitation']]
    (ROOT / '09_reports/短测与预算.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    selected_summary()


if __name__ == "__main__":
    main()
