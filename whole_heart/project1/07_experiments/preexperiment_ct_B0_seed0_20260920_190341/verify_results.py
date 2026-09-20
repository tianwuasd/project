"""独立从保存的NIfTI重算Dice、核对检查点轮数及冻结记录，生成中文摘要。"""
import csv
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
sys.path.insert(0, 'D:/code_project/whole_heart/project1/05_src')
from data_pipeline import map_labels, load_json
from server_data import digest, write_json

ROOT = Path('D:/code_project/whole_heart/project1')
run = ROOT/'07_experiments/preexperiment_ct_B0_seed0_20260920_190341'
r = load_json(run/'run.json')
assert r['status'] == 'completed', r['status']
assert r['baseline_completed_epochs'] == 5 and r['baseline_optimizer_steps'] == 1250
assert r['tiny_completed_steps'] == 200 and len(r['tiny_training_losses']) == 200
assert r['tiny_initial_weight_sha256'] == r['baseline_initial_weight_sha256']
assert len(r['epoch_history']) == 5 and len(r['volume_previews']) == 6
assert not r['target_evaluated']
assert digest(ROOT/'04_data/manifests/splits_v1.json') == r['split_sha256']
prep = ROOT/'04_data/derived/nnUNet_preprocessed/Dataset703_ct_holdG'
assert digest(prep/'Project1Plans.json') == r['plan_sha256']
for name, expected in r['code_sha256'].items():
    assert digest(run/'code_snapshot'/name) == expected
    assert digest(ROOT/'05_src'/name) == expected
names = ['Myo','LA','LV','RA','RV','AO','PA']
checked = []
for entry in r['volume_previews']:
    assert entry['case_id'] in r['split']['val'] and entry['case_id'] not in r['split']['target']
    pred_img = nib.load(entry['prediction'])
    ref_img = nib.load(ROOT/f"04_data/derived/nnUNet_raw/Dataset703_ct_holdG/labelsTr/{entry['case_id']}.nii.gz")
    assert pred_img.shape == ref_img.shape and np.allclose(pred_img.affine, ref_img.affine, atol=1e-4, rtol=0)
    pred = map_labels(np.asanyarray(pred_img.dataobj))
    ref = np.asanyarray(ref_img.dataobj).astype(np.int16)
    # 与运行时逐类布尔掩码不同，采用8x8混淆矩阵独立计算。
    counts = np.bincount((ref.ravel()*8 + pred.ravel()).astype(np.int64), minlength=64).reshape(8,8)
    values = []
    for label, name in enumerate(names, 1):
        denominator = int(counts[label,:].sum()) + int(counts[:,label].sum())
        dice = 2*int(counts[label,label])/denominator if denominator else None
        actual = entry['dice'][name]
        assert (dice is None and actual is None) or (dice is not None and actual is not None and abs(dice-actual)<1e-12)
        if dice is not None:
            values.append(dice)
    assert abs(float(np.mean(values))-entry['macro_dice']) < 1e-12
    assert Path(entry['figure']).stat().st_size > 10000
    checked.append({'epoch':entry['epoch'], 'case_id':entry['case_id'], 'dice_independent_recalculation':'passed', 'grid_check':'passed'})
    print('Verified volume:', entry['epoch'], entry['case_id'], flush=True)
    del pred, ref, counts
checkpoint = torch.load(run/'baseline/fold_0/checkpoint_preexperiment.pth', map_location='cpu', weights_only=False)
assert checkpoint['current_epoch'] == 5
assert checkpoint['optimizer_state']['state']
assert all(torch.isfinite(t).all().item() for t in checkpoint['network_weights'].values() if t.is_floating_point())
verification = {'status':'passed', 'type':'output consistency verification; not an independent training replication', 'checkpoint_completed_epochs': checkpoint['current_epoch'], 'optimizer_state_present':True, 'weights_finite':True, 'source_split_and_plan_unchanged':True, 'code_snapshot_hashes_match':True, 'volumes':checked}
verification['verification_script_sha256'] = digest(__file__)
if '--write' not in sys.argv:
    print(json.dumps(verification, indent=2))
    raise SystemExit(0)
write_json(run/'output_verification.json', verification)
(run/'verify_results.py').write_bytes(Path(__file__).read_bytes())
tiny = r['tiny_probe_history']
rows = []
for case in r['fixed_preview_cases']:
    vals = {e['epoch']: e for e in r['volume_previews'] if e['case_id']==case}
    rows.append(f"| {case} | {vals[0]['macro_dice']:.4f} | {vals[1]['macro_dice']:.4f} | {vals[5]['macro_dice']:.4f} |")
last = [e for e in r['volume_previews'] if e['epoch']==5]
worst = min(((e['dice'][n], e['case_id'], n) for e in last for n in names if e['dice'][n] is not None))
content = f'''# 本机 CT 前期训练结果（2026-09-20）

## Material Passport

Origin: academic_tools / experiment-agent；Status: 实际执行完成，输出一致性已核验；未独立重复训练。

本次用 RTX 5060 Ti 完成 **200 步固定训练块拟合检查 + 5 轮 CT 基线预实验**，运行耗时 **{r['elapsed_seconds']/60:.1f} 分钟**，不含入口实现与交付后核验。仅 B0、ct_holdG、seed0，没有训练 MRI 或增强 B1，也没有启动完整正式训练。

## 看到了什么

固定训练块平均 Dice：**{tiny[0]['same_training_patch_mean_dice']:.4f} → {tiny[-1]['same_training_patch_mean_dice']:.4f}**。这说明参数更新能学习这些固定样本；仅为两例中的四个三维块，不代表整例拟合成功，更不代表泛化能力。

随后重新从同样的随机初始化开始 B0：30 例来源训练，8 例来源开发，每轮250个训练步骤和50个开发小块验证步骤。实际完成1250个基线优化步骤，保留原250轮学习率调度，只截取前5轮。

两个预先选定的来源开发病例完整体积七类平均 Dice：

| 来源开发病例 | 未训练 | 第1轮 | 第5轮 |
|---|---:|---:|---:|
{chr(10).join(rows)}

这不是分类准确率或官方体积加权WHS分数。数字是七个部位完整体积Dice的简单平均；本次未做血管截断、后处理或HD计算。第5轮所列逐类结果中最低项为 **{worst[1]} / {worst[2]}：{worst[0]:.4f}**，应结合对比图查看错误，不能只看平均值。

## 如何看记录

- [完整预实验记录与全部对比图](../07_experiments/{run.name}/预实验记录.md)
- [学习曲线](../07_experiments/{run.name}/learning_curves.png)
- [每例每部位 Dice 表](../07_experiments/{run.name}/volume_dice.csv)
- [完整参数、时间、设备、病例和代码哈希](../07_experiments/{run.name}/run.json)
- [控制台日志](../07_experiments/{run.name}/console.log)
- [输出核验记录](../07_experiments/{run.name}/output_verification.json)

![学习曲线](../07_experiments/{run.name}/learning_curves.png)

## 解释与限制

- 训练损失采用含负Dice项的组合损失，因此出现负数并不意味着程序出错。
- 原生训练日志的 Pseudo Dice 来自开发小块，与上表完整病例Dice的计算范围不同，不能直接混用。
- 完整预测只覆盖两个提前选定的来源开发病例；全8例开发集完整体积、MRI、目标中心和多随机种子均未评价。
- 200步排错的权重没有用于基线初始化；代码、计划、划分的哈希和保存的代码快照已核对。
- 已从保存的6份预测文件用独立混淆矩阵算法重算逐类Dice，均与记录一致；已核对三维网格、检查点实际完成5轮、优化器状态与权重有限性。这不是独立重复实验。
- 本次只能作为继续排查和观察学习趋势的依据，不能报告跨中心性能、增强收益或临床可用性。
'''
(ROOT/'09_reports/20260920_CT前期训练结果.md').write_text(content, encoding='utf-8')
print('Verification and readable summary saved.')
