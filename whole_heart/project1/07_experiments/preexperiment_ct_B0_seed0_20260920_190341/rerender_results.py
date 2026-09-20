"""仅重排图像，不更新模型、预测或评价数值；保存重绘来源记录。"""
from pathlib import Path
import sys
import time
import nibabel as nib
import numpy as np
sys.path.insert(0, 'D:/code_project/whole_heart/project1/05_src')
from data_pipeline import ROOT, load_json, map_labels
from local_preexperiment import plot_overlay, build_report
from server_data import digest, write_json

out = ROOT/'07_experiments/preexperiment_ct_B0_seed0_20260920_190341'
r = load_json(out/'run.json')
raw = ROOT/'04_data/derived/nnUNet_raw/Dataset703_ct_holdG'
records = []
for entry in r['volume_previews']:
    case = entry['case_id']
    image = np.asanyarray(nib.load(raw/'imagesTr'/f'{case}_0000.nii.gz').dataobj)
    ref = np.asanyarray(nib.load(raw/'labelsTr'/f'{case}.nii.gz').dataobj)
    pred = map_labels(np.asanyarray(nib.load(entry['prediction']).dataobj))
    slices = plot_overlay(image, ref, pred, entry['figure'], f"{case} SOURCE DEVELOPMENT, epoch {entry['epoch']}")
    assert slices == entry['native_slice_indices']
    records.append({'figure':entry['figure'], 'sha256':digest(entry['figure'])})
    print('Redrawn:',case,entry['epoch'], flush=True)
build_report(out, r)
records.append({'figure':str(out/'learning_curves.png'), 'sha256':digest(out/'learning_curves.png')})
write_json(out/'figure_render_record.json', {'rendered_at':time.strftime('%Y-%m-%d %H:%M:%S'), 'reason':'nnU-Net changed global matplotlib fonts; reset plotting defaults', 'prediction_and_metrics_changed':False, 'training_rerun':False, 'current_plotting_source_sha256':digest(ROOT/'05_src/local_preexperiment.py'), 'original_training_source_sha256':r['code_sha256']['local_preexperiment.py'], 'figures':records})
(out/'rerender_results.py').write_bytes(Path(__file__).read_bytes())
for path in [out/'预实验记录.md', ROOT/'09_reports/20260920_CT前期训练结果.md']:
    text = path.read_text(encoding='utf-8')
    path.write_text(text+'\n\n绘图备注：完成训练后修复了 nnU-Net 全局字体设置导致的图例遮挡，仅重新绘图，没有更新模型、预测或指标。原始训练代码保存在实验目录的 `code_snapshot/`；重绘记录见该目录的 `figure_render_record.json`。\n', encoding='utf-8')
print('All figures redrawn; original training snapshot and numerical records retained.')
