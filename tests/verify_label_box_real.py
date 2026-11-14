import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter
from unittest import TestCase

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import write_json, file_digest, digest
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.models.workflow import options, predict
from chd_segmentation.models.medsam2.adapter import activate
from chd_segmentation.models.medsam2.prompts import label_box_prompts


def run(args):
    experiments = yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
    experiment = next(item for item in experiments if item['dataset'] == 'imagechd')
    check = TestCase()
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        started = perf_counter()
        row = deepcopy(experiment['case'])
        row['roles'] = ['train', 'validation']
        write_json(workspace.root/'manifest.json', dict(dataset='imagechd', scope='technical',
            data_root=experiment['data_root'], cases=[row], plan_scope='technical'))
        config = deepcopy(DEFAULTS)
        config.update(dataset='imagechd', scope='technical', manifest=str(workspace.root/'manifest.json'),
            pretrained=experiment['pretrained'], work_root=str(workspace.root/'course'), run_dir=str(workspace.root/'course/stu'))
        config['training'].update(patch=[32]*3, attempts=2, batch=1)
        config['technical'] = dict(roi_xyz=experiment['roi_xyz'], allow_role_overlap=True)
        write_json(workspace.root/'prepare.json', config)
        prepared_path = workspace.root/'course/prepared.json'
        prepared = prepare(load_config(workspace.root/'prepare.json'), prepared_path)
        record = dict(case_id=row['case_id'], dataset='imagechd', roi_xyz=experiment['roi_xyz'],
            scope='technical', device='cpu', optimizer_updates=0, source_identity=row['sha256'],
            prepare_seconds=perf_counter()-started)
        workspace.results.append(record)
        model_config = dict(model='medsam2', run_dir=str(workspace.root/'course/medsam2'),
            work_root=str(workspace.root/'course'), device='cpu', technical=True, updates=1,
            batch=1, workers=0, official_root=str(ROOT/'vendor/MedSAM2'),
            dependency_root=str(ROOT/'env/medsam2'), pretrained=str(ROOT/'data/model_weights/MedSAM2_latest.pt'),
            execution=dict(cpu_processes=1, workers=0, prefetch=1))
        write_json(workspace.root/'model.json', model_config)
        applied = options(workspace.root/'model.json')
        activate(applied)
        from chd_segmentation.models.medsam2.data import NativeCases
        cases = NativeCases(prepared, with_label=True)
        entry = cases.get(row['case_id'])
        known = sorted(set(map(int, np.unique(entry['label'])))-{0, -1}-set(entry['unknown_classes']))
        if len(known) < 2:
            raise ValueError('实际子区域需要至少两个已知结构来检查独立传播')
        known.sort(key=lambda class_id:(int((entry['label']==class_id).sum((1,2)).argmax())==0,class_id))
        selected = known[:args.structures]
        prompts, protocol = label_box_prompts(entry, row['sha256']['label'], 8, selected)
        assert (prompts, protocol) == label_box_prompts(entry, row['sha256']['label'], 8, selected)
        for item in prompts[row['case_id']]:
            label = entry['label']
            frame = int((label == item['class_id']).sum((1, 2)).argmax())
            assert frame == item['frame']
            yy, xx = np.where(label[frame] == item['class_id'])
            x0, y0, x1, y1 = item['box_xyxy']
            assert x0 <= xx.min() <= xx.max() < x1 and y0 <= yy.min() <= yy.max() < y1
        with check.assertRaises(ValueError):
            label_box_prompts(entry, row['sha256']['label'], 8, [5])
        cases.clear()
        del cases, entry
        with check.assertRaises(ValueError):
            predict(applied, prepared_path, model_config['pretrained'], workspace.root/'ordinary', official_weights=True)
        predicted = predict(applied, prepared_path, model_config['pretrained'], workspace.root/'result',
            official_weights=True, prompt_mode='label-box', label_classes=selected)
        assert predicted['initialization']['optimizer_updates'] == 0
        assert predicted['cases'][0]['prompt_protocol'] == protocol
        assert set(predicted['evaluation']['case_classes'][row['case_id']]) == set(map(str, selected))
        logs = list((workspace.root/'course/model-predictions').rglob('events.jsonl'))
        assert len(logs) == 2
        import json
        events = [json.loads(line) for path in logs for line in path.read_text(encoding='utf-8').splitlines()]
        assert {'resources_requested','resources_allocated','resources_released','model_case_started',
                'label_box_generated','model_case_completed'} <= {item['event'] for item in events}
        assert any(item['frame']>0 for item in protocol['prompts'])
        assert any(step['reverse'] and step['frames']>1 for step in predicted['cases'][0]['diagnostics']['propagation_steps'])
        for value in predicted['evaluation']['case_classes'][row['case_id']].values():
            assert value['reference_voxels'] > 0 and 0 <= value['dice'] <= 1
        # 普通路径在没有标签的独立准备描述中也必须要求显式外部提示。
        independent = deepcopy(prepared)
        independent['manifest'] = deepcopy(prepared['manifest'])
        for item in independent['manifest']['cases']:
            item['roles'] = ['predict']
            item['label'] = None
            item['sha256'].pop('label')
            item['source_stamp'].pop('label')
        independent.pop('identity')
        independent['identity'] = digest(independent)
        write_json(workspace.root/'no-label.json', independent)
        with check.assertRaises(ValueError):
            predict(applied, workspace.root/'no-label.json', model_config['pretrained'], workspace.root/'no-label-result', official_weights=True)
        for kind in ('image', 'label'):
            assert file_digest(row[kind]) == row['sha256'][kind]
        record.update(prediction=predicted, selected_classes=selected, deterministic_boxes=True,
            ordinary_prompt_guard=True, unknown_class_guard=True, source_unchanged=True,
            seconds=perf_counter()-started,
            prompted_structure_frames_per_second=32*len(selected)/predicted['cases'][0]['inference_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('spec', 'work-root', 'report'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--structures',type=int,choices=(1,2),default=2)
    run(parser.parse_args())
