import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
from unittest import TestCase

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import file_digest, read_json, write_json
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.preprocessing.transforms import CaseTransform
from chd_segmentation.models.workflow import options
from chd_segmentation.training.engine import code_identity


def signature(value):
    if isinstance(value, np.ndarray):
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): signature(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [signature(item) for item in value]
    return value


Q_FIELDS = ('case_id', 'image', 'label', 'unknown_classes', 'properties', 'source_shape_xyz',
            'source_affine', 'source_units', 'working_affine', 'spacing_xyz', 'roi_xyz')


def source_config(workspace, item, frozen_plan=None):
    row = deepcopy(item['case'])
    row['roles'] = ['train', 'validation']
    write_json(workspace.root/'manifest.json', dict(dataset='imagechd', scope='technical',
        data_root=item['data_root'], cases=[row], plan_scope='technical'))
    config = deepcopy(DEFAULTS)
    config.update(dataset='imagechd', scope='technical', manifest=str(workspace.root/'manifest.json'),
        pretrained=item['pretrained'], work_root=str(workspace.root/'course'),
        run_dir=str(workspace.root/'course/source-logs'))
    config['resources'].update(device='cpu', workers=0, reserve_ram_bytes=2*2**30)
    config['training'].update(patch=[32]*3, batch=1, attempts=2)
    config['technical'] = dict(roi_xyz=item['roi_xyz'], allow_role_overlap=True)
    if frozen_plan is not None:
        write_json(workspace.root/'frozen-plan.json', frozen_plan)
        config['plan'] = str(workspace.root/'frozen-plan.json')
        config['planning']['mode'] = 'existing'
    write_json(workspace.root/'source-config.json', config)
    applied = load_config(workspace.root/'source-config.json')
    source_path = workspace.root/'course/source.json'
    prepared = prepare(applied, source_path)
    return applied, prepared, source_path


def model_config(workspace, model, pretrained):
    raw = dict(model=model, technical=True, device='cpu', precision='fp32', updates=2,
        patch=32, local_size=32, run_dir=str(workspace.root/'course'/model),
        work_root=str(workspace.root/'course'), pretrained=pretrained, parents={},
        official_root=str(ROOT/'vendor/MedSAM2'), dependency_root=str(ROOT/'env/medsam2'),
        accumulation=1 if model=='medsam2' else 4, batch=1, workers=0,
        execution=dict(workers=0, reserve_ram_bytes=2*2**30, cache_bytes=16*2**20))
    path = workspace.root/(model+'.json')
    write_json(path, raw)
    return path, options(path)


def run(args):
    item = next(row for row in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
                if row['dataset']=='imagechd')
    check = TestCase()
    check.assertEqual(item['case']['case_id'], 'ct_1003')
    check.assertEqual([b-a for a, b in item['roi_xyz']], [32]*3)
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        if args.collect_baseline:
            config, prepared, _ = source_config(workspace, item)
            row = prepared['manifest']['cases'][0]
            transform = CaseTransform(prepared, 2*2**30)
            q_entry = transform(row)
            raw, label, image, _, roi, _ = transform.read_case(row)
            med_path, med_config = model_config(workspace, 'medsam2', str(ROOT/'data/model_weights/MedSAM2_latest.pt'))
            from chd_segmentation.models.medsam2.adapter import activate
            activate(med_config)
            from chd_segmentation.models.medsam2.data import window_image
            stats = prepared['plan']['foreground_intensity_properties_per_channel']['0']
            window = med_config['gray_window'] or [stats['percentile_00_5'], stats['percentile_99_5']]
            pixels = window_image(raw.transpose(2, 1, 0), window)[None]
            baseline = dict(schema=1, state='collected_from_online_baseline',
                source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
                frozen_online_commit='69289532a3da2c94bc72769f5726ca260f6980a7', package_code_identity=code_identity(),
                case_id=row['case_id'], sha256=row['sha256'], roi_xyz=roi, plan=prepared['plan'],
                q=signature({key:q_entry[key] for key in Q_FIELDS}),
                medsam2=dict(gray_window=list(map(float, window)), image=signature(pixels),
                    label=signature(np.ascontiguousarray(label.transpose(2, 1, 0))),
                    source_shape_xyz=list(image.shape), source_affine=image.affine.tolist()),
                optimizer_updates=0, interpretation='32³ numeric reference; no model training or effect validation')
            write_json(workspace.report/'baseline.json', baseline)
            workspace.results.append(dict(case_id=row['case_id'], mode='online_numeric_baseline',
                q_shape=list(q_entry['image'].shape), medsam_shape=list(pixels.shape),
                baseline_file=str(workspace.report/'baseline.json'), optimizer_updates=0))
        else:
            baseline = read_json(args.baseline)
            config, prepared, source_path = source_config(workspace, item, baseline['plan'])
            check.assertEqual(prepared['manifest']['cases'][0]['sha256'], baseline['sha256'])
            from chd_segmentation.models.offline import preprocess_model
            from chd_segmentation.preprocessing.offline_store import OfflineTransform
            for model in ('q_route1', 'medsam2'):
                weight = item['pretrained'] if model=='q_route1' else str(ROOT/'data/model_weights/MedSAM2_latest.pt')
                model_path, _ = model_config(workspace, model, weight)
                output = workspace.root/'course'/(model+'-offline.json')
                prepared = preprocess_model(model_path, workspace.root/'source-config.json',
                    source_path, workspace.root/'data', output)
                row = prepared['manifest']['cases'][0]
                if model=='q_route1':
                    entry = OfflineTransform(prepared, True)(row)
                    check.assertEqual(signature({key:entry[key] for key in Q_FIELDS}), baseline['q'])
                else:
                    from chd_segmentation.models.medsam2.native import NativeCases
                    entry = NativeCases(prepared).get(row['case_id'])
                    check.assertEqual(signature(entry['image']), baseline['medsam2']['image'])
                    check.assertEqual(signature(entry['label']), baseline['medsam2']['label'])
                    check.assertEqual(entry['gray_window'], baseline['medsam2']['gray_window'])
                    check.assertEqual(entry['source_shape_xyz'], baseline['medsam2']['source_shape_xyz'])
                    check.assertEqual(entry['source_affine'], baseline['medsam2']['source_affine'])
                workspace.results.append(dict(model=model, case_id=row['case_id'],
                    offline_identity=prepared['identity'], baseline_exact=True, optimizer_updates=0))
        for kind in ('image', 'label'):
            check.assertEqual(file_digest(item['case'][kind]), item['case']['sha256'][kind])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--collect-baseline', action='store_true')
    parser.add_argument('--baseline')
    args = parser.parse_args()
    if not args.collect_baseline and not args.baseline:
        parser.error('离线验证需要先前实际收集的--baseline')
    run(args)
