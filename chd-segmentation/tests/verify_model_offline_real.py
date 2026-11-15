import argparse
from copy import deepcopy
import hashlib
import gc
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
from chd_segmentation.artifacts.store import digest, file_digest, read_json, write_json
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
            from chd_segmentation.models.offline import model_storage_lease, preprocess_model, validate_model_storage
            from chd_segmentation.preprocessing.offline import data_info, preprocess
            from chd_segmentation.preprocessing.offline_store import OfflineTransform, case_directory, storage_lease
            for model in ('q_route1', 'medsam2'):
                weight = item['pretrained'] if model=='q_route1' else str(ROOT/'data/model_weights/MedSAM2_latest.pt')
                model_path, model_options = model_config(workspace, model, weight)
                output = workspace.root/'course'/(model+'-offline.json')
                prepared = preprocess_model(model_path, workspace.root/'source-config.json',
                    source_path, workspace.root/'data', output)
                row = prepared['manifest']['cases'][0]
                entry, reader = None, None
                try:
                    if model=='q_route1':
                        entry = OfflineTransform(prepared, True)(row)
                        check.assertEqual(signature({key:entry[key] for key in Q_FIELDS}), baseline['q'])
                    else:
                        from chd_segmentation.models.medsam2.native import NativeCases
                        reader = NativeCases(prepared)
                        entry = reader.get(row['case_id'])
                        check.assertEqual(signature(entry['image']), baseline['medsam2']['image'])
                        check.assertEqual(signature(entry['label']), baseline['medsam2']['label'])
                        check.assertEqual(entry['gray_window'], baseline['medsam2']['gray_window'])
                        check.assertEqual(entry['source_shape_xyz'], baseline['medsam2']['source_shape_xyz'])
                        check.assertEqual(entry['source_affine'], baseline['medsam2']['source_affine'])
                finally:
                    if reader is not None:
                        reader.clear()
                    if entry is not None:
                        entry.clear()
                    gc.collect()
                pointer = prepared['storage']['cases'][row['case_id']]
                case_root = case_directory(prepared['storage']['root'], pointer['key'])
                before = {p.name: p.stat().st_mtime_ns for p in case_root.iterdir()}
                changed = read_json(model_path)
                changed.update(run_dir=str(workspace.root/'course'/(model+'-another-run')), updates=1, threads=1)
                changed['logging'] = dict(heartbeat_seconds=0, case_details=False)
                write_json(workspace.root/(model+'-reuse.json'), changed)
                reused = preprocess_model(workspace.root/(model+'-reuse.json'), workspace.root/'source-config.json',
                    source_path, workspace.root/'data', workspace.root/'course'/(model+'-reused.json'))
                check.assertEqual(reused, prepared)
                check.assertEqual(before, {p.name:p.stat().st_mtime_ns for p in case_root.iterdir()})
                with model_storage_lease(model_options, prepared, 'consumer-one'):
                    if model=='q_route1':
                        common = deepcopy(config)
                        common['preprocessing'].update(root=str(workspace.root/'data'), statistics_source='Q-A-technical')
                        shared = preprocess(common, source_path, workspace.root/'course/stu-shared.json')
                        check.assertEqual(shared, prepared)
                        common.update(model='stunet', run_dir=str(workspace.root/'course/stu-consumer'))
                        with storage_lease(shared, common):
                            info = data_info(output)
                            check.assertEqual(len(info['active_references']), 2)
                            check.assertTrue(info['cleanup_blocked'])
                        changed_stats = deepcopy(common)
                        changed_stats['preprocessing']['statistics_source'] = 'STU-590'
                        with check.assertRaises(ValueError):
                            with storage_lease(shared, changed_stats):
                                check.fail('错误统计来源被接受')
                        check.assertEqual(before, {p.name:p.stat().st_mtime_ns for p in case_root.iterdir()})
                    else:
                        wrong_window = dict(model_options, gray_window=[0, 1])
                        with check.assertRaises(ValueError):
                            validate_model_storage(wrong_window, prepared)
                    (case_root/'locations.npz').unlink()
                    with check.assertRaises(ValueError):
                        validate_model_storage(model_options, prepared)
                    with check.assertRaises(RuntimeError):
                        preprocess_model(model_path, workspace.root/'source-config.json', source_path,
                            workspace.root/'data', output)
                repaired = preprocess_model(model_path, workspace.root/'source-config.json', source_path,
                    workspace.root/'data', output)
                check.assertEqual(repaired['identity'], prepared['identity'])
                check.assertEqual(len(data_info(output)['active_references']), 0)
                wrong = deepcopy(prepared)
                wrong['storage']['complete'] = False
                wrong.pop('identity')
                wrong['identity'] = digest(wrong)
                wrong_path = workspace.root/'course'/(model+'-incomplete.json')
                write_json(wrong_path, wrong)
                from chd_segmentation.models.workflow import train
                with check.assertRaisesRegex(ValueError, '需要先运行 preprocess'):
                    train(model_options, wrong_path, 'base' if model=='q_route1' else 'sequence', stop_after=1)
                reader = OfflineTransform(repaired, False)
                entry = reader(row)
                try:
                    check.assertIsNone(entry['label'])
                    check.assertEqual(entry['properties']['class_locations'], {})
                finally:
                    entry.clear()
                    del reader, entry
                    gc.collect()
                workspace.results.append(dict(model=model, case_id=row['case_id'],
                    offline_identity=prepared['identity'], baseline_exact=True, optimizer_updates=0,
                    reused_without_rewrite=True, active_repair_blocked=True, missing_case_repaired=True,
                    incomplete_training_rejected=True, unlabeled_array_read=True,
                    exact_STU_Q_shared=model=='q_route1', wrong_statistics_rejected=model=='q_route1',
                    wrong_window_rejected=model=='medsam2'))
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
