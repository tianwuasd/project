import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter
from unittest import TestCase

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import read_json, write_json, file_digest
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.datasets.splits import apply_split
from chd_segmentation.models.workflow import options, training_names, train
from chd_segmentation.preprocessing.prepare import prepare


def run(args):
    check = TestCase()
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        started = perf_counter()
        if args.mode == 'q-groups':
            packet = Path(args.bundle)
            groups = read_json(packet/'q_groups.json')
            config = deepcopy(DEFAULTS)
            config['split']['path'] = str(packet/'q1282_split.json')
            manifest = apply_split(read_json(packet/'chd1282_manifest.json'), config)
            prepared = dict(scope='formal', manifest=manifest, plan=dict(fit_ids=groups['A']))
            model = dict(model='q_route1', technical=False, groups=str(packet/'q_groups.json'))
            assert training_names(model, prepared, 'base') == groups['A']
            wrong_path = workspace.root/'wrong_groups.json'
            for key in ('B', 'C'):
                changed = deepcopy(groups)
                changed[key][0], changed['D'][0] = changed['D'][0], changed[key][0]
                write_json(wrong_path, changed)
                with check.assertRaisesRegex(ValueError, '冻结'):
                    training_names(dict(model, groups=str(wrong_path)), prepared, 'local')
            for variant in ('duplicate', 'omission'):
                changed = deepcopy(groups)
                if variant == 'duplicate':
                    changed['B'].append(changed['B'][0])
                else:
                    changed['B'].pop()
                write_json(wrong_path, changed)
                with check.assertRaises(ValueError):
                    training_names(dict(model, groups=str(wrong_path)), prepared, 'local')
            workspace.results.append(dict(check='frozen_groups', records=len(manifest['cases']),
                rejected=['B_D_swap', 'C_D_swap', 'duplicate', 'omission'], CT_voxels_read=0,
                seconds=perf_counter()-started))
        else:
            experiment = next(item for item in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
                              if item['dataset'] == 'imagechd')
            row = deepcopy(experiment['case'])
            row['roles'] = ['train', 'validation']
            write_json(workspace.root/'manifest.json', dict(dataset='imagechd', scope='technical',
                data_root=experiment['data_root'], cases=[row], plan_scope='technical'))
            config = deepcopy(DEFAULTS)
            config.update(dataset='imagechd', scope='technical', manifest=str(workspace.root/'manifest.json'),
                pretrained=experiment['pretrained'], work_root=str(workspace.root/'course'), run_dir=str(workspace.root/'course/stu'))
            config['training'].update(patch=[32]*3, attempts=1, batch=1)
            config['technical'] = dict(roi_xyz=experiment['roi_xyz'], allow_role_overlap=True)
            write_json(workspace.root/'prepare.json', config)
            prepared_path = workspace.root/'course/prepared.json'
            prepare(load_config(workspace.root/'prepare.json'), prepared_path)
            record = dict(case_id=row['case_id'], roi_xyz=experiment['roi_xyz'], optimizer_updates=0,
                          device='cpu', rejected_starts={}, source_identity=row['sha256'])
            workspace.results.append(record)
            for model, stage in (('q_route1', 'base'), ('medsam2', 'sequence')):
                spec = dict(model=model, run_dir=str(workspace.root/'course'/model),
                    work_root=str(workspace.root/'course'), device='cpu', technical=True, updates=1,
                    patch=32, local_size=32, accumulation=1, batch=1, workers=0,
                    official_root=str(ROOT/'vendor/MedSAM2'), dependency_root=str(ROOT/'env/medsam2'),
                    pretrained=str(ROOT/'data/model_weights/MedSAM2_latest.pt') if model == 'medsam2' else experiment['pretrained'],
                    execution=dict(cpu_processes=1, workers=0, cpu_fraction=.0001, wait_seconds=0))
                path = workspace.root/(model+'.json')
                write_json(path, spec)
                for attempt in range(2):
                    with check.assertRaises(MemoryError):
                        train(options(path), prepared_path, stage)
                    check.assertFalse((Path(spec['run_dir'])/stage).exists())
                record['rejected_starts'][model] = 2
            assert all(file_digest(row[kind]) == row['sha256'][kind] for kind in ('image', 'label'))
            record.update(source_unchanged=True, seconds=perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('q-groups','admission'),required=True)
    for name in ('bundle', 'spec', 'work-root', 'report'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
