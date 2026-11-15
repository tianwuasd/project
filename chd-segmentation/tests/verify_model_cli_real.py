import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from chd_segmentation.artifacts.store import file_digest, read_json, write_json
from chd_segmentation.config import DEFAULTS


def run(args):
    sys.path.insert(0, str(Path(args.simulator).resolve(strict=True)))
    from common import clean_owned, files_under, supervise
    root, report = Path(args.work_root).resolve(), Path(args.report).resolve()
    if root.exists():
        raise FileExistsError('CLI检查目录必须尚不存在')
    root.mkdir(parents=True)
    token = uuid4().hex
    write_json(root/'owner.json', dict(token=token))
    baseline = read_json(args.baseline)
    item = next(row for row in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
                if row['dataset']=='imagechd')
    row = deepcopy(item['case'])
    row['roles'] = ['train', 'validation']
    write_json(root/'manifest.json', dict(dataset='imagechd', scope='technical',
        data_root=item['data_root'], cases=[row], plan_scope='technical'))
    write_json(root/'plan.json', baseline['plan'])
    config = deepcopy(DEFAULTS)
    config.update(dataset='imagechd', scope='technical', manifest=str(root/'manifest.json'),
        plan=str(root/'plan.json'), pretrained=item['pretrained'], work_root=str(root/'course'),
        run_dir=str(root/'course/source'))
    config['planning']['mode'] = 'existing'
    config['technical'] = dict(roi_xyz=item['roi_xyz'], allow_role_overlap=True)
    config['resources'].update(device='cpu', workers=0, reserve_ram_bytes=2*2**30)
    write_json(root/'config.json', config)
    stages, receipts = [], []
    def command(name, words):
        record = supervise([sys.executable, '-B', str(ROOT/'run.py'), *words], ROOT, root, report/name, timeout=180)
        stages.append(dict(name=name, seconds=record['seconds']))
    try:
        source = root/'course/source.json'
        command('prepare', ['prepare', '--config', str(root/'config.json'), '--output', str(source)])
        for model in ('q_route1', 'medsam2'):
            model_config = dict(model=model, technical=True, device='cpu', updates=2, patch=32, local_size=32,
                run_dir=str(root/'course'/model), work_root=str(root/'course'), batch=1, workers=0,
                pretrained=item['pretrained'] if model=='q_route1' else str(ROOT/'data/model_weights/MedSAM2_latest.pt'),
                official_root=str(ROOT/'vendor/MedSAM2'), dependency_root=str(ROOT/'env/medsam2'))
            write_json(root/(model+'.json'), model_config)
            output = root/'course'/(model+'-offline.json')
            words = ['models', 'preprocess', '--config', str(root/(model+'.json')), '--prepare-config',
                str(root/'config.json'), '--prepared', str(source), '--root', str(root/'data'), '--output', str(output)]
            command(model, words)
            prepared = read_json(output)
            case = prepared['storage']['cases'][row['case_id']]
            directory = Path(prepared['storage']['root'])/case['key'][:16]
            before = {p.name:p.stat().st_mtime_ns for p in directory.iterdir()}
            command(model+'-reuse', words)
            assert before=={p.name:p.stat().st_mtime_ns for p in directory.iterdir()}
            expected = baseline['q'] if model=='q_route1' else baseline['medsam2']
            for name in ('image', 'label'):
                array = np.load(directory/(name+'.npy'), allow_pickle=False)
                assert str(array.dtype)==expected[name]['dtype']
                assert list(array.shape)==expected[name]['shape']
                assert hashlib.sha256(array.tobytes()).hexdigest()==expected[name]['sha256']
                del array
            command(model+'-info', ['data-info', '--prepared', str(output)])
            receipts.append(dict(model=model, exact=True, reused_without_rewrite=True, optimizer_updates=0))
        files = files_under(root)
        write_json(report/'acceptance.json', dict(state='passed', cases=[row['case_id']], models=receipts,
            stages=stages, files=len(files), bytes=sum(p['bytes'] for p in files),
            source_unchanged=all(file_digest(row[k])==row['sha256'][k] for k in ('image', 'label'))))
    finally:
        clean_owned(root, root.parent, token, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='D:/projects/test')
    for key in ('spec', 'baseline', 'work-root', 'report'):
        parser.add_argument('--'+key, required=True)
    run(parser.parse_args())
