import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
from uuid import uuid4

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from chd_segmentation.artifacts.store import file_digest, read_json, write_json
from chd_segmentation.config import DEFAULTS


def run(args):
    sys.path.insert(0, args.simulator)
    from common import clean_owned, files_under, supervise
    root, report = Path(args.work_root).resolve(), Path(args.report).resolve()
    if root.exists():
        raise FileExistsError('脚本短测目录必须尚不存在')
    root.mkdir(parents=True)
    token = uuid4().hex
    write_json(root/'owner.json', dict(token=token))
    item = next(e for e in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
                if e['dataset']=='imagechd')
    assert item['case']['case_id']=='ct_1003'
    row = deepcopy(item['case'])
    row['roles'] = ['train', 'validation']
    course = root/'course/chd-segmentation'
    label = 'CHD1282-Q' if args.model == 'q' else 'ImageCHD-MedSAM2'
    inputs, configs, old = course/'inputs/q1282', course/'configs'/label/'20261115', course/'runs/existing-stu-task'
    write_json(old/'protected.json', dict(owner='independent-existing-task'))
    before = {p.name:file_digest(p) for p in old.iterdir()}
    write_json(root/'manifest.json', dict(dataset='imagechd', scope='technical',
        data_root=item['data_root'], cases=[row], plan_scope='technical'))
    write_json(root/'plan.json', read_json(args.baseline)['plan'])
    config = deepcopy(DEFAULTS)
    config.update(dataset='imagechd', scope='technical', manifest=str(root/'manifest.json'),
        plan=str(root/'plan.json'), pretrained=item['pretrained'], work_root=str(course), run_dir=str(old))
    config['planning']['mode'] = 'existing'
    config['technical'] = dict(roi_xyz=item['roi_xyz'], allow_role_overlap=True)
    config['resources'].update(device='cpu', workers=0, reserve_ram_bytes=2**30)
    source_config = configs/'prepare.yaml'
    source_config.parent.mkdir(parents=True)
    source_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    model = 'q_route1' if args.model=='q' else 'medsam2'
    model_config = dict(model=model, technical=True, device='cpu', updates=2, patch=32, local_size=32,
        run_dir=str(course/'runs'/label/'20261115-offline'), work_root=str(course), batch=1, workers=0,
        pretrained=item['pretrained'] if model=='q_route1' else str(ROOT/'data/model_weights/MedSAM2_latest.pt'),
        official_root=str(ROOT/'vendor/MedSAM2'), dependency_root=str(ROOT/'env/medsam2'))
    model_path = configs/'base.yaml' if args.model=='q' else configs/'model.yaml'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding='utf-8')
    script = ROOT/'scripts'/('CHD1282-Q-20261115_run.sh' if args.model=='q' else 'ImageCHD-CHD1282-MedSAM2-20261115_run.sh')
    previous = dict(os.environ)
    os.environ['PATH'] = str(Path(args.bash).parent)+os.pathsep+previous['PATH']
    os.environ.update(CHD_ROOT=root.as_posix(), CHD_PYTHON=Path(sys.executable).as_posix(),
        Q_PREPARE_CONFIG=source_config.as_posix(), Q_CONFIG_DIR=configs.as_posix(),
        MEDSAM2_CONFIG=model_path.as_posix(), MEDSAM2_PREPARE_CONFIG=source_config.as_posix())
    try:
        words = ['prepare'] if args.model=='q' else ['imagechd', 'prepare']
        result = supervise([args.bash, '--noprofile', '--norc', script.as_posix(), *words], ROOT, root, report, timeout=240)
        assert before=={p.name:file_digest(p) for p in old.iterdir()}, '来源准备改写了原任务日志目录'
        ready = read_json(course/'prepared'/label/'20261115/offline.json')
        assert ready['storage']['complete']
        files = files_under(root)
        write_json(report/'acceptance.json', dict(state='passed', case_id=row['case_id'],
            model=model, source_old_task_unchanged=True, separate_model_config_used=True,
            actual_script=str(script), optimizer_updates=0, seconds=result['seconds'],
            artifact_count=len(files), artifact_bytes=sum(p['bytes'] for p in files)))
    finally:
        os.environ.clear()
        os.environ.update(previous)
        clean_owned(root, root.parent, token, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('q','med'), required=True)
    parser.add_argument('--bash', default='C:/msys64/usr/bin/bash.exe')
    parser.add_argument('--simulator', default='D:/projects/test')
    for key in ('spec', 'baseline', 'work-root', 'report'):
        parser.add_argument('--'+key, required=True)
    run(parser.parse_args())
