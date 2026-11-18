import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import yaml


def setup(args):
    root = Path(args.work).resolve()
    reports = Path(args.reports).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    runtime = root/'runtime'
    runtime.mkdir(exist_ok=True)
    for key, name in dict(MPLCONFIGDIR='matplotlib', TORCH_HOME='torch-home', HF_HOME='huggingface',
                          XDG_CACHE_HOME='cache', TORCHINDUCTOR_CACHE_DIR='inductor').items():
        os.environ[key] = str(runtime/name)
    for key in ('TMPDIR','TEMP','TMP'):
        os.environ[key] = str(runtime)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', USE_LIBUV='0', OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    engine = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(engine/'src'))
    return engine, root, reports


def prepare(args):
    engine, root, reports = setup(args)
    from chd_segmentation.artifacts.store import read_json, write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import prepare as prepare_source
    from chd_segmentation.preprocessing.offline import preprocess
    settings = read_json(args.settings)
    manifest = read_json(settings['manifest'])
    ids = ('ct_1001','ct_1012')
    selected = []
    for name, role in zip(ids, ('train','validation')):
        row = deepcopy(next(row for row in manifest['cases'] if row['case_id'] == name))
        row.update(image=str(Path(settings['data_root'])/(name+'_image.nii.gz')),
                   label=str(Path(settings['data_root'])/(name+'_label.nii.gz')), roles=[role])
        selected.append(row)
    manifest.update(dataset='imagechd', scope='technical', data_root=settings['data_root'], cases=selected,
                    plan_scope='technical')
    course = root/'course'
    write_json(course/'manifest.json', manifest)
    config = dict(model='stunet', dataset='imagechd', scope='technical', fold=0,
        manifest=str(course/'manifest.json'), pretrained=settings['pretrained'], work_root=str(course),
        run_dir=str(course/'runs/STU'), result_root=str(root/'result'),
        preprocessing=dict(root=str(root/'data/shared'),statistics_source='technical-train'),
        technical=dict(roi_xyz=[[224,288],[224,288],[71,135]],allow_role_overlap=False),
        planning=dict(mode='auto',samples_per_case=1000,max_samples=1000,target_spacing_xyz=[.6,.6,1.2]),
        training=dict(patch=[32]*3,batch=1,attempts=2,epochs=2,steps_per_epoch=1,
                      validation_steps=1,save_every_attempts=1),
        resources=dict(device='cpu',cpu_threads=1,cpu_processes=1,workers=0,prefetch=1,
            cache_bytes=0,max_workers=1,max_readers=1,memory_limit_bytes=16*2**30,ram_fraction=.8),
        logging=dict(heartbeat_seconds=15,case_details=True))
    write_json(course/'stu.json',config)
    applied = load_config(course/'stu.json')
    began = perf_counter()
    source = prepare_source(applied,course/'source.json')
    offline = preprocess(applied,course/'source.json',course/'shared-offline.json')
    write_json(course/'med-offline.json',offline)
    roles = {stage:dict(train=['ct_1001'],validation=['ct_1012'],test=[])
             for stage in ('base','localizer','local','sequence')}
    common = dict(technical=True,technical_roles=roles,device='cpu',precision='fp32',threads=1,
        updates=2,validate_every=1,patch=32,local_size=32,work_root=str(course),
        execution=dict(device='cpu',cpu_processes=1,workers=0,prefetch=1,cache_bytes=0,
            max_workers=1,max_readers=1,memory_limit_bytes=16*2**30,ram_fraction=.8),
        logging=dict(heartbeat_seconds=15,case_details=True))
    for stage in ('base','localizer','local'):
        model = dict(common,model='q_route1',run_dir=str(course/'runs/Q'),accumulation=4,
            pretrained=settings['pretrained'] if stage=='base' else None)
        if stage == 'local':
            model['parents'] = dict(base=str(course/'runs/Q/base/best.pt'),
                                    localizer=str(course/'runs/Q/localizer/best.pt'))
        write_json(course/('q-'+stage+'.json'),model)
    med = dict(common,model='medsam2',run_dir=str(course/'runs/Med'),accumulation=1,batch=1,workers=0,
        frames=8,resolution=512,epochs=75,pretrained=str(engine/'data/model_weights/MedSAM2_latest.pt'),
        official_root=str(engine/'vendor/MedSAM2'),dependency_root=str(engine/'env/medsam2'),
        input_source=dict(intensity_mode='source_window'))
    write_json(course/'med.json',med)
    write_json(reports/'prepare.json',dict(state='completed',roles=roles['base'],cases=list(ids),
        source=str(course/'shared-offline.json'),seconds=perf_counter()-began,
        arrays={key:val for key,val in offline['storage'].items() if key!='cases'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',required=True)
    parser.add_argument('--work',required=True)
    parser.add_argument('--reports',required=True)
    parser.add_argument('--phase',choices=('prepare',),required=True)
    arguments = parser.parse_args()
    prepare(arguments)
