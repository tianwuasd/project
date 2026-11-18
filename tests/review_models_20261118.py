import argparse
from copy import deepcopy
from functools import partial
import gc
import json
import os
from pathlib import Path
import subprocess
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
    manifest.update(dataset='imagechd', scope='technical', data_root=settings['data_root'], binding_required=False, cases=selected,
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
        input_source=dict(intensity_mode='auto'))
    write_json(course/'med.json',med)
    write_json(reports/'prepare.json',dict(state='completed',roles=roles['base'],cases=list(ids),
        source=str(course/'shared-offline.json'),seconds=perf_counter()-began,
        arrays={key:val for key,val in offline['storage'].items() if key!='cases'}))


def inspect_inputs(args):
    engine, root, reports = setup(args)
    from chd_segmentation.artifacts.store import write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import load_prepared
    from chd_segmentation.preprocessing.offline import preprocess
    from chd_segmentation.models.workflow import options, adapter
    from chd_segmentation.models.offline import validate_model_storage
    from chd_segmentation.models.cases import ModelCases
    from chd_segmentation.models.medsam2.native import produce_native_case, native_recipe, resolve_window
    from chd_segmentation.models.medsam2.official import effective_config
    from chd_segmentation.models.stunet.adapter import StunetAdapter
    import torch
    torch.set_num_threads(1)
    course = root/'course'
    source = load_prepared(course/'source.json')
    shared = load_prepared(course/'shared-offline.json')
    stu = load_config(course/'stu.json')
    spec = StunetAdapter.spec(stu, source['plan'])
    model = StunetAdapter.build(spec)
    weight = StunetAdapter.load_pretrained(model, stu['pretrained'])
    del model
    gc.collect()
    checks = dict(stu_initialization=weight)
    for stage in ('base','localizer'):
        q = options(course/('q-'+stage+'.json'))
        bundle = adapter(q, shared, stage)
        checks['q_'+stage] = dict(parameters=sum(p.numel() for p in bundle.model.parameters()))
        del bundle
        gc.collect()
    med = options(course/'med.json')
    native_config = deepcopy(stu)
    native_config['preprocessing'].update(root=str(root/'data/native'), statistics_source='MedSAM2-training')
    window = resolve_window(med, source)
    native = load_prepared(course/'med-native.json') if (course/'med-native.json').exists() else preprocess(
        native_config, course/'source.json', course/'med-native.json', method_label='MedSAM2',
        producer=partial(produce_native_case,window=window),
        recipe_builder=partial(native_recipe,window=window,derived_window=True))
    checks['med_sources'] = {}
    for label, prepared in (('shared_stu',shared),('native',native)):
        validate_model_storage(med,prepared)
        cases = ModelCases(prepared,med,with_label=True)
        entry = cases.get('ct_1001')
        checks['med_sources'][label] = dict(shape=list(entry['image'].shape),
            dtype=str(entry['image'].dtype),intensity=entry['intensity_conversion'])
        cases.clear()
    recipe = effective_config(med, shared, course/'med-offline.json',course/'runs/Med/sequence',['ct_1001'])
    from hydra.utils import instantiate
    leaf = recipe['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]
    dataset = instantiate(leaf)
    sample = dataset[0]
    checks['med_official_sample'] = dict(type=type(sample).__name__,frames=len(sample.frames))
    bundle = adapter(med,shared,'sequence')
    checks['med_initialization'] = dict(parameters=sum(p.numel() for p in bundle.model.parameters()))
    write_json(reports/'inspect.json',dict(state='completed',checks=checks,optimizer_updates=0))


def command(args):
    engine, root, reports = setup(args)
    os.environ['CHD_MINI_TOOL_ROOT'] = str(engine.parent/'chd-segmentation-assistant3')
    os.environ['CHD_MINI_ATTEMPT_LEDGER'] = str(reports.parent/'attempts.json')
    os.environ['CHD_MINI_STAGE'] = args.stage
    from chd_segmentation.cli import main
    course = root/'course'
    stage = args.stage
    model_stage = stage.removeprefix('q_') if stage.startswith('q_') else 'sequence'
    config = course/('stu.json' if stage=='stu' else 'med.json' if stage=='med_sequence' else 'q-'+model_stage+'.json')
    prepared = course/('med-offline.json' if stage=='med_sequence' else 'shared-offline.json')
    if args.phase=='train':
        command_args = ['train','--config',str(config),'--prepared',str(prepared),'--stop-after',str(args.stop_after)]
        if stage!='stu':
            command_args = ['models']+command_args+['--stage',model_stage]
        if args.resume:
            command_args += ['--resume']
    elif args.phase=='predict':
        command_args = ['predict','--config',str(config),'--output',str(root/'result'/stage),'--checkpoint','best']
        if stage!='stu':
            command_args = ['models']+command_args+['--prepared',str(prepared)]
        if stage=='med_sequence':
            command_args += ['--prompt-mode','label-box']
        for name in (('ct_1001','ct_1012') if args.resume else ('ct_1001',)):
            command_args += ['--case',name]
    else:
        command_args = ['evaluate','--config',str(course/'stu.json'),'--prepared',str(prepared),
            '--predictions',str(root/'result'/stage),'--output',str(root/'result'/(stage+'-evaluation.json'))]
    main(command_args)


def supervised(args):
    engine, root, reports = setup(args)
    tool = engine.parent/'chd-segmentation-assistant3'
    sys.path.insert(0,str(tool/'src'))
    from chd_mini.windows_job import supervise
    from chd_segmentation.artifacts.store import read_json,write_json
    tag = args.tag or '-'.join((args.stage or 'inputs',args.phase,'resume' if args.resume else 'first',str(args.stop_after)))
    receipt = reports/(tag+'-job.json')
    command_line = [sys.executable,'-B',str(Path(__file__).resolve()),'--settings',args.settings,
        '--work',str(root),'--reports',str(reports),'--phase',args.phase,'--stop-after',str(args.stop_after)]
    if args.stage:
        command_line += ['--stage',args.stage]
    if args.resume:
        command_line += ['--resume']
    source = subprocess.run(['git','-C',str(engine),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    environment = dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    try:
        supervise(command_line,engine,environment,reports/(tag+'.log'),timeout=1800,
            log_limit=1024*1024,ram_limit=16*2**30,receipt_path=receipt)
    finally:
        files = [path for path in root.rglob('*') if path.is_file()]
        inventory = dict(files=len(files),bytes=sum(path.stat().st_size for path in files))
        result = read_json(receipt)
        result.update(source_commit=source,tool_commit=subprocess.run(['git','-C',str(tool),'rev-parse','HEAD'],
            check=True,capture_output=True,text=True).stdout.strip(),ram_limit_bytes=16*2**30,
            cases=dict(train=['ct_1001'],validation=['ct_1012'],test=[]),work_inventory=inventory)
        write_json(receipt,result)
        if inventory['files']>200 or inventory['bytes']>8*2**30:
            raise RuntimeError('本阶段工作产物超过批准额度')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',required=True)
    parser.add_argument('--work',required=True)
    parser.add_argument('--reports',required=True)
    parser.add_argument('--phase',choices=('prepare','inspect','train','predict','evaluate'),required=True)
    parser.add_argument('--stage',choices=('stu','q_base','q_localizer','q_local','med_sequence'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int,choices=(1,2),default=1)
    parser.add_argument('--supervise',action='store_true')
    parser.add_argument('--tag')
    arguments = parser.parse_args()
    if arguments.supervise:
        supervised(arguments)
    elif arguments.phase=='prepare':
        prepare(arguments)
    elif arguments.phase=='inspect':
        inspect_inputs(arguments)
    else:
        command(arguments)
