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
    if args.prepared is not None:
        prepared = Path(args.prepared).resolve(strict=True)
    if stage in ('q_localizer','q_local') and args.prepared is None:
        raise ValueError('定位及局部阶段必须显式指定本次prepared来源')
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


def prepare_q_inputs(args):
    engine, root, reports = setup(args)
    import nibabel as nib
    import numpy as np
    import torch
    from scipy.ndimage import distance_transform_edt
    from chd_segmentation.artifacts.store import read_json,write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import prepare as prepare_source
    from chd_segmentation.preprocessing.offline import preprocess
    from chd_segmentation.data.dataset import CaseDataset
    from chd_segmentation.models.workflow import options,adapter
    from chd_segmentation.models.q_route1.adapter import allowed
    from chd_segmentation.models.q_route1.targets import build_heatmap_targets
    torch.set_num_threads(1)
    course = root/'course'
    manifest = read_json(course/'manifest.json')
    selections = {}
    for row in manifest['cases']:
        label = np.asanyarray(nib.load(row['label']).dataobj)
        ventricle, vessel = np.isin(label,[1,2]),np.isin(label,[6,7])
        if not ventricle.any() or not vessel.any():
            raise ValueError('真实标签缺少心室或大血管：'+row['case_id'])
        distances = distance_transform_edt(~vessel,sampling=row['spacing_xyz'])
        minimum = float(distances[ventricle].min())
        junction = ventricle & (distances<=minimum+max(row['spacing_xyz']))
        center = np.asarray(np.where(junction)).mean(1)
        extent = np.array([128,128,96])
        start = np.clip(np.rint(center).astype(int)-extent//2,0,np.array(label.shape)-extent)
        selections[row['case_id']] = dict(center_xyz=center.tolist(),minimum_vessel_distance_mm=minimum,
            roi_xyz=np.stack((start,start+extent),axis=1).tolist())
        del label,ventricle,vessel,distances,junction
    # 两例分别依据真实标签确定区域，公共技术配置按逐病例ROI读取。
    if any(value['roi_xyz']!=selections['ct_1001']['roi_xyz'] for value in selections.values()):
        # 统一区域覆盖两个真实连接部位，尺寸由两例选区的包围框确定。
        boxes = np.asarray([value['roi_xyz'] for value in selections.values()])
        roi = np.stack((boxes[:,:,0].min(0),boxes[:,:,1].max(0)),axis=1).tolist()
    else:
        roi = selections['ct_1001']['roi_xyz']
    raw = read_json(course/'stu.json')
    raw.update(plan=str(course/'q-wide-plan.json'),run_dir=str(course/'runs/Q-wide-preprocess'))
    raw['technical']['roi_xyz'] = roi
    raw['preprocessing']['root'] = str(root/'data/q-wide')
    write_json(course/'q-wide-prepare.json',raw)
    applied = load_config(course/'q-wide-prepare.json')
    prepare_source(applied,course/'q-wide-source.json')
    prepared = preprocess(applied,course/'q-wide-source.json',course/'q-wide-offline.json')
    for stage in ('localizer','local'):
        path = course/('q-'+stage+'.json')
        config = read_json(path)
        config['run_dir'] = str(course/'runs/Q-wide')
        if stage=='local':
            config['parents']['localizer'] = str(course/'runs/Q-wide/localizer/best.pt')
        write_json(path,config)
    dataset = CaseDataset(prepared,dict(cache_bytes=0),with_label=True)
    q = options(course/'q-localizer.json')
    bundle = adapter(q,prepared,'localizer',initialize=False)
    checks = {}
    for row in manifest['cases']:
        entry = dataset.get(row['case_id'])
        _,mask = build_heatmap_targets(allowed(entry['label'],entry['unknown_classes'],'imagechd'),
            entry['label']>=0,entry['processed_spacing'])
        with torch.no_grad():
            loss = float(bundle.loss(entry,np.random.default_rng(123),0))
        if not np.isfinite(loss) or not mask.any():
            raise ValueError('真实定位监督核验未通过')
        checks[row['case_id']] = dict(shape=list(entry['image'].shape),valid_supervision=int(mask.sum()),loss=loss)
    dataset.clear()
    write_json(reports/'q-inputs.json',dict(state='completed',selection='real_label_ventricle_vessel_proximity',
        selections=selections,applied_shared_roi_xyz=roi,checks=checks,optimizer_updates=0))


def inspect_restore(args):
    engine, root, reports = setup(args)
    from chd_segmentation.artifacts.store import read_json,write_json
    from chd_segmentation.models.workflow import load_state,adapter
    from chd_segmentation.training.checkpoint import verify_optimizer_device
    import torch
    torch.set_num_threads(1)
    ledger = reports.parent/'attempts.json'
    before = read_json(ledger)
    state = load_state(root/'course/runs/Q/base/latest.pt')
    if state['stage']!='base' or state['training_cases']!=['ct_1001'] or state['update']!=1:
        raise ValueError('Q基础状态与本次实际训练不符')
    bundle = adapter(state['config'],state['prepared'],'base',initialize=False)
    bundle.model.load_state_dict(state['model'],strict=True)
    optimizer = bundle.optimizer()
    optimizer.load_state_dict(state['optimizer'])
    verify_optimizer_device(optimizer)
    bundle.restore_extra(state['rank_states'][0]['extra'])
    if read_json(ledger)!=before:
        raise ValueError('只读恢复检查期间台账发生变化')
    write_json(reports/'q-base-restore.json',dict(state='completed',checkpoint_update=state['update'],
        stage=state['stage'],model_loaded_strict=True,optimizer_states=len(optimizer.state),
        sampler_extra_loaded=True,optimizer_step_calls=0,ledger_unchanged=True,
        resume_with_additional_update_verified=False))


def inspect_med_supervision(args):
    engine, root, reports = setup(args)
    from chd_segmentation.artifacts.store import write_json
    from chd_segmentation.models.workflow import options
    from chd_segmentation.preprocessing.prepare import load_prepared
    from chd_segmentation.models.medsam2.official import effective_config
    from chd_segmentation.acceleration.host import host_capacity
    from chd_segmentation.acceleration.memory import system_headroom
    import torch
    torch.set_num_threads(1)
    config = options(root/'course/med.json')
    source_path = Path(args.prepared).resolve(strict=True)
    prepared = load_prepared(source_path)
    recipe = effective_config(config,prepared,source_path,root/'course/runs/Med/sequence',['ct_1001'])
    from hydra.utils import instantiate
    checks = {}
    for name in ('ct_1001','ct_1012'):
        leaf = deepcopy(recipe['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0])
        leaf['video_dataset']['names'] = [name]
        dataset = instantiate(leaf)
        sample = dataset[0]
        frames = [dict(objects=len(frame.objects),positive_pixels=sum(int(obj.segment.sum()) for obj in frame.objects))
                  for frame in sample.frames]
        if len(frames)!=8 or not all(frame['objects']>0 for frame in frames) or not any(frame['positive_pixels']>0 for frame in frames):
            raise ValueError('官方序列没有实际有监督对象：'+name)
        checks[name] = frames
    facts = host_capacity()
    headroom,scopes = system_headroom(facts,.8,4*2**30)
    write_json(reports/'med-supervision.json',dict(state='completed',prepared_source=str(source_path),
        frames=checks,optimizer_updates=0,host=facts,headroom_bytes=headroom,memory_scopes=scopes,
        first_admission_requested_bytes=9660917382))


def inspect_q_local(args):
    engine, root, reports = setup(args)
    from chd_segmentation.artifacts.store import write_json
    from chd_segmentation.models.workflow import options,adapter
    from chd_segmentation.preprocessing.prepare import load_prepared
    from chd_segmentation.data.dataset import CaseDataset
    import numpy as np
    import torch
    torch.set_num_threads(1)
    source = Path(args.prepared).resolve(strict=True)
    prepared = load_prepared(source)
    config = options(root/'course/q-local.json')
    config = dict(config,run_dir=str(root/'checks/q-local-inputs'))
    bundle = adapter(config,prepared,'local')
    dataset = CaseDataset(prepared,dict(cache_bytes=0),with_label=True)
    bundle.prepare_training(dataset,['ct_1001'])
    rng = np.random.default_rng(123)
    entry = bundle.sample(dataset,['ct_1001'],rng)
    with torch.no_grad():
        loss = float(bundle.loss(entry,rng,0))
        validation = bundle.validate(dataset,['ct_1012'])
    if not np.isfinite(loss) or not np.isfinite(validation):
        raise ValueError('Q局部真实训练或验证损失不是有限数值')
    write_json(reports/'q-local-inputs.json',dict(state='completed',prepared_source=str(source),
        training_case='ct_1001',validation_case='ct_1012',loss=loss,validation_loss=validation,optimizer_updates=0))
    dataset.clear()


def inspect_preprocess_reuse(args):
    engine, root, reports = setup(args)
    from chd_segmentation.artifacts.store import read_json,write_json
    from chd_segmentation.models.offline import preprocess_model
    course = root/'course'
    checks = {}
    for name,model,prepared in (('q_stu','q-base.json','shared-offline.json'),
            ('med_stu','med.json','shared-offline.json'),('med_native','med.json','med-native.json')):
        source = read_json(course/prepared)
        result = preprocess_model(course/model,course/'stu.json',course/prepared,
            source['storage']['root'],course/('reuse-'+name+'.json'))
        if result['storage']!=source['storage']:
            raise ValueError('预处理复用入口改变了已有数组来源')
        checks[name] = dict(prepared_source=str(course/prepared),root=result['storage']['root'],
            reused_cases=list(result['storage']['cases']),new_arrays=0)
    write_json(reports/'preprocess-reuse.json',dict(state='completed',checks=checks,optimizer_updates=0))


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
    if args.prepared is not None:
        command_line += ['--prepared',str(Path(args.prepared).resolve(strict=True))]
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
        result['prepared_source'] = args.prepared
        write_json(receipt,result)
        if inventory['files']>200 or inventory['bytes']>8*2**30:
            raise RuntimeError('本阶段工作产物超过批准额度')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',required=True)
    parser.add_argument('--work',required=True)
    parser.add_argument('--reports',required=True)
    parser.add_argument('--phase',choices=('prepare','inspect','preprocess-reuse','q-inputs','q-local-inputs','med-supervision','restore','train','predict','evaluate'),required=True)
    parser.add_argument('--stage',choices=('stu','q_base','q_localizer','q_local','med_sequence'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int,choices=(1,2),default=1)
    parser.add_argument('--supervise',action='store_true')
    parser.add_argument('--tag')
    parser.add_argument('--prepared')
    arguments = parser.parse_args()
    if arguments.supervise:
        supervised(arguments)
    elif arguments.phase=='prepare':
        prepare(arguments)
    elif arguments.phase=='inspect':
        inspect_inputs(arguments)
    elif arguments.phase=='restore':
        inspect_restore(arguments)
    elif arguments.phase=='q-inputs':
        prepare_q_inputs(arguments)
    elif arguments.phase=='q-local-inputs':
        inspect_q_local(arguments)
    elif arguments.phase=='med-supervision':
        inspect_med_supervision(arguments)
    elif arguments.phase=='preprocess-reuse':
        inspect_preprocess_reuse(arguments)
    else:
        command(arguments)
