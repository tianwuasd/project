import argparse
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace
import sys

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import operation_lock, write_status
from ..prediction.heldout import GROUPS
from ..storage_paths import assert_output_location
from ..storage_migration import canonical_path, load_relocation, resolve_relocated_path
from . import PROTOCOL


def prediction_cases(args, cache):
    if args.development:
        if not args.technical_subset or args.config or args.policy:
            raise ValueError('开发集入口需要 --technical-subset，直接读取已提交的原网格')
        admitted = set(cache.case_ids('development'))
        names = args.case or sorted(admitted)
        if len(set(names)) != len(names) or not set(names) <= admitted:
            raise ValueError('请选择冻结开发集中互不重复的病例')
        cases = {}
        for name in names:
            relative = f'staged/{name}/image{cache.staged_extension}'
            path = cache._artifact(f'stage-{name}',relative)
            geo = cache.geometry(name)
            cases[name] = dict(role='development',image=str(path),shape=geo.native_shape_xyz,
                               spacing=geo.spacing_xyz,sha256=dict(image=sha256_file(path)))
        return cases
    if not args.config or not args.policy:
        raise ValueError('留出集预测需要原预处理的 --config 和 --policy')
    from ..prediction.heldout import plan_inputs
    return plan_inputs(cache,read_json(args.config),args.policy,args.groups,args.case)


def make_job(args, model, cache):
    dataset = cache.run_dir/'preprocessed/Dataset282_CHDSource'
    job = dict(protocol=PROTOCOL,model=model,command=args.command,
               prepared=str(cache.run_dir),prepared_digest=cache.request_digest,
               plans=sha256_file(dataset/'nnUNetPlans.json'),dataset=sha256_file(dataset/'dataset.json'),
               technical_subset=args.technical_subset)
    if args.command == 'train':
        from .stunet import topology, WEIGHT_SHA256
        from math import isfinite

        if (min(args.epochs,args.train_steps,args.development_steps,args.batch,*args.patch) < 1
                or not isfinite(args.learning_rate) or args.learning_rate <= 0 or args.seed < 0):
            raise ValueError('训练轮数、步数、批量、尺寸、随机种子或学习率无效')
        if sha256_file(args.weights) != WEIGHT_SHA256:
            raise ValueError('STU-Net 官方权重 SHA256 不匹配')
        if not cache.case_ids('train') or not cache.case_ids('development'):
            raise ValueError('需要冻结的训练集和开发集')
        spacing = read_json(dataset/'nnUNetPlans.json')['configurations']['3d_fullres']['spacing']
        precision = args.precision or ('fp32' if args.device=='cpu' else 'bf16')
        return dict(job,weights=str(args.weights.resolve()),weights_sha256=WEIGHT_SHA256,
                    settings=dict(patch=args.patch,strides=topology(spacing,args.patch),
                                  train_steps=args.train_steps,development_steps=args.development_steps,
                                  batch=args.batch,learning_rate=args.learning_rate,seed=args.seed,
                                  precision=precision))
    cases = prediction_cases(args,cache)
    job.update(cases=cases,development=args.development)
    if model == 'stunet':
        training = args.training.resolve()
        return dict(job,training=str(training),training_summary=sha256_file(training/'training.json'))
    from .sam_med3d import WEIGHT_SHA256, validate_point
    if sha256_file(args.weights) != WEIGHT_SHA256:
        raise ValueError('SAM-Med3D Turbo 权重 SHA256 不匹配')
    points = read_json(args.prompts)
    if not isinstance(points,dict) or not any(name in points for name in cases):
        raise ValueError('所选病例没有提示点')
    for name,row in cases.items():
        for structure,point in points.get(name,{}).items():
            if structure not in {str(i) for i in range(1,11)}:
                raise ValueError('提示点结构编号必须属于 Dataset1282 的 1 至 10 类')
            validate_point(point,row['shape'])
    return dict(job,weights=str(args.weights.resolve()),weights_sha256=WEIGHT_SHA256,
                prompts={name:points.get(name,{}) for name in cases})


def resource_request(job, cache):
    gib = 1024**3
    if job['command'] == 'train':
        settings = job['settings']
        return 3*gib, int((.8+prod(settings['patch'])*settings['batch']/240_000)*gib)
    if job['model'] == 'sam_med3d':
        voxels = max(prod(row['shape']) for row in job['cases'].values())
        return 3*gib + voxels*14, 3*gib
    training = read_json(Path(job['training'])/'run.json')
    patch = training['settings']['patch']
    if job['development']:
        voxels = max(prod(cache.geometry(name).cache_shape) for name in job['cases'])
    else:
        from ..prediction.heldout import prepare_memory
        return prepare_memory(job) + 2*gib, 3*gib
    return 2*gib+voxels*56, int((.8+prod(patch)/240_000)*gib)


def _training_relocation(output, job):
    if job['model'] != 'stunet' or job['command'] != 'train':
        return None
    relocation = load_relocation(output)
    if relocation is None:
        return None
    relative = (output/'run.json').relative_to(Path(relocation['destination'])).as_posix()
    return relocation if relative in relocation['contracts'] else None


def _resume_training_job(output, job):
    stored = read_json(output/'run.json')
    relocation = _training_relocation(output, stored)
    candidate = dict(job)
    if relocation is not None:
        for key in ('prepared','weights'):
            candidate[key] = canonical_path(candidate[key])
    if candidate != stored:
        raise ValueError('续跑的病例、配置或权重发生变化')
    return stored


def _training_execution_paths(output, job):
    relocation = _training_relocation(output,job)
    paths = {key:Path(job[key]) for key in ('prepared','weights')}
    if relocation is not None:
        source = Path(relocation['source'])
        for key,path in paths.items():
            if path == source or source in path.parents:
                paths[key] = resolve_relocated_path(path,Path(relocation['destination'])/'.storage-relocation.json',output/'run.json')
    if sha256_file(paths['weights']) != job['weights_sha256']:
        raise ValueError('迁移后的 STU-Net 预训练权重摘要不一致')
    return paths


def execute(args, model, output, owner, job, cache):
    import torch
    from ..xu_dispatch import Children, select_devices
    from ..xu_gpu_isolation import configure_coordinator
    from ..xu_pipeline import _execute
    from ..xu_reallocation import run_with_reallocation
    from ..xu_resources import ResourceUnavailable, combined_environment, stage_resources

    configure_coordinator(cpu_only=args.device=='cpu')
    host,cuda = resource_request(job,cache)
    args.resource_recovery = True
    args.resource_poll_seconds = args.wait_seconds
    args.prefetch_workers = args.prefetch_depth = 1

    def attempt(running):
        children = Children(output,owner)
        devices = select_devices(args.device,'all-visible' if args.device is None else None,torch)
        errors = []
        for candidate in devices:
            device = candidate['device']
            try:
                plan = stage_resources([device],threads=args.threads,policy='v2',
                    owned_pids=children.pids(),require_pid_evidence=device!='cpu',
                    host_demand_bytes=host,cuda_demand_bytes=0 if device=='cpu' else cuda)
            except ResourceUnavailable as error:
                errors.append(str(error))
                continue
            break
        else:
            raise ResourceUnavailable('; '.join(errors))
        settings = SimpleNamespace(threads=plan['threads'],children=children,device=device,
                                   overlay=combined_environment(plan,[device]),assigned_devices=[device])
        atomic_json(output/'resources'/'admission.json',plan)
        command = ['worker','--output',str(output),'--device',device]
        if args.command == 'train':
            command += ['--epochs',str(args.epochs)]
        try:
            _execute(model+'.py',command,model+'-'+args.command,output,owner,settings)
        finally:
            children.stop()
    run_with_reallocation(attempt,args,output)


def start(args, model):
    from ..xu_geometry import PreparedCache
    from ..xu_pipeline import ROOT

    output = assert_output_location(args.output)
    sources = [args.prepared.resolve(),ROOT]
    for name in ('training','weights','prompts','config','policy'):
        value = getattr(args,name,None)
        if value:
            sources.append(value.resolve())
    if getattr(args,'config',None):
        config = read_json(args.config)
        sources.extend(Path(config[key]).resolve() for key in ('data_root','metadata_excel'))
    if any(output == p or output in p.parents or p in output.parents for p in sources):
        raise ValueError('输出目录必须与工程、源数据、权重和其他运行目录分开')
    if args.threads < 1 or not 1 <= args.wait_seconds <= 300:
        raise ValueError('线程数或资源等待秒数无效')
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('新任务需要空输出目录')
    cache = PreparedCache(args.prepared,allow_technical_subset=args.technical_subset)
    with operation_lock(output,PROTOCOL,output) as owner:
        job = make_job(args,model,cache)
        path = output/'run.json'
        job = json.loads(json.dumps(job,default=str))
        if args.resume:
            job = _resume_training_job(output,job)
        else:
            atomic_json(path,job)
        try:
            execute(args,model,output,owner,job,cache)
        except (Exception,KeyboardInterrupt) as error:
            write_status(output,state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                         stage=args.command,error=str(error),active=[])
            raise
        write_status(output,state='complete',stage=args.command,active=[])
        print(f'{model} {args.command}: {output}',flush=True)


def worker(args,model):
    import torch
    from ..xu_geometry import PreparedCache
    from ..xu_gpu_isolation import validate_entry
    from ..xu_resources import apply_env_limit
    import os

    validate_entry(sys.argv[1:],os.environ)
    resources = apply_env_limit(args.device)
    print('Applied limits:',resources,flush=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    output = assert_output_location(args.output)
    job = read_json(output/'run.json')
    if job['protocol'] != PROTOCOL or job['model'] != model:
        raise ValueError('模型运行身份不一致')
    execution_paths = _training_execution_paths(output,job) if job['command']=='train' and model=='stunet' else None
    cache = PreparedCache(execution_paths['prepared'] if execution_paths else job['prepared'],allow_technical_subset=job['technical_subset'])
    if cache.request_digest != job['prepared_digest']:
        raise ValueError('预处理身份发生变化')
    device = torch.device(args.device)
    if job['command'] == 'train':
        from .stunet import train
        train(cache,job,output,args.epochs,device,weights_path=execution_paths['weights'])
    else:
        from .prediction import predict
        predict(cache,job,output,device)


def main(model):
    parser = argparse.ArgumentParser(description=f'Dataset1282 {model} independent model')
    commands = parser.add_subparsers(dest='command',required=True)
    names = ('train','predict') if model=='stunet' else ('predict',)
    for name in names:
        p = commands.add_parser(name)
        p.add_argument('--prepared',required=True,type=Path)
        p.add_argument('--output',required=True,type=Path)
        p.add_argument('--device',help='cpu、cuda:序号或省略以使用获准的空闲卡')
        p.add_argument('--threads',type=int,default=4)
        p.add_argument('--wait-seconds',type=int,default=30)
        p.add_argument('--technical-subset',action='store_true')
        p.add_argument('--resume',action='store_true')
        if name=='train':
            p.add_argument('--weights',required=True,type=Path)
            p.add_argument('--epochs',type=int,default=250)
            p.add_argument('--patch',nargs=3,type=int,default=[64,96,96])
            p.add_argument('--batch',type=int,default=1)
            p.add_argument('--train-steps',type=int,default=250)
            p.add_argument('--development-steps',type=int,default=50)
            p.add_argument('--seed',type=int,default=2023)
            p.add_argument('--learning-rate',type=float,default=.001)
            p.add_argument('--precision',choices=('fp32','bf16'))
        else:
            p.add_argument('--case',action='append')
            p.add_argument('--config',type=Path)
            p.add_argument('--policy',type=Path)
            p.add_argument('--groups',nargs='+',choices=GROUPS,default=list(GROUPS))
            p.add_argument('--development',action='store_true')
            if model=='stunet':
                p.add_argument('--training',required=True,type=Path)
            else:
                p.add_argument('--weights',required=True,type=Path)
                p.add_argument('--prompts',required=True,type=Path)
    for name in ('worker','status','evaluate'):
        p = commands.add_parser(name)
        p.add_argument('--output',required=True,type=Path)
        if name=='worker':
            p.add_argument('--device',required=True)
            p.add_argument('--epochs',type=int)
    args = parser.parse_args()
    if args.command=='worker':
        worker(args,model)
    elif args.command=='status':
        print((args.output/'status.json').read_text(encoding='utf-8'))
    elif args.command=='evaluate':
        from .prediction import evaluate
        evaluate(assert_output_location(args.output),model)
    else:
        start(args,model)
