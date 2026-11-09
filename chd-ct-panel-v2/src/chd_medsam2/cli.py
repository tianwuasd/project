import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from chd_stunet.config import ROOT, read_config, resolve_path, training_paths

from .contracts import read_json, settings


def selected(arguments):
    return any(value == '--model=medsam2' or value == '--method=medsam2'
               or (value in ('--model', '--method') and arguments[index+1:index+2] == ['medsam2'])
               for index, value in enumerate(arguments))


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('要求正整数')
    return parsed


def add_shared(parser):
    parser.add_argument('--model', choices=['medsam2'], default='medsam2')
    parser.add_argument('--output')
    parser.add_argument('--fold', type=int, choices=(0, 1, 2))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--share-desktop-pid', type=positive)


def parser_for(command):
    parser = argparse.ArgumentParser(prog=f'run.py {command} --model medsam2')
    if command == 'prepare':
        parser.add_argument('--method', choices=['medsam2'], default='medsam2')
        parser.add_argument('--model', choices=['medsam2'], default='medsam2')
        parser.add_argument('--source-prepared', help='已有审核的 STU-Net-S partial 预处理运行')
        parser.add_argument('--fold', nargs='+', choices=['0', '1', '2', 'all'], default=['all'])
        parser.add_argument('--output')
        parser.add_argument('--cases', nargs='+')
        parser.add_argument('--limit', type=positive)
        parser.add_argument('--workers', type=positive)
        parser.add_argument('--readers', type=positive, default=2)
        parser.add_argument('--resume', action='store_true')
    elif command == 'train':
        add_shared(parser)
        parser.add_argument('--prepared')
        parser.add_argument('--weights')
        parser.add_argument('--epochs', type=positive)
        parser.add_argument('--train-cases', nargs='+')
        parser.add_argument('--val-cases', nargs='+')
        parser.add_argument('--validation-frames', type=positive, help='仅短测限制验证切片范围')
        parser.add_argument('--gpus')
        parser.add_argument('--max-gpus', type=int, choices=(1, 2))
        parser.add_argument('--cpu-ranks', type=int, choices=(1, 2), help='本机短测验证双进程同步')
        parser.add_argument('--prefetch-depth', type=int, choices=(0, 1, 2),
                            help='每进程最多提前准备的批次数，默认 1，0 表示同步读取')
        parser.add_argument('--recovery-root', help='与结果目录独立的检查点存储根目录')
        parser.add_argument('--gpu-wait-seconds', type=int)
        parser.add_argument('--stop-after-steps', type=positive)
        for name in ('background', 'resume', 'check-only', 'stop', 'capacity-only'):
            parser.add_argument('--'+name, action='store_true')
    elif command == 'predict':
        add_shared(parser)
        parser.add_argument('--training-run', required=True)
        parser.add_argument('--cases', nargs='+')
        parser.add_argument('--image')
        parser.add_argument('--prompts')
        parser.add_argument('--prompt-source', choices=['manual_box', 'simulated_gt_single_box'], required=True)
        parser.add_argument('--checkpoint', choices=['best', 'latest'], default='best')
    else:
        raise ValueError('MedSAM2 支持 prepare、train 和 predict')
    return parser


def interpreter(local):
    configured = local.get('medsam2_python')
    candidate = configured or str(ROOT/'.venv_medsam2'/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))
    return os.path.abspath(os.path.expanduser(candidate))


def main(arguments):
    command = arguments[0]
    args = parser_for(command).parse_args(arguments[1:])
    if command == 'train' and args.stop:
        from chd_stunet.observability.runlog import write_json
        output = resolve_path(args.output)
        if not (output/'job.json').is_file():
            raise ValueError('指定目录不是 MedSAM2 训练任务')
        if read_json(output/'job.json')['contract']['config']['protocol'] != 'imagechd-medsam2-box-v1':
            raise ValueError('指定目录属于其他模型')
        write_json(output/'control/stop.json', {'requested': True})
        print('已请求 MedSAM2 在更新边界停止：'+str(output), flush=True)
        return 0
    _, local = read_config()
    selected_python = interpreter(local)
    if Path(selected_python).is_file() and os.path.normcase(selected_python) != os.path.normcase(sys.executable):
        if os.environ.get('CHD_MEDSAM2_REEXEC') == '1':
            raise RuntimeError('MedSAM2 解释器重复切换')
        return subprocess.call([selected_python, '-u', str(ROOT/'run.py'), *arguments],
                               env=dict(os.environ, CHD_MEDSAM2_REEXEC='1', PYTHONUTF8='1'))
    paths = training_paths(local)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if command == 'prepare':
        from .preprocessing import read_source, run_prepare
        source = resolve_path(args.source_prepared or paths.get('completed_preprocessing_default'))
        output = resolve_path(args.output) if args.output else resolve_path(
            local.get('preprocess_output_root', paths.get('preprocess_output_root', 'outputs/preprocessing')))/'medsam2'/stamp
        folds = (0,1,2) if 'all' in args.fold else tuple(sorted({int(fold) for fold in args.fold}))
        cases = args.cases or []
        if args.limit:
            rows, _, _, _ = read_source(source, folds)
            cases = (cases or sorted(row['case_id'] for row in rows))[:args.limit]
        result = run_prepare(source, output, folds=folds, cases=cases, resume=args.resume,
                             workers=args.workers, readers=args.readers)
    elif command == 'train':
        result, output = train(args, local, paths, stamp)
    else:
        result, output = predict(args, paths, stamp)
    print(json.dumps(dict(result, output=str(output)), ensure_ascii=False, indent=2), flush=True)
    return 1 if result.get('status') == 'needs_attention' else 0


def locate_prepared(args, local, paths, fold):
    configured = args.prepared or local.get('medsam2_prepared_root')
    if configured:
        return resolve_path(configured)
    root = resolve_path(local.get('preprocess_output_root',
                                  paths.get('preprocess_output_root', 'outputs/preprocessing')))/'medsam2'
    candidates = [path.parent for path in root.glob('*/summary.json')
                  if read_json(path)['status']=='completed' and
                  str(fold) in read_json(path.parent/'manifest.json')['folds'] and
                  (args.smoke or read_json(path.parent/'manifest.json')['full_cohort'])]
    if len(candidates) != 1:
        raise ValueError('请用 --prepared 指定 MedSAM2 缓存，当前候选：'+str(candidates))
    return candidates[0]


def train(args, local, paths, stamp):
    from chd_stunet.acceleration.prediction import desktop_identity
    from .contracts import file_sha
    from .data import open_prepared
    from .model import upstream_root, versions
    from .runtime import launch
    fold = args.fold if args.fold is not None else 0
    output = resolve_path(args.output) if args.output else resolve_path(
        paths.get('training_output_root', 'outputs/training'))/'medsam2'/f'fold_{fold}'/stamp
    train_cases, val_cases = args.train_cases or [], args.val_cases or []
    if args.gpu_wait_seconds is not None and args.gpu_wait_seconds < 0:
        raise ValueError('等待时限必须非负')
    if os.name=='nt' and args.max_gpus==2:
        raise ValueError('Windows 本机支持单 GPU；双 GPU 使用 Linux NCCL')
    if args.cpu_ranks and not ((args.smoke or args.resume) and (args.device=='cpu' or args.resume)):
        raise ValueError('--cpu-ranks 仅供 CPU 短测使用')
    desktop = desktop_identity(args.share_desktop_pid, args.smoke or args.capacity_only) if args.share_desktop_pid else None
    if args.resume:
        if not args.output or any((args.weights, args.prepared, args.epochs, train_cases, val_cases,
                                   args.validation_frames, args.smoke, args.check_only, args.capacity_only, args.recovery_root,
                                   args.fold is not None)):
            raise ValueError('MedSAM2 恢复只指定已有 --output，沿用冻结的科学配置')
        contract, weights = None, None
    else:
        if not args.smoke and args.validation_frames:
            raise ValueError('正式训练须按全卷验证')
        if args.capacity_only and (args.smoke or args.epochs or args.stop_after_steps or args.check_only):
            raise ValueError('容量检查使用正式输入配置，可独立选择真实病例')
        if args.device=='cpu' and not (args.smoke or args.capacity_only or args.check_only):
            raise ValueError('CPU 仅供短测和容量检查；正式训练使用 BF16 GPU')
        config = settings(args.smoke, args.epochs)
        if args.capacity_only:
            config['scope'] = 'capacity_probe'
        upstream_root(config)
        prepared = locate_prepared(args, local, paths, fold)
        root, manifest, rows, metadata, train_ids, val_ids, window = open_prepared(
            prepared, fold, smoke=args.smoke or args.capacity_only, train_cases=train_cases, val_cases=val_cases)
        runtime = resolve_path(paths.get('runtime_root', 'outputs/runtime'))
        weights = resolve_path(args.weights or local.get('medsam2_weights_path') or runtime/'weights/medsam2/MedSAM2_latest.pt')
        if file_sha(weights) != config['weights_sha256']:
            raise ValueError('MedSAM2 作者权重 SHA256 不匹配')
        if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (root, weights)):
            raise ValueError('MedSAM2 训练输出须和输入分离')
        contract = dict(model='medsam2', config=config, prepared=str(root), fold=fold,
                        data_identity=manifest['identity'], train_ids=train_ids, val_ids=val_ids,
                        case_identities={cid: record['identity'] for cid, record in metadata.items()},
                        window=window, versions=versions(), prompt_source='simulated_gt_single_box')
        if args.check_only:
            return dict(status='checked', contract=contract), output
    recovery = resolve_path(args.recovery_root or local.get('medsam2_recovery_root') or
                            resolve_path(paths.get('runtime_root', 'outputs/runtime'))/'recovery/medsam2')
    result = launch(output, contract, weights=weights, train_cases=train_cases, val_cases=val_cases,
                    validation_frames=args.validation_frames, requested=args.device, gpus=args.gpus,
                    max_gpus=args.max_gpus, cpu_ranks=args.cpu_ranks, prefetch_depth=args.prefetch_depth,
                    wait_seconds=args.gpu_wait_seconds, shared_desktop=desktop, background=args.background,
                    resume=args.resume, stop_after_steps=args.stop_after_steps, recovery_root=recovery)
    return result, output


def predict(args, paths, stamp):
    import torch
    from chd_stunet.acceleration.prediction import admit_prediction_device, desktop_identity
    from chd_stunet.observability.runlog import RunLog, write_json
    from .checkpoints import load
    from .inference import evaluate_case, predict_image
    from .model import build, versions
    from .preprocessing import case_metadata
    from .runtime import frozen_job
    from .contracts import check_signature

    training = resolve_path(args.training_run)
    contract = frozen_job(training)['contract']
    config = contract['config']
    if (config['scope']=='capacity_probe' or (config['scope']=='smoke' and not args.smoke)
            or versions()!=contract['versions']):
        raise ValueError('MedSAM2 预测范围或依赖与训练合同不匹配')
    if args.fold is not None and args.fold != contract['fold']:
        raise ValueError('MedSAM2 预测与来源训练折不一致')
    output = resolve_path(args.output) if args.output else resolve_path(paths.get('prediction_output_root',
                                                      'outputs/prediction'))/'medsam2'/stamp
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('MedSAM2 预测输出须为空目录')
    prepared = Path(contract['prepared'])
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (training, prepared)):
        raise ValueError('MedSAM2 预测输出须和输入分离')
    if args.prompt_source == 'manual_box' and (not args.image or not args.prompts or args.cases):
        raise ValueError('人工提示需要 --image 和 --prompts')
    if args.prompt_source != 'manual_box' and (args.image or args.prompts):
        raise ValueError('真值模拟提示只用于原折留出病例')
    shared = desktop_identity(args.share_desktop_pid, args.smoke) if args.share_desktop_pid else None
    device, admission = admit_prediction_device(args.device or 'cuda', shared)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(admission['usable_bytes']/admission['total_bytes'], device)
    predictor = build(config, training=False, device=device)
    predictor.load_state_dict(load(training, contract, args.checkpoint)['model'], strict=True)
    if args.prompt_source == 'manual_box':
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type=='cuda'):
            return predict_image(predictor, resolve_path(args.image), resolve_path(args.prompts),
                                 contract['window'], output), output
    manifest = read_json(prepared/'manifest.json')
    prepared_summary = read_json(prepared/'summary.json')
    check_signature(manifest)
    if manifest['identity'] != contract['data_identity']:
        raise ValueError('MedSAM2 测试清单与训练合同不同')
    candidates = {row['case_id']: row for row in manifest['cases'] if row['roles'][str(contract['fold'])]=='test'}
    wanted = args.cases or sorted(candidates)
    if not wanted or len(wanted)!=len(set(wanted)) or set(wanted)-candidates.keys():
        raise ValueError('MedSAM2 测试病例必须属于来源训练折留出集')
    metadata = {cid: case_metadata(prepared/'cases'/cid, candidates[cid], track_files=True) for cid in wanted}
    if any(prepared_summary['completed'].get(cid) != marker['identity'] for cid, marker in metadata.items()):
        raise ValueError('MedSAM2 留出病例未在已提交清单中')
    log = RunLog(output)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type=='cuda'):
        rows = [evaluate_case(predictor, prepared, cid, metadata[cid], contract['window'],
                              output/'cases'/cid, log) for cid in wanted]
    summary = dict(status='completed', scope='whole_volume', fold=contract['fold'],
                   prompt_source=args.prompt_source, results=rows,
                   mean_dice=sum(row['mean_dice'] for row in rows)/len(rows))
    write_json(output/'summary.json', summary)
    return summary, output


def worker(output, device):
    import torch
    from chd_stunet.observability.runlog import write_json
    from chd_stunet.training.resources import GPUUnavailable
    from .checkpoints import CheckpointMirrorError
    from .engine import run_worker
    from .io_recovery import InputUnavailable, input_path
    from .runtime import frozen_job
    output = Path(output).resolve()
    job = frozen_job(output)
    rank = os.environ.get('RANK', '0')
    try:
        run_worker(output, job, device)
    except CheckpointMirrorError as error:
        write_json(Path(job['recovery'])/'storage-error.json', {'reason': 'checkpoint_mirror', 'error': str(error)})
        raise
    except (torch.cuda.OutOfMemoryError, GPUUnavailable) as error:
        reason = 'cuda_oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'admission_changed'
        write_json(output/'control'/f'capacity-failure-{rank}.json', {'reason': reason, 'error': str(error)})
        raise
    except Exception as error:
        pending = input_path(error, job['contract']['prepared']) if isinstance(error, OSError) else None
        if isinstance(error, InputUnavailable) or pending is not None:
            paths = error.paths if isinstance(error, InputUnavailable) else [pending]
            details = error.details if isinstance(error, InputUnavailable) else {'error': str(error)}
            write_json(Path(job['recovery'])/f'input-error-{rank}.json', dict(paths=paths, details=details))
            raise
        write_json(output/'control'/f'fatal-{rank}.json',
                   {'rank': int(rank), 'type': type(error).__name__, 'error': str(error)})
        raise
