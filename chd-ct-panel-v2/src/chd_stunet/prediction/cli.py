from datetime import datetime
from pathlib import Path

from ..config import default_prepared, prediction_paths, resolve_path


def add_arguments(sub, positive, nonnegative):
    import argparse
    parser = sub.add_parser('predict', help='whole-volume heldout prediction with frozen STU-Net-S models')
    parser.add_argument('--model', choices=('stunet-s', 'medsam2'), default='stunet-s')
    parser.add_argument('--training-run', nargs='+', help='one existing training run per selected fold')
    parser.add_argument('--prepared')
    parser.add_argument('--fold', nargs='+', choices=['0', '1', '2', 'all'])
    parser.add_argument('--checkpoint', choices=['best', 'epoch', 'latest'])
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--images', nargs='+', help='explicit new NIfTI CT paths with verified mm geometry')
    parser.add_argument('--output')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--gpus', help='auto or CUDA-visible ordinals, for example 0,1')
    parser.add_argument('--max-gpus', type=positive)
    parser.add_argument('--gpu-memory-gib', type=float)
    parser.add_argument('--max-batch', type=positive)
    parser.add_argument('--threads', type=positive)
    parser.add_argument('--tta', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--prefetch', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--wait-seconds', type=nonnegative)
    parser.add_argument('--limit', type=positive)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--share-desktop-pid', type=positive, help='explicit Windows smoke-only sharing with the authorized Codex desktop PID')
    parser.add_argument('--allow-intermediate', action='store_true')
    parser.add_argument('--skip-producer-audit', action='store_true',
                        help='skip the producer allowlist; retain frozen code and model integrity checks')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--stop', action='store_true', help='request stop for --output without killing other jobs')


def execute(args, local):
    print('预测准备：导入计算依赖并核对来源模型，请查看后续阶段日志。', flush=True)
    from .model import read_json
    from .run import discover_training, run_prediction
    paths = prediction_paths(local)
    if args.resume:
        if not args.output:
            raise ValueError('--resume 必须明确指定已有预测目录')
        options = {name: getattr(args, name) for name in ('tta', 'max_gpus', 'device', 'max_batch', 'checkpoint',
                   'gpus', 'gpu_memory_gib', 'threads', 'prefetch', 'wait_seconds')
                   if getattr(args, name) is not None}
        if (args.training_run or args.prepared or args.images or args.cases or args.fold or args.limit
                or args.check_only or args.share_desktop_pid or args.skip_producer_audit):
            raise ValueError('恢复使用原任务已冻结的模型和病例；请省略输入选择参数')
        options.update(output=str(resolve_path(args.output)), resume=True, background=args.background)
        return run_prediction(options)
    from ..config import ROOT
    defaults = read_json(ROOT/'configs/prediction/stunet-s.json')
    options = {key: getattr(args, key) if getattr(args, key, None) is not None else value
               for key, value in defaults.items()}
    if options.get('gpu_memory_gib') is not None:
        import math
        if not math.isfinite(options['gpu_memory_gib']) or options['gpu_memory_gib'] <= 0:
            raise ValueError('显存上限必须是有限正数')
    if args.fold and 'all' in args.fold and args.fold != ['all']:
        raise ValueError('--fold all 不能与编号混用')
    if args.images and (args.cases or args.prepared):
        raise ValueError('--images 使用独立输入，不能同时指定 --cases 或 --prepared')
    if args.training_run and args.fold:
        raise ValueError('--training-run 已明确来源折，请省略 --fold')
    folds = 'all' if args.fold == ['all'] else [int(f) for f in (args.fold or ['0'])]
    runs = [str(resolve_path(p)) for p in args.training_run] if args.training_run else discover_training(paths['training_output_root'], folds)
    prepared = None if args.images else str(resolve_path(args.prepared) if args.prepared else default_prepared(paths))
    root = resolve_path(paths.get('prediction_output_root', 'outputs/prediction'))
    output = resolve_path(args.output) if args.output else root/('smoke' if args.smoke else 'formal')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    options.update(training_runs=runs, prepared=prepared, images=[str(resolve_path(p)) for p in (args.images or [])],
                   cases=args.cases or [], output=str(output), smoke=args.smoke,
                   allow_intermediate=args.allow_intermediate, background=args.background,
                   skip_producer_audit=args.skip_producer_audit,
                   check_only=args.check_only, limit=args.limit or (2 if args.smoke else None))
    if args.share_desktop_pid:
        from ..acceleration.prediction import desktop_identity
        if options['device'] != 'cuda':
            raise ValueError('桌面共享只适用于 CUDA 测试')
        options['shared_desktop'] = desktop_identity(args.share_desktop_pid, args.smoke)
    return run_prediction(options)
