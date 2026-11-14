import argparse
import json
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description='MedSAM2与Q路线一r2模型入口',allow_abbrev=False)
    commands = parser.add_subparsers(dest='command',required=True)
    train = commands.add_parser('train')
    train.add_argument('--config',required=True)
    train.add_argument('--prepared',required=True)
    train.add_argument('--stage',choices=('base','localizer','local','sequence'),required=True)
    train.add_argument('--resume',action='store_true')
    train.add_argument('--stop-after',type=int)
    predict = commands.add_parser('predict')
    predict.add_argument('--config',required=True)
    predict.add_argument('--prepared',required=True)
    predict.add_argument('--checkpoint',required=True)
    predict.add_argument('--output',required=True)
    predict.add_argument('--case',action='append')
    predict.add_argument('--prompts')
    for command in (train,predict):
        command.add_argument('--device',choices=('cpu','cuda'))
        command.add_argument('--gpus',type=int,choices=(1,2,3,4))
        command.add_argument('--gpu-ids')
        command.add_argument('--batch-layout')
        command.add_argument('--workers')
    train.add_argument('--batch',type=int)
    args = parser.parse_args(argv)
    from .models.workflow import options,train as train_model,predict as predict_model
    execution = {}
    if args.gpus is not None:
        execution['gpu_count'] = args.gpus
    if args.gpu_ids:
        execution['gpu_ids'] = args.gpu_ids.split(',')
        execution['gpu_count'] = len(execution['gpu_ids'])
    if args.batch_layout:
        execution['batch_layout'] = list(map(int,args.batch_layout.split(',')))
    if args.workers is not None:
        execution['workers'] = 'auto' if args.workers=='auto' else int(args.workers)
    overrides = dict(execution=execution)
    if args.device is not None:
        overrides['device'] = args.device
    if args.command=='train' and args.batch is not None:
        overrides['global_batch'] = args.batch
    config = options(args.config,overrides)
    if args.command=='train':
        result = train_model(config,args.prepared,args.stage,args.resume,args.stop_after)
    else:
        prompts = yaml.safe_load(Path(args.prompts).read_text(encoding='utf-8')) if args.prompts else None
        result = predict_model(config,args.prepared,args.checkpoint,args.output,args.case,prompts)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
