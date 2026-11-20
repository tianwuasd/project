import argparse
import json
from pathlib import Path

import yaml


def wait_seconds(value):
    return None if value.lower()=='forever' else float(value)


def main(argv=None):
    import sys
    actual = sys.argv[1:] if argv is None else argv
    if actual[:1] == ['medsam3']:
        from .models.medsam3.cli import main as medsam3_main
        return medsam3_main(actual[1:])
    parser = argparse.ArgumentParser(description='MedSAM2与Q路线一r2模型入口',allow_abbrev=False)
    commands = parser.add_subparsers(dest='command',required=True)
    split = commands.add_parser('split-q1282',help='从冻结1282迁移包生成独立的Q组合')
    split.add_argument('--source',required=True)
    split.add_argument('--output',required=True)
    split.add_argument('--seed',type=int,default=20261104)
    preprocess = commands.add_parser('preprocess',help='按Q或MedSAM2配方保存一次离线输入')
    for name in ('config','prepare-config','prepared','root','output'):
        preprocess.add_argument('--'+name,required=True)
    train = commands.add_parser('train')
    train.add_argument('--config',required=True)
    train.add_argument('--prepared','--preprocessed-source',dest='prepared',required=True)
    train.add_argument('--stage',choices=('base','localizer','local','sequence'),required=True)
    train.add_argument('--resume',action='store_true')
    train.add_argument('--stop-after',type=int)
    predict = commands.add_parser('predict')
    predict.add_argument('--config',required=True)
    predict.add_argument('--prepared','--preprocessed-source',dest='prepared',required=True)
    predict.add_argument('--checkpoint',required=True)
    predict.add_argument('--output',required=True)
    predict.add_argument('--case',action='append')
    predict.add_argument('--prompts')
    predict.add_argument('--prompt-mode',choices=('external','label-box'),default='external')
    predict.add_argument('--label-classes',help='逗号分隔的已知前景类别；默认所有已知且存在的类别')
    predict.add_argument('--box-seed',type=int,default=2024)
    predict.add_argument('--box-max-shift',type=int,default=20)
    predict.add_argument('--official-weights',action='store_true',help='直接使用经过摘要核验的官方MedSAM2发布权重')
    predict.add_argument('--test-set',choices=('internal','external'))
    predict.add_argument('--test-group',choices=('source_test','target_primary','target_secondary'))
    for command in (train,predict):
        command.add_argument('--intensity-mode',choices=('auto','source_window','windowed_uint8','normalized_percentile'))
        command.add_argument('--source-semantics',choices=('hu','clipped_hu','zscore_clipped_hu','windowed_uint8','normalized_float'))
        command.add_argument('--source-mean',type=float)
        command.add_argument('--source-std',type=float)
        command.add_argument('--source-window',nargs=2,type=float)
        command.add_argument('--device',choices=('cpu','cuda'))
        command.add_argument('--gpus',type=int,choices=(1,2,3,4))
        command.add_argument('--gpu-ids')
        command.add_argument('--batch-layout')
        command.add_argument('--workers')
        sharing = command.add_mutually_exclusive_group()
        sharing.add_argument('--share-gpu',dest='share_gpu',action='store_true')
        sharing.add_argument('--no-share-gpu',dest='share_gpu',action='store_false')
        command.set_defaults(share_gpu=None)
        command.add_argument('--wait-seconds',type=wait_seconds,default=argparse.SUPPRESS,
            help='显卡等待秒数；forever持续等待，默认持续等待')
    train.add_argument('--batch',type=int)
    train.add_argument('--epochs',type=int)
    train.add_argument('--updates',type=int)
    train.add_argument('--updates-per-epoch',type=int)
    train.add_argument('--early-stop',action=argparse.BooleanOptionalAction,default=None)
    train.add_argument('--patience',type=int)
    train.add_argument('--min-delta',type=float)
    args = parser.parse_args(actual)
    if args.command=='split-q1282':
        from .datasets.q_split import create_q_bundle
        print(json.dumps(create_q_bundle(args.source,args.output,args.seed),ensure_ascii=False,allow_nan=False))
        return
    if args.command=='preprocess':
        from .models.offline import preprocess_model
        result = preprocess_model(args.config,args.prepare_config,args.prepared,args.root,args.output)
        print(json.dumps(dict(identity=result['identity'],data=result['storage']['root'],
            cases=len(result['storage']['cases']),complete=True),ensure_ascii=False,allow_nan=False))
        return
    from .models.workflow import options,train as train_model,predict as predict_model
    execution = {}
    if args.gpus is not None:
        execution['gpu_count'] = args.gpus
    if args.gpu_ids:
        execution['gpu_ids'] = args.gpu_ids.split(',')
        if args.gpus is not None and args.gpus!=len(execution['gpu_ids']):
            parser.error('--gpus与--gpu-ids数量不同')
        execution['gpu_count'] = len(execution['gpu_ids'])
    if args.batch_layout:
        execution['batch_layout'] = list(map(int,args.batch_layout.split(',')))
    if args.workers is not None:
        execution['workers'] = 'auto' if args.workers=='auto' else int(args.workers)
    if hasattr(args,'wait_seconds'):
        execution['wait_seconds'] = args.wait_seconds
    if args.share_gpu is not None:
        execution['share'] = args.share_gpu
    overrides = dict(execution=execution)
    if args.device is not None:
        overrides['device'] = args.device
    if args.command=='train' and args.batch is not None:
        overrides['global_batch'] = args.batch
    if args.command == 'train':
        for name in ('epochs','updates','updates_per_epoch'):
            if getattr(args,name) is not None:
                overrides[name] = getattr(args,name)
        stopping = {key:value for key,value in dict(enabled=args.early_stop,
            patience=args.patience,min_delta=args.min_delta).items() if value is not None}
        if stopping:
            overrides['early_stopping'] = stopping
    source = {key:value for key,value in dict(intensity_mode=args.intensity_mode,
        intensity_semantics=args.source_semantics,mean=args.source_mean,std=args.source_std,
        gray_window=args.source_window).items() if value is not None}
    if source:
        overrides['input_source'] = source
    config = options(args.config,overrides)
    from .observability.events import Stage
    with Stage(None,'model_command',heartbeat_seconds=config['logging']['heartbeat_seconds'],
               model=config['model'],operation=args.command):
        result = execute(args,config,train_model,predict_model)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


def execute(args,config,train_model,predict_model):
    if args.command=='train':
        result = train_model(config,args.prepared,args.stage,args.resume,args.stop_after)
    else:
        prompts = yaml.safe_load(Path(args.prompts).read_text(encoding='utf-8')) if args.prompts else None
        label_classes = list(map(int,args.label_classes.split(','))) if args.label_classes else None
        checkpoint = args.checkpoint
        if checkpoint in ('best','latest'):
            stage = 'local' if config['model']=='q_route1' else 'sequence'
            checkpoint = str(Path(config.get('checkpoint_dir') or config['run_dir'])/stage/(checkpoint+'.pt'))
        result = predict_model(config,args.prepared,checkpoint,args.output,args.case,prompts,
            prompt_mode=args.prompt_mode,label_classes=label_classes,box_seed=args.box_seed,
            box_max_shift=args.box_max_shift,official_weights=args.official_weights,
            test_set=args.test_set,test_group=args.test_group)
    return result
