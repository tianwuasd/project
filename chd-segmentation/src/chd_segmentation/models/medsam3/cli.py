import argparse
import json

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='MedSAM3医学LoRA微调、文本提示预测与独立覆盖评价', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    configure = commands.add_parser('configure',help='在服务器读取已绑定清单，生成私有基础及微调配置')
    configure.add_argument('--root',required=True,help='包含code、course、data和env的服务器根目录')
    for field in ('manifest','cohort','python','official-root','weights-dir'):
        configure.add_argument('--'+field,help='覆盖既有服务器目录约定')
    for name in ('check-training', 'train', 'validate'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True, help='医学LoRA独立训练配置')
        command.add_argument('--prepared', '--preprocessed-source', help='已有切片NPZ的登记JSON')
        if name != 'check-training':
            command.add_argument('--deadline', type=float, help='本次固定Unix截止，受配置max_seconds约束')
        if name == 'train':
            command.add_argument('--resume', action='store_true')
            command.add_argument('--stop-after', type=int, help='本次新增更新次数上限')
        if name == 'validate':
            command.add_argument('--checkpoint', choices=('best', 'latest'), default='best')
    for name in ('predict', 'evaluate', 'inspect'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--case', action='append', required=True)
        if name != 'inspect':
            command.add_argument('--output', required=True)
        if name == 'predict':
            command.add_argument('--stop-after-slices', type=int)
    args = parser.parse_args(argv)
    if args.command == 'configure':
        from .configure import generate
        result = generate(args.root,args.manifest,args.cohort,args.python,args.official_root,args.weights_dir)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return
    if args.command in ('check-training', 'train', 'validate'):
        from .finetune import settings, check, run
        config = settings(args.config, args.prepared)
        if args.command == 'check-training':
            result = check(config)
            if result['ready']:
                result = dict(ready=True, model_loaded=False,
                    samples={k:len(v) for k,v in result['sources'].items()},
                    dependencies=result['dependencies'], environment=result['environment'])
        else:
            result = run(config, args.command, resume=getattr(args, 'resume', False),
                stop_after=getattr(args, 'stop_after', None), checkpoint=getattr(args, 'checkpoint', 'best'),
                deadline=args.deadline)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return
    config = load_config(args.config)
    if args.command == 'inspect':
        from .input import selected_cases
        result = dict(config=config, cases=selected_cases(config, args.case),
            model_loaded=False, data_voxels_read=False)
    elif args.command == 'predict':
        from .runtime import predict
        result = predict(config, args.case, args.output, args.stop_after_slices)
    else:
        from .evaluation import evaluate
        result = evaluate(config, args.case, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
