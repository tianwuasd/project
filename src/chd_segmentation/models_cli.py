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
    args = parser.parse_args(argv)
    from .models.workflow import options,train as train_model,predict as predict_model
    config = options(args.config)
    if args.command=='train':
        result = train_model(config,args.prepared,args.stage,args.resume,args.stop_after)
    else:
        prompts = yaml.safe_load(Path(args.prompts).read_text(encoding='utf-8')) if args.prompts else None
        result = predict_model(config,args.prepared,args.checkpoint,args.output,args.case,prompts)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
