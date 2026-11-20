import argparse
import json

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='MedSAM3固定文本提示预测与独立覆盖评价', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('predict', 'evaluate', 'inspect'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--case', action='append', required=True)
        if name != 'inspect':
            command.add_argument('--output', required=True)
        if name == 'predict':
            command.add_argument('--stop-after-slices', type=int)
    args = parser.parse_args(argv)
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
