import argparse
import json

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='SAM3原模型有限切片检查，最多两片且零训练', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--case', required=True)
        command.add_argument('--slice', dest='slices', action='append', type=int, required=True)
        if name == 'predict':
            command.add_argument('--output', required=True)
            command.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args(argv)
    if len(args.slices) != 1:
        raise ValueError('本批有限检查只允许一张切片')
    config = load_config(args.config)
    from .runtime import inspect_inputs, predict
    if args.command == 'inspect':
        result = inspect_inputs(config, args.case, args.slices)
    else:
        result = predict(config, args.case, args.slices, args.output, args.deadline)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
