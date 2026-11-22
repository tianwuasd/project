import argparse
import json

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='SAM3原模型单病例单切片检查，一次文本提示且零训练', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True, help='私有YAML配置，路径以配置文件目录为基准')
        command.add_argument('--case', required=True, help='来源清单中的既有病例标识')
        command.add_argument('--slice', dest='slices', action='append', type=int, required=True,
                             help='唯一一张切片的零基k索引')
        if name == 'predict':
            command.add_argument('--output', required=True, help='工作根内尚不存在的本次独占输出目录')
            command.add_argument('--deadline', type=float, required=True,
                                 help='本批固定截止的Unix秒时间戳，预留120秒诊断清理')
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
