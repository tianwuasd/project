import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sam-weights', type=Path)
    parser.add_argument('--stunet-weights', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checks', choices=('all', 'models', 'integration'), default='all')
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--models', nargs='+', choices=('xu', 'stunet', 'sam'), default=['xu', 'stunet', 'sam'])
    args = parser.parse_args()
    assert importlib.util.find_spec('dataset1282_analysis.online_input') is not None, '缺少真实在线数据接口'
    from dataset1282_analysis.online_acceptance import accept
    accept(args.prepared, args.output, sam_weights=args.sam_weights,
           stunet_weights=args.stunet_weights, device=args.device, checks=args.checks,
           models=args.models, targets=args.targets)


if __name__ == '__main__':
    main()
