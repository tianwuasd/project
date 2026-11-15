import argparse
from pathlib import Path
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def run(args):
    sys.path.insert(0, str(Path(args.simulator).resolve(strict=True)))
    from common import clean_owned, source_identity, supervise, write
    work, report = Path(args.work_root).resolve(), Path(args.report).resolve()
    if work.exists():
        raise FileExistsError('监督工作目录必须尚不存在')
    work.mkdir(parents=True)
    token = uuid4().hex
    write(work/'owner.json', dict(token=token, source=str(ROOT)))
    scripts = dict(q='verify_models_real.py', meddata='verify_model_supply_real.py',
        medpredict='verify_external_medsam_real.py', labelbox='verify_label_box_real.py',
        updater='verify_assistant2_branch_update.py', lifecycle='verify_model_offline_real.py')
    command = [sys.executable, '-B', str(ROOT/'tests'/scripts[args.stage])]
    command += ['--work-root', str(work/'case-work')]
    command += ['--report-root' if args.stage=='q' else '--report', str(report/'acceptance')]
    if args.stage != 'updater':
        command += ['--spec', str(Path(args.spec).resolve(strict=True))]
    if args.stage == 'q':
        command += ['--model', 'q_route1', '--dataset', 'imagechd', '--world', '2']
    if args.stage == 'lifecycle':
        command += ['--baseline', str(Path(args.baseline).resolve(strict=True))]
    if args.stage == 'meddata':
        command += ['--world', str(args.meddata_world)]
    write(report/'source.json', source_identity(ROOT))
    try:
        supervise(command, ROOT, work, report, timeout=args.timeout)
    finally:
        clean_owned(work, work.parent, token, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='D:/projects/test')
    parser.add_argument('--stage', choices=('q', 'meddata', 'medpredict', 'labelbox', 'updater', 'lifecycle'), required=True)
    for key in ('work-root', 'report'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--spec')
    parser.add_argument('--baseline')
    parser.add_argument('--meddata-world', type=int, choices=(1,2), default=2)
    parser.add_argument('--timeout', type=float, default=900)
    run(parser.parse_args())
