import argparse
import getpass
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from . import __version__
from .operations import apply_license, install, probe, python_path, runtime_env, stage_license
from .resources import select_devices
from .schema import export_inventory, read_manifest
from .service_state import CODE_ROOT, file_lock, private_dir, read_json
from .workbench import DEFAULTS, Workbench, status


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('必须为正数')
    return number


def gpu_indices(value):
    values = value.split(',')
    if any(not item.strip().isdigit() for item in values):
        raise argparse.ArgumentTypeError('GPU 编号须为逗号分隔的非负整数')
    indices = [int(item) for item in values]
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError('GPU 编号不能重复')
    return indices


def parser():
    root = argparse.ArgumentParser(description='AAOCA 原始 CT 三模型直接分割')
    root.add_argument('--version', action='version', version=__version__)
    sub = root.add_subparsers(dest='action', required=True)
    for name in ('install', 'license', 'check', 'inventory', 'plan', 'run', 'resume', 'status', 'logs', 'stop'):
        command = sub.add_parser(name)
        command.add_argument('--runtime', default=os.environ.get('AAOCA_SEG_RUNTIME', '~/.local/state/aaoca-seg'))
        if name == 'install':
            command.add_argument('--torch-build', choices=['auto', 'cpu', 'cu118', 'cu126'], default='auto')
        if name == 'license':
            command.add_argument('--stdin', action='store_true', help='从标准输入读取许可，不写入命令参数')
        if name == 'inventory':
            command.add_argument('--inventory', required=True)
            command.add_argument('--manifest', required=True)
            command.add_argument('--data-root')
        if name in {'plan', 'run'}:
            command.add_argument('--manifest', required=True)
            command.add_argument('--output-root', required=True)
            command.add_argument('--case-id', action='append', dest='case_ids')
            command.add_argument('--limit', type=int)
            command.add_argument('--threads', type=int, default=2)
        if name == 'plan':
            command.add_argument('--device', default='cpu')
        if name == 'run':
            devices = command.add_mutually_exclusive_group(required=True)
            devices.add_argument('--gpus', type=gpu_indices, help='明确获准使用的可见 GPU 编号，例如 0,1')
            devices.add_argument('--cpu', action='store_true')
            command.add_argument('--name', default='AAOCA')
            command.add_argument('--workers', type=int, help='病例并发上限，默认等于所选 GPU 数量')
            command.add_argument('--gpu-count', type=int, help='最多同时使用的 GPU 数，默认等于所选数量')
            command.add_argument('--min-free-gib', type=positive, default=8)
            command.add_argument('--gpu-reserve-gib', type=positive, default=2)
            command.add_argument('--cpu-limit', type=int)
            command.add_argument('--memory-gib', type=positive)
            command.add_argument('--io-workers', type=int)
        if name in {'run', 'resume'}:
            command.add_argument('--background', action='store_true', help='提交后立即返回；任务保留在后台运行')
        if name in {'resume', 'status', 'logs', 'stop'}:
            command.add_argument('--job', help='任务编号；省略时使用最近的分割任务')
    return root


def prepare_runtime(workspace):
    private_dir(workspace)
    for name in ('tmp', 'pip-cache'):
        private_dir(workspace / name)
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                      TMPDIR=str(workspace / 'tmp'), TEMP=str(workspace / 'tmp'), TMP=str(workspace / 'tmp'),
                      PIP_CACHE_DIR=str(workspace / 'pip-cache'))


def enter_environment(workspace, argv):
    executable = python_path(workspace)
    if not executable.is_file():
        raise ValueError('运行环境不存在，请先执行 install --runtime 指定运行目录')
    if Path(sys.prefix).resolve() != executable.parent.parent.resolve():
        env = runtime_env(workspace)
        env['PYTHONPATH'] = str(CODE_ROOT)
        os.execve(str(executable), [str(executable), '-B', '-m', 'aaoca_seg.direct', *argv], env)
    os.environ.update(runtime_env(workspace))


def resolve_job(work, job_id):
    if job_id:
        return work.resolve(job_id)
    entries = work.entries()
    return Path(entries[0]['job']) if entries else None


def report(work, job):
    if job is None:
        return {'status': 'empty'}
    request = read_json(job / 'request.json')
    return {'job': job.name, 'output': request['output'], 'state': status(job),
            'progress': read_json(Path(request['output']) / 'status.json', {})}


def wait_job(work, job, background):
    print(json.dumps({'job': job.name, 'output': read_json(job / 'request.json')['output'],
                      'log': str(job / 'queue.log')}, ensure_ascii=False), flush=True)
    if background:
        return 0
    def requested(*_):
        (job / 'stop').touch()
    signal.signal(signal.SIGINT, requested)
    signal.signal(signal.SIGTERM, requested)
    next_report = 0
    while status(job)['active']:
        if time.monotonic() >= next_report:
            progress = report(work, job)['progress']
            print(json.dumps({key: progress.get(key) for key in
                              ('status', 'finished', 'total', 'failed', 'waiting_gpu', 'waiting_disk', 'active')},
                             ensure_ascii=False), flush=True)
            next_report = time.monotonic() + 10
        time.sleep(.25)
    final = report(work, job)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0 if final['state']['status'] == 'completed' else (130 if final['state']['status'] == 'stopped' else 2)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    workspace = Path(args.runtime).expanduser().resolve()
    work = Workbench(workspace)
    if args.action in {'status', 'logs', 'stop', 'resume'}:
        job = resolve_job(work, args.job)
        if args.action == 'status':
            print(json.dumps(report(work, job), ensure_ascii=False, indent=2))
            return 0
        if job is None:
            raise ValueError('没有可用的分割任务')
        if args.action == 'logs':
            print(work.logs(job))
            return 0
        if args.action == 'stop':
            print(work.stop(job.name))
            return 0
    if args.action == 'inventory':
        count = export_inventory(args.inventory, args.manifest, args.data_root)
        print(json.dumps({'volumes': count, 'manifest': str(Path(args.manifest).resolve())}, ensure_ascii=False))
        return 0
    if args.action == 'plan':
        enter_environment(workspace, argv)
        from .pipeline import plan
        cases = read_manifest(args.manifest, args.case_ids, args.limit)
        result = plan(cases, args.output_root, args.device, args.threads)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['ready'] == len(cases) else 2
    prepare_runtime(workspace)
    if args.action == 'install':
        work.idle()
        with file_lock(workspace / 'operation.lock'):
            install(workspace, {'torch_build': args.torch_build})
        return 0
    enter_environment(workspace, argv)
    if args.action == 'license':
        work.idle()
        secret = sys.stdin.read().strip() if args.stdin else getpass.getpass('TotalSegmentator 许可码（不回显）：')
        with file_lock(workspace / 'operation.lock'):
            stage_license(workspace, secret)
            apply_license(workspace)
        print('许可验证完成。')
        return 0
    if args.action == 'check':
        print(json.dumps(probe(workspace), ensure_ascii=False, indent=2))
        return 0
    if args.action == 'resume':
        work.start(job)
        return wait_job(work, job, args.background)
    cases = read_manifest(args.manifest, args.case_ids, args.limit)
    devices = select_devices(probe(workspace), args.gpus) if args.gpus is not None else []
    settings = {**DEFAULTS, 'manifest': str(Path(args.manifest).resolve()),
                'results': str(Path(args.output_root).resolve()), 'mode': 'gpu' if devices else 'cpu',
                'devices': devices, 'gpu_count': args.gpu_count if args.gpu_count is not None else max(1, len(devices)),
                'workers': args.workers if args.workers is not None else max(1, len(devices)),
                'threads': args.threads, 'cpu_limit': args.cpu_limit,
                'memory_gib': args.memory_gib,
                'io_workers': args.io_workers if args.io_workers is not None else max(1, len(devices)),
                'min_gpu_gib': args.min_free_gib, 'allow_shared_gpu': True,
                'gpu_reserve_gib': args.gpu_reserve_gib}
    job = work.create(args.name, ['plan', 'infer', 'merge'], settings=settings,
                      case_ids=[case['case_id'] for case in cases])
    work.start(job)
    return wait_job(work, job, args.background)


if __name__ == '__main__':
    raise SystemExit(main())
