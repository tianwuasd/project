"""CLI for planning, processing and reading private progress records."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

import psutil

from . import __version__
from .config import Config
from .io import now, read_json
from .pipeline import available_memory, plan, run


def parser():
    result = argparse.ArgumentParser(description='AAOCA CT preprocessing: clip HU [-1000,1000] -> [-1,1]. CPU only.')
    result.add_argument('--version', action='version', version=__version__)
    actions = result.add_subparsers(dest='action', required=True)
    for name in ['plan', 'run']:
        command = actions.add_parser(name, help='inspect headers without writing' if name == 'plan' else 'process or verify/reuse existing cases')
        command.add_argument('--input', required=True, type=Path, help='directory of native HU NIfTI CT (never labels)')
        command.add_argument('--manifest', type=Path, help='optional JSON original-CT list, paths relative to --input')
        command.add_argument('--spacing', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='optional nominal millimetre spacing in RAS order; default preserves native resolution')
        command.add_argument('--workers', type=int, default=1, help='concurrent cases, default 1; CPU only')
        command.add_argument('--chunk-depth', type=int, default=8, help='output slices per interpolation slab, default 8')
        command.add_argument('--max-case-gib', type=float, default=8, help='estimated array-memory ceiling per case; default 8 GiB')
        if name == 'run':
            command.add_argument('--output', required=True, type=Path, help='separate empty directory or compatible previous output')
            command.add_argument('--scratch', type=Path, help='optional scratch directory; default OUTPUT/.scratch')
    status = actions.add_parser('status', help='read last recorded progress; does not start or restart work')
    status.add_argument('--output', required=True, type=Path)
    doctor = actions.add_parser('doctor', help='read-only current-machine CPU/memory/path check; no GPU or training')
    doctor.add_argument('--path', action='append', type=Path, default=[], help='inspect an explicit existing path; repeatable')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == 'status':
            data = read_json(args.output / 'progress.json')
        elif args.action == 'doctor':
            data = {'observed_at': now(), 'host': platform.node(), 'system': platform.platform(),
                    'python': platform.python_version(), 'cpu_logical': psutil.cpu_count(),
                    'available_memory_bytes': available_memory(), 'gpu_required': False,
                    'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'nibabel', 'psutil')},
                    'paths': []}
            for path in args.path:
                path = path.expanduser().resolve()
                try:
                    usage = shutil.disk_usage(path)
                    data['paths'].append({'path': str(path), 'exists': path.exists(), 'free_bytes': usage.free,
                                          'total_bytes': usage.total, 'note': 'filesystem space, not personal quota or I/O speed'})
                except OSError as exc:
                    data['paths'].append({'path': str(path), 'exists': path.exists(), 'error': str(exc)})
        else:
            config = Config(target_spacing=tuple(args.spacing) if args.spacing else None,
                            workers=args.workers, chunk_depth=args.chunk_depth, max_case_gib=args.max_case_gib)
            if args.action == 'plan':
                data = plan(args.input, config, args.manifest)
            else:
                def progress(case, state):
                    print(f"[{state['total']-state['pending']}/{state['total']}] {case['state']} {case['case_id']}"
                          + (f" {case['error']}" if 'error' in case else ''), file=sys.stderr, flush=True)
                data = run(args.input, args.output, config, args.manifest, args.scratch, progress)
        print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if (data.get('failed', 0) or data.get('invalid', 0)
                     or data.get('state') in {'failed', 'interrupted', 'completed_with_errors'}) else 0
    except KeyboardInterrupt:
        print('Interrupted. Completed case directories are preserved; repeat the same command to verify and resume.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
