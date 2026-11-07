"""Standard-library-only environment orchestration, separate from training."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from .contracts import atomic_json, read_json
from .observability.status import operation_lock

ROOT = Path(__file__).resolve().parents[2]
TORCH_INDEX = 'https://download.pytorch.org/whl/cu126'
TORCH_SPEC = 'torch==2.12.1+cu126'
OWNER = {'project': 'dataset1282_analysis', 'purpose': 'gpu-training', 'schema': 1}

# This probe does not reserve a GPU or start a model. Capacity/NCCL qualification
# remains the responsibility of xu_fleet_probe.py in the gated trial command.
PROBE = r'''
import importlib, importlib.metadata, json, os, sys
result = dict(python=sys.executable, python_version=list(sys.version_info[:3]),
              supported_python=(3,11)<=sys.version_info[:2]<(3,14),
              versions={}, errors={}, cuda=None, cuda_available=False, device_count=0)
for package, module in [('torch','torch'),('numpy','numpy'),('scipy','scipy'),
    ('nibabel','nibabel'),('scikit-image','skimage'),('blosc2','blosc2'),
    ('psutil','psutil'),('threadpoolctl','threadpoolctl'),('matplotlib','matplotlib')]:
    try:
        importlib.import_module(module)
        result['versions'][package] = importlib.metadata.version(package)
    except Exception as exc:
        result['errors'][package] = str(exc)
if 'torch' not in result['errors']:
    try:
        import torch
        result['cuda'] = torch.version.cuda
        result['cuda_available'] = torch.cuda.is_available()
        result['device_count'] = torch.cuda.device_count()
        result['nccl'] = torch.distributed.is_available() and torch.distributed.is_nccl_available()
        assert hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler')
        x = torch.ones((2,2), requires_grad=True)
        (x*x).sum().backward()
        assert x.grad.tolist() == [[2.,2.],[2.,2.]]
    except Exception as exc:
        result['errors']['torch_api'] = str(exc)
if not result['errors']:
    try:
        import numpy as np, blosc2, nibabel as nib, psutil
        from scipy.ndimage import zoom
        from skimage.morphology import skeletonize
        from skimage.segmentation import watershed
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from threadpoolctl import threadpool_limits
        a = np.ones((3,3,3), dtype=np.float32)
        np.testing.assert_array_equal(blosc2.asarray(a)[:], a)
        assert zoom(a, 1).shape == a.shape
        assert nib.Nifti1Image(a, np.eye(4)).shape == a.shape
        assert skeletonize(a.astype(bool)).shape == a.shape
        assert watershed(a, np.ones_like(a,dtype=np.int32)).shape == a.shape
        assert psutil.virtual_memory().total > 0
        FigureCanvasAgg(Figure(figsize=(1,1))).draw()
        with threadpool_limits(limits=1): pass
    except Exception as exc:
        result['errors']['data_api'] = str(exc)
if os.name == 'posix' and result['device_count'] > 1 and not result.get('nccl'):
    result['errors']['nccl'] = 'Multiple visible GPUs require an NCCL-enabled torch build'
result['ready'] = bool(result['supported_python'] and not result['errors'] and
                       result['cuda_available'] and result['device_count'] > 0)
print('GPU_ENV_JSON=' + json.dumps(result))
'''


def interpreter(path):
    # Resolving the final bin/python symlink escapes a Linux venv.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def venv_python(path):
    return Path(path) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def child_environment():
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


def probe(python):
    python = interpreter(python)
    if not python.is_file():
        return dict(python=str(python), ready=False, error='missing_interpreter')
    try:
        proc = subprocess.run([str(python), '-B', '-c', PROBE], env=child_environment(),
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        lines = [line for line in proc.stdout.splitlines() if line.startswith('GPU_ENV_JSON=')]
        if proc.returncode or not lines:
            raise RuntimeError((proc.stderr or proc.stdout)[-4000:] or 'Probe returned no result')
        result = json.loads(lines[-1].split('=', 1)[1])
        result['python'] = str(python)
        if proc.stderr.strip():
            result['warnings'] = proc.stderr[-4000:]
        return result
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        return dict(python=str(python), ready=False, error=str(exc))


def candidates(runtime, explicit=None, root=ROOT):
    paths = [explicit] if explicit else []
    paths += [venv_python(runtime/'training'), venv_python(runtime/'research'),
              venv_python(root.parent/'chd-ct-panel-v2'/'.venv_preprocess')]
    return list(dict.fromkeys(interpreter(p) for p in paths))


def scan(runtime, explicit=None):
    rows = []
    for path in candidates(runtime, explicit):
        row = probe(path)
        rows.append(row)
        print(('READY ' if row['ready'] else 'CHECK ') + str(path), flush=True)
        if row['ready']:
            return rows, row
    return rows, None


def gpu_driver():
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) in ('', '-1'):
        raise RuntimeError('CUDA_VISIBLE_DEVICES hides all GPUs; request a GPU allocation first')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError('nvidia-smi cannot see a working NVIDIA driver/GPU')
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('No accessible NVIDIA driver; pip cannot install or repair the system driver') from exc


def choose_base(explicit=None, runtime=None):
    paths = [explicit] if explicit else [shutil.which('python3.11'), sys.executable]
    if runtime is not None and not explicit:
        paths += [str(p) for p in candidates(runtime) if p.is_file()]
    for path in filter(None, paths):
        result = subprocess.run([str(interpreter(path)), '-B', '-c',
                                 'import sys,venv;sys.exit(not ((3,11)<=sys.version_info[:2]<(3,14)))'],
                                capture_output=True, timeout=30)
        if result.returncode == 0:
            return str(interpreter(path))
    raise RuntimeError('A base Python 3.11-3.13 with venv is required; use --base-python PATH')


def run_logged(command, log, env, timeout=1800):
    log.write('\n$ ' + shlex.join(command) + '\n')
    log.flush()
    started = time.monotonic()
    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    log.write(f'\nexit={result.returncode} seconds={time.monotonic()-started:.2f}\n')
    log.flush()
    if result.returncode:
        raise RuntimeError(f'Environment command failed (exit {result.returncode}); see installation log')


def install(runtime, base=None, proxy=None):
    target = runtime/'training'
    marker = target/'.dataset1282-environment.json'
    if target.exists():
        if target.is_symlink() or not marker.is_file() or read_json(marker) != OWNER:
            raise RuntimeError('Existing training directory is not owned by this installer; refusing to modify it')
    else:
        target.mkdir()
        atomic_json(marker, OWNER)
    env = child_environment()
    # Keep package and proxy choices local to this child, with no global pip edits.
    for key in list(env):
        if key.startswith('PIP_') or key.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):
            env.pop(key, None)
    env['PIP_CONFIG_FILE'] = os.devnull
    if proxy:
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
    logfile = runtime/'training-install.log'
    print('Installation log: ' + str(logfile), flush=True)
    with logfile.open('a', encoding='utf-8') as log:
        if not (target/'pyvenv.cfg').is_file() or not venv_python(target).is_file():
            print('Creating separate training venv', flush=True)
            run_logged([choose_base(base, runtime), '-m', 'venv', str(target)], log, env, timeout=180)
        python = str(venv_python(target))
        pip = [python, '-m', 'pip', '--disable-pip-version-check', 'install',
               '--only-binary=:all:', '--timeout', '60', '--retries', '2']
        print('Checking/installing CUDA PyTorch', flush=True)
        run_logged([*pip, TORCH_SPEC, '--index-url', TORCH_INDEX], log, env)
        print('Checking/installing training dependencies', flush=True)
        run_logged([*pip, '-r', str(ROOT/'configs/training-requirements.txt'),
                    '--index-url', 'https://pypi.org/simple'], log, env)
        run_logged([python, '-m', 'pip', 'check'], log, env, timeout=120)
        with (runtime/'training-freeze.txt').open('w', encoding='utf-8') as frozen:
            subprocess.run([python, '-m', 'pip', 'freeze'], stdout=frozen, stderr=log,
                           env=env, check=True, timeout=120)
    return probe(python)


def ensure(runtime, explicit=None, base=None, proxy=None):
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    with operation_lock(runtime, 'gpu-environment'):
        shell = runtime/'training-env.sh'
        # Never leave an old success pointer behind after a failed new check.
        shell.unlink(missing_ok=True)
        report = dict(state='checking', checked_at=time.time(), candidates=[])
        try:
            rows, selected = scan(runtime, explicit)
            report['candidates'] = rows
            if selected is None:
                report['drivers'] = gpu_driver()
                if any(r.get('cuda') and not r.get('cuda_available') for r in rows):
                    raise RuntimeError('CUDA PyTorch is installed but GPU access failed; inspect candidates in report before downloading again')
                report['action'] = 'install_separate_training_environment'
                selected = install(runtime, base, proxy)
            else:
                report['action'] = 'reuse_without_changes'
            report['selected'] = selected
            if not selected['ready']:
                raise RuntimeError('Installed runtime failed verification; inspect selected in report')
            report['state'] = 'ready'
            # Publication only after all checks succeed, under the operation lock.
            staged = shell.with_suffix('.tmp')
            with staged.open('w', encoding='utf-8', newline='\n') as stream:
                stream.write('export PY=' + shlex.quote(selected['python']) + '\n')
            os.replace(staged, shell)
            return report
        except Exception as exc:
            report.update(state='failed', error=str(exc))
            raise
        finally:
            atomic_json(runtime/'training-environment.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Check only; no environment changes or installation')
    mode.add_argument('--install', action='store_true', help='Reuse a passing environment or install into RUNTIME/training')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--reuse-python', help='Check this interpreter first; never install into it')
    parser.add_argument('--base-python', help='Python 3.11-3.13 used only to create a new venv')
    parser.add_argument('--proxy', help='Explicit proxy used only by installation children')
    args = parser.parse_args(argv)
    try:
        if args.check:
            rows, selected = scan(args.runtime.resolve(), args.reuse_python)
            report = dict(state='ready' if selected else 'unavailable', candidates=rows, selected=selected)
        else:
            report = ensure(args.runtime, args.reuse_python, args.base_python, args.proxy)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report['state'] == 'ready' and args.install:
            print('Source this file before the trial: ' + str(args.runtime.resolve()/'training-env.sh'))
        return 0 if report['state'] == 'ready' else 1
    except Exception as exc:
        print('Environment setup failed: ' + str(exc), file=sys.stderr)
        return 1
