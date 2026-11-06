"""Environment management uses only stdlib until the target interpreter's API probe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from .config import resolve_config
from .contracts import atomic_json, sha256_file

API_PROBE = r'''
import importlib.metadata as m, json, pathlib, sys, tempfile
packages = ['numpy','nibabel','scipy','openpyxl','matplotlib','psutil']
versions, errors = {}, []
for name in packages:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: errors.append('missing_package:' + name)
if not errors:
    try:
        import numpy as np, nibabel as nib, openpyxl, psutil, matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from scipy import ndimage
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            p = d / 'probe.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((3,4,5),dtype=np.int16), np.eye(4)), p)
            assert np.asarray(nib.load(p).dataobj).sum() == 60
            assert ndimage.label(np.ones((3,4,5)), ndimage.generate_binary_structure(3,1))[1] == 1
            w = openpyxl.Workbook(); w.active.append(['id', '1234567890123456']); w.save(d/'probe.xlsx')
            r = openpyxl.load_workbook(d/'probe.xlsx',read_only=True); assert r.active['B1'].value == '1234567890123456'; r.close()
            fig, ax = plt.subplots(); ax.plot([0,1]); fig.savefig(d/'probe.png'); plt.close(fig)
            assert psutil.virtual_memory().available > 0
    except Exception as e: errors.append('api_failure:' + type(e).__name__ + ':' + str(e))
print(json.dumps(dict(python=sys.executable,python_version=sys.version,packages=versions,errors=errors,ok=not errors)))
sys.exit(bool(errors))
'''


def environment_python(env):
    return Path(env) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def check_environment(python_path, lock_path):
    if not Path(python_path).is_file():
        return {'ok': False, 'errors': ['missing_interpreter'], 'python': str(python_path)}
    result = subprocess.run([str(python_path), '-B', '-c', API_PROBE], capture_output=True,
                            text=True, encoding='utf-8', timeout=180)
    try:
        report = json.loads(result.stdout)
    except ValueError:
        return {'ok': False, 'errors': ['probe_failed'], 'detail': result.stderr[-2000:]}
    from importlib import metadata  # stdlib; do not inspect the caller environment's packages
    del metadata
    expected = {}
    for line in Path(lock_path).read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            k, v = line.strip().split('==', 1)
            expected[k.lower().replace('_', '-')] = v
    # Query every locked distribution, including transitive packages.
    code = 'import importlib.metadata as m,json;print(json.dumps({x:m.version(x) for x in ' + repr(list(expected)) + '}))'
    query = subprocess.run([str(python_path), '-B', '-c', code], capture_output=True, text=True, encoding='utf-8', timeout=30)
    if query.returncode:
        report['errors'].append('locked_distribution_missing')
    else:
        actual = json.loads(query.stdout)
        report['locked_packages'] = actual
        report['errors'].extend('version_mismatch:' + k for k, v in expected.items() if actual.get(k) != v)
    report['lock_sha256'] = sha256_file(lock_path)
    report['ok'] = not report['errors'] and result.returncode == 0
    return report


def install_environment(config, code_root, base_python=None, proxy=None, index_url='https://pypi.org/simple'):
    env = Path(config['environment'])
    work = Path(config['work_root'])
    logdir = work / 'environment_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.log')
    python = environment_python(env)
    if env.exists() and not (env / 'pyvenv.cfg').is_file():
        raise ValueError('Refusing to install into an existing non-venv directory')
    child_env = os.environ.copy()
    child_env['PYTHONUTF8'] = '1'
    child_env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        child_env.pop(key, None)
    if proxy:
        child_env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
    commands = []
    if not python.is_file():
        commands.append([base_python or sys.executable, '-m', 'venv', str(env)])
    commands.append([str(python), '-m', 'pip', 'install', '--retries', '1', '--timeout', '45',
                     '--only-binary=:all:', '--index-url', index_url, '-r', str(Path(code_root) / 'requirements-lock.txt')])
    with log.open('w', encoding='utf-8') as stream:
        for command in commands:
            stream.write('Stage: ' + ('create_environment' if 'venv' in command else 'install_locked_packages') + '\n')
            stream.flush()
            result = subprocess.run(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
            if result.returncode:
                return {'ok': False, 'errors': ['installation_failed'], 'log': str(log)}
    report = check_environment(python, Path(code_root) / 'requirements-lock.txt')
    report['log'] = str(log)
    atomic_json(work / 'environment-report.json', report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--plan', action='store_true')
    action.add_argument('--check', action='store_true')
    action.add_argument('--install', action='store_true')
    p.add_argument('--env'); p.add_argument('--base-python'); p.add_argument('--proxy')
    p.add_argument('--index-url', default='https://pypi.org/simple')
    args = p.parse_args(argv)
    root = Path(os.environ.get('DATASET1282_CODE_ROOT', Path(__file__).resolve().parents[2]))
    home = Path(os.environ.get('DATASET1282_HOME', root))
    config = resolve_config({'environment': args.env}, home / 'project.local.json', root)
    if args.plan:
        report = {'ok': True, 'environment': config['environment'], 'python': str(environment_python(config['environment'])),
                  'lock': str(root / 'requirements-lock.txt'), 'lock_sha256': sha256_file(root / 'requirements-lock.txt'),
                  'actions': ['create dedicated venv if absent', 'install locked wheels', 'verify APIs and versions'],
                  'modifies_existing_research_environments': False}
    elif args.check:
        report = check_environment(environment_python(config['environment']), root / 'requirements-lock.txt')
    else:
        report = install_environment(config, root, args.base_python, args.proxy, args.index_url)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1
