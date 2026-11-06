"""Stable protocol-v1 launcher. Select code before importing any project package."""
import json
import os
from pathlib import Path
import sys

PROTOCOL = 1


def interpreter_path(path):
    """Keep the venv invocation path; Linux venv binaries often link to system Python."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def resolve_launch_config(home,root,argv):
    import argparse
    from dataset1282_analysis.config import resolve_config
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config')
    args,_=parser.parse_known_args(argv)
    return resolve_config({},Path(home)/'project.local.json',root,args.config)


def select_root(root):
    root = Path(root).resolve()
    pointer = root / 'current.json'
    if not pointer.exists():
        return root
    value = json.loads(pointer.read_text(encoding='utf-8'))
    commit = value['commit']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Invalid active release identity')
    release = root / 'code_versions' / commit
    if not release.is_dir() or release.is_symlink():
        raise ValueError('Active release is missing or linked')
    # Full release digest verification is shared with the stdlib-only updater.
    sys.path.insert(0, str(root / 'src'))
    from dataset1282_analysis.maintenance import verify_release
    verify_release(release)
    for key in list(sys.modules):
        if key == 'dataset1282_analysis' or key.startswith('dataset1282_analysis.'):
            del sys.modules[key]
    sys.path.remove(str(root / 'src'))
    return release


def launch(entry):
    # JSON/log consumers use UTF-8 even when the Windows console code page is GBK.
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    home = Path(__file__).resolve().parent
    try:
        root = select_root(home)
    except (OSError,ValueError,KeyError):
        if entry!='maintenance': raise
        # The stable base updater remains reachable if the active release is damaged.
        root=home
        print('Active release unavailable; using stable base recovery updater.',file=sys.stderr)
        for key in list(sys.modules):
            if key=='dataset1282_analysis' or key.startswith('dataset1282_analysis.'):
                del sys.modules[key]
    sys.path.insert(0, str(root / 'src'))
    os.environ['DATASET1282_HOME'] = str(home)
    os.environ['DATASET1282_CODE_ROOT'] = str(root)
    # The environment/update entrypoints remain usable with the base Python.
    if entry not in ('environment', 'maintenance'):
        config = resolve_launch_config(home,root,sys.argv[1:])
        python = Path(config['environment']) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if python.is_file() and interpreter_path(sys.executable) != interpreter_path(python):
            if os.environ.get('DATASET1282_REEXEC') == interpreter_path(python):
                raise RuntimeError('Configured interpreter did not retain its environment identity')
            os.environ['DATASET1282_REEXEC'] = interpreter_path(python)
            os.environ['PYTHONUTF8'] = '1'
            os.execv(str(python), [str(python), '-B', str(home / (entry if entry == 'smoke' else 'run')) + '.py', *sys.argv[1:]])
    import importlib
    return importlib.import_module('dataset1282_analysis.' + entry).main()
