"""Stable protocol-v1 launcher. Select code before importing any project package."""
import json
import os
from pathlib import Path
import sys

PROTOCOL = 1


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
        if python.is_file() and Path(sys.executable).resolve() != python.resolve():
            os.execv(str(python), [str(python), '-B', str(home / (entry if entry == 'smoke' else 'run')) + '.py', *sys.argv[1:]])
    import importlib
    return importlib.import_module('dataset1282_analysis.' + entry).main()
