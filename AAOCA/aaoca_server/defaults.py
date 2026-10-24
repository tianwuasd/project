"""Server defaults are fallbacks; saved configuration and explicit choices win."""
import os
from pathlib import Path

WORKSPACE = Path('/data5/zhougaowei/zhangruichen_workspace')
INPUT = '/data_nas/zhangruichen/AAOCA'
RESULTS = '/data_nas/zhangruichen/aaoca_preprocessed'
SERVER_RUNTIME = WORKSPACE / 'aaoca_runtime/panel'
ENVIRONMENT = WORKSPACE / 'aaoca_runtime/venv'
LEGACY_RUNTIME = Path('~/.local/state/aaoca-server').expanduser()


def runtime_default():
    explicit = os.environ.get('AAOCA_SERVER_RUNTIME')
    if explicit:
        return explicit
    for root in (SERVER_RUNTIME, LEGACY_RUNTIME):
        if any((root / name).exists() for name in ('history.json', 'settings.json', 'release.json')):
            return str(root)
    return str(SERVER_RUNTIME)


def environment_default():
    return Path(os.environ.get('AAOCA_PREPROCESS_ENV', str(ENVIRONMENT))).expanduser()
