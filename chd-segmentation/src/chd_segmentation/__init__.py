__version__ = "0.1.0"

import importlib.util as _util
from pathlib import Path as _Path

_bootstrap = _Path(__file__).resolve().parents[2] / 'code_runtime.py'
if _bootstrap.is_file():
    _spec = _util.spec_from_file_location('_chd_code_runtime', _bootstrap)
    _runtime = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_runtime)
    _runtime.activate_package(globals())
