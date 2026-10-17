"""Locate optional venv-local Tcl/Tk resources for portable Windows runtimes."""
import os
from pathlib import Path
import sys


def configure_tk():
    resource_root = Path(sys.prefix) / 'tcl'
    for variable, child, entry in [('TCL_LIBRARY','tcl8.6','init.tcl'),
                                   ('TK_LIBRARY','tk8.6','tk.tcl')]:
        candidate = resource_root / child
        if (candidate / entry).is_file():
            os.environ.setdefault(variable, str(candidate))
