"""Compatibility entry; the maintained V2 now lives inside chd-ct-reproduction."""
from pathlib import Path
import runpy

if __name__=='__main__':
    target=Path(__file__).resolve().parent.parent/'chd-ct-reproduction/start_server_v2.py'
    if not target.is_file():
        raise SystemExit('Update the complete main checkout; V2 now starts from chd-ct-reproduction/start_server_v2.sh.')
    runpy.run_path(str(target),run_name='__main__')
