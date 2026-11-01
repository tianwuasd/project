"""V2 entry beside the original V1; only the isolated v2 package is loaded."""
from pathlib import Path
import runpy

if __name__=='__main__':
    target=Path(__file__).resolve().parent/'v2/start_server_v2.py'
    if not target.is_file():
        raise SystemExit('V2 code is incomplete. Update main with the complete chd-ct-reproduction/v2 directory.')
    runpy.run_path(str(target),run_name='__main__')
