"""Compatibility for the former six-network V2 entry."""
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    sys.argv[1:]=['--engine','multistage',*sys.argv[1:]]
    runpy.run_path(str(Path(__file__).resolve().parent/'start_server_v2.py'),run_name='__main__')
