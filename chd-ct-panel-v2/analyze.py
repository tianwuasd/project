"""Direct local-data analysis entry, using the same implementation as run.py."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from chd_stunet.cli import main

if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main(["analyze",*sys.argv[1:]]))
