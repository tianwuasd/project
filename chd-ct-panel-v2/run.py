"""Project CLI entry; launches no menu and keeps initial imports lightweight."""
from pathlib import Path
import sys
if __name__=='__mp_main__':
    import signal
    signal.signal(signal.SIGINT,signal.SIG_IGN)

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from chd_stunet.cli import main

if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
