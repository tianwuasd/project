"""Two real source pairs through the same read-only analysis pipeline."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from chd_stunet.cli import main

if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    arguments=sys.argv[1:]
    if '--stage' in arguments:
        at=arguments.index('--stage')
        if at+1>=len(arguments) or arguments[at+1] not in ('analyze','prepare'):
            raise SystemExit('--stage requires analyze or prepare')
        stage=arguments[at+1]
        arguments=arguments[:at]+arguments[at+2:]
    else:
        stage='analyze'
    if stage=='prepare':
        # Uses the full source audit and full training-only fit. Only the output
        # cases are limited; the native/resampled sizes are never downscaled.
        raise SystemExit(main(['prepare','--method','stunet-s','--fold','0','--limit','2',*arguments]))
    raise SystemExit(main(["analyze","--limit","2","--components","none","--tag","smoke",*arguments]))
