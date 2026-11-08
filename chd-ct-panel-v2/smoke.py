"""Two real source pairs through the same read-only analysis pipeline."""
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
    arguments=sys.argv[1:]
    if '--stage' in arguments:
        at=arguments.index('--stage')
        if at+1>=len(arguments) or arguments[at+1] not in ('analyze','prepare','train','predict'):
            raise SystemExit('--stage requires analyze, prepare, train or predict')
        stage=arguments[at+1]
        arguments=arguments[:at]+arguments[at+2:]
    else:
        stage='analyze'
    if stage=='predict':
        raise SystemExit(main(['predict','--smoke',*arguments]))
    if stage=='train':
        raise SystemExit(main(['train','--smoke',*arguments]))
    if stage=='prepare':
        # Uses the full source audit and full training-only fit. Only the output
        # cases are limited; the native/resampled sizes are never downscaled.
        # The partial method must exercise one complete AND one missing-MYO
        # case by default. Explicit --cases/--cohort remains the caller's choice.
        method='stunet-s'
        for index,argument in enumerate(arguments):
            if argument=='--method' and index+1<len(arguments):method=arguments[index+1]
            elif argument.startswith('--method='):method=argument.split('=',1)[1]
        explicit_selection=any(a in ('--cases','--cohort') or a.startswith('--cohort=')
                               or a.startswith('--cases=') for a in arguments)
        defaults=['--cases','ct_1001','ct_1003'] if method=='stunet-s-partial' and not explicit_selection else []
        raise SystemExit(main(['prepare','--method',method,'--fold','0','--limit','2',*defaults,*arguments]))
    raise SystemExit(main(["analyze","--limit","2","--components","none","--tag","smoke",*arguments]))
