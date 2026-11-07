import os
from pathlib import Path
import sys
import time

out = Path(sys.argv[1])
me, peer, mode = sys.argv[2:]
(out/(me+'.ready')).write_text(str(os.getpid()))
deadline = time.monotonic()+8
while not (out/(peer+'.ready')).exists() and time.monotonic()<deadline:
    time.sleep(.02)
if not (out/(peer+'.ready')).exists():
    sys.exit(91)
if mode == 'fail':
    sys.exit(7)
if mode == 'wait':
    time.sleep(40)
time.sleep(.25)
(out/(me+'.done')).write_text('ok')
