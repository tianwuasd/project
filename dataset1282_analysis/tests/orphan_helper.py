"""Synthetic supervisor used only for hard-exit lifecycle tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataset1282_analysis.observability.status import operation_lock,register_workers

home=Path(sys.argv[1]);ready=Path(sys.argv[2])
with operation_lock(home,'synthetic-supervisor') as owner:
    code="import json,os,pathlib,sys,time;from dataset1282_analysis.engine.worker import start_owner_watchdog;start_owner_watchdog(sys.argv[1],json.loads(sys.argv[2]));pathlib.Path(sys.argv[3]).write_text(json.dumps({'pid':os.getpid()}));time.sleep(60)"
    child=subprocess.Popen([sys.executable,'-B','-c',code,str(home),json.dumps(owner),str(ready)])
    register_workers(home,owner,[child.pid])
    (home/'supervisor.json').write_text(json.dumps({'pid':os.getpid(),'worker_launcher':child.pid}))
    time.sleep(60)
