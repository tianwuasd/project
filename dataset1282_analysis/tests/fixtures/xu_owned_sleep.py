import argparse
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from dataset1282_analysis.observability.status import operation_lock

parser = argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--wait-resources',action='store_true')
args = parser.parse_args()

def stop(signum,frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM,stop)
with operation_lock(args.output,'xu-model-run',args.output):
    (args.output/'ready').touch()
    try:
        if args.wait_resources:
            from types import SimpleNamespace
            from dataset1282_analysis.xu_resources import stage_resources
            from dataset1282_analysis.resources import available_memory_bytes
            from dataset1282_analysis.xu_reallocation import run_with_reallocation
            options = SimpleNamespace(resume=False,resource_recovery=True,resource_poll_seconds=60)
            def require_resources(current):
                stage_resources(['cpu'],host_demand_bytes=2*available_memory_bytes())
            run_with_reallocation(require_resources,options,args.output)
        else:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
