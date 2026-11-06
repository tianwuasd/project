"""Readable log, ordered events, and atomic status written by the supervisor."""
import json
import os
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone

CHINA = timezone(timedelta(hours=8))


def write_json(path, value):
    """Commit a complete JSON file atomically; no NaN/Inf extension."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    # Windows readers/antivirus can briefly open the old file without delete
    # sharing. Keep the old complete JSON visible and retry only documented
    # sharing/access errors, for <=0.32s. Persistent permissions still fail;
    # never truncate/delete the destination to make a replacement succeed.
    for attempt in range(7):
        try:
            os.replace(temporary,path)
            break
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==6:raise
            time.sleep(.005*2**attempt)


class RunLog:
    def __init__(self, output):
        self.output=Path(output); self.started=time.perf_counter(); self.seq=0
        (self.output/"logs").mkdir(parents=True,exist_ok=True)

    def event(self, event, message, **fields):
        self.seq+=1
        now=datetime.now(CHINA).isoformat(timespec="seconds")
        record=dict(seq=self.seq,timestamp=now,elapsed_seconds=time.perf_counter()-self.started,
                    event=event,message=message,**fields)
        with (self.output/"logs/events.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        line=f"[{now}] {message}"
        with (self.output/"logs/run.log").open("a",encoding="utf-8") as stream:
            stream.write(line+"\n")
        print(line,flush=True)

    def status(self, state, **fields):
        write_json(self.output/"status.json",dict(state=state,pid=os.getpid(),
            heartbeat=datetime.now(CHINA).isoformat(timespec="seconds"),
            elapsed_seconds=time.perf_counter()-self.started,**fields))
