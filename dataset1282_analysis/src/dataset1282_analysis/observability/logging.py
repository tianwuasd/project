from datetime import datetime, timezone
import json
from pathlib import Path


def event(directory,name,**fields):
    row={'time':datetime.now(timezone.utc).isoformat(),'event':name,**fields}
    with (Path(directory)/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=str)+'\n')
    with (Path(directory)/'run.log').open('a',encoding='utf-8') as f:
        f.write(f"{row['time']} {name} " + json.dumps(fields,ensure_ascii=False,default=str)+'\n')
