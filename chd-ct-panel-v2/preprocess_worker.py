"""Owned CPU worker; called by the supervisor, never a separate pipeline."""
from pathlib import Path
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main():
    from chd_stunet.observability.runlog import write_json
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    started = time.monotonic()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            from chd_stunet.preprocessing.registry import get_adapter
            adapter=get_adapter(job['method']['name'])
            result = (adapter.audit_case if job['stage']=='audit' else adapter.preprocess_case)(job)
    except BaseException as error:
        traceback.print_exc()
        result = dict(success=False,case_id=job['pair']['case_id'],error=str(error),
                      traceback=traceback.format_exc(),worker_pid=os.getpid(),
                      elapsed_seconds=time.monotonic()-started)
    write_json(job['result'],result)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
