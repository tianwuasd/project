import errno
import os
from pathlib import Path
import sys
import time
import traceback

import psutil
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from chd_stunet.observability.runlog import write_json
from chd_medsam2.contracts import read_json
from chd_medsam2.preprocessing import prepare_case


def main(task_path):
    task = read_json(task_path)
    pid = os.getpid()
    born = psutil.Process(pid).create_time()
    result = Path(task['result'])
    retryable = {errno.ECONNRESET, errno.ETIMEDOUT, errno.ENETDOWN, errno.ENETUNREACH,
                 errno.EHOSTUNREACH, errno.ESTALE}

    def phase(name, **detail):
        write_json(result.with_suffix('.progress.json'),
                   dict(worker_pid=pid, worker_born=born, stage=name, time=time.time(), event='phase', **detail))

    with threadpool_limits(limits=task['cpu_threads']):
        for attempt, delay in enumerate((2, 5, 15, 30, 60, None), 1):
            try:
                marker, reused = prepare_case(Path(task['folder']), task['row'], Path(task['gate']), phase)
            except (OSError, ValueError) as error:
                if isinstance(error, OSError) and error.errno in retryable and delay is not None:
                    phase('waiting_io', attempt=attempt, next_retry_at=time.time()+delay, error=str(error))
                    print(f'病例 {task["row"]["case_id"]} 的存储暂时不可用，{delay} 秒后重试', flush=True)
                    time.sleep(delay)
                    continue
                traceback.print_exc()
                write_json(result, dict(success=False, worker_pid=pid, error=str(error)))
                return 1
            write_json(result, dict(success=True, worker_pid=pid, reused=reused, identity=marker['identity']))
            return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
