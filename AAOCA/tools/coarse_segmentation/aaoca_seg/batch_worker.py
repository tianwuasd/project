"""Dependency-free detached owner for a frozen terminal job."""
import os
import subprocess
from pathlib import Path

from .operations import python_path, runtime_env
from .service_state import (
    file_lock,
    process_identity,
    read_json,
    redact,
    timestamp,
    write_json,
)


def main(workspace, job_id):
    workspace = Path(workspace)
    job = workspace / 'jobs' / job_id
    state = read_json(job / 'state.json')
    write_json(job / 'owner.json', process_identity())
    try:
        with file_lock(workspace / 'operation.lock') as fd:
            env = runtime_env(workspace)
            fds = (fd,) if os.name != 'nt' else ()
            env['AAOCA_LOCK_FDS'] = ','.join(map(str, fds))
            state.update(status='running')
            write_json(job / 'state.json', state)
            with (job / 'queue.log').open('ab') as log:
                child = subprocess.Popen([str(python_path(workspace)), '-u', str(job / 'code' / 'queue_job.py'), str(job)],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=job / 'code', env=env, **({'pass_fds': fds} if os.name != 'nt' else {}))
                write_json(job / 'child.json', process_identity(child.pid))
                code = child.wait()
            state.update(status='completed' if code == 0 else ('stopped' if code == 130 else 'failed'), exit_code=code)
    except Exception as error:
        state.update(status='failed', error=redact(error))
    state['ended_at'] = timestamp()
    write_json(job / 'state.json', state)
    return 0 if state['status'] == 'completed' else 1
