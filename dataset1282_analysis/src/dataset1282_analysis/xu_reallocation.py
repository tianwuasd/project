import copy
import time

from .contracts import atomic_json, read_json
from .observability.status import write_status
from .xu_resources import GpuResourcesChanged


def run_with_reallocation(operation, args, out):
    attempt = 0
    running = copy.copy(args)
    while True:
        try:
            return operation(running)
        except GpuResourcesChanged as error:
            attempt += 1
            record_reallocation(out, error, attempt, getattr(args,'max_reallocations',3))
            running.resume = True
            running.reallocate = True


def record_reallocation(out, error, attempt, maximum):
    if attempt > maximum:
        raise error
    unfinished = {}
    for file in sorted((out/'assignments').glob('*.json')):
        if (out/'receipts'/file.name).exists():
            continue
        stage = file.stem
        checkpoints = [str(p) for p in (out/stage).glob('*last.pt')]
        unfinished[stage] = dict(assignment=read_json(file), checkpoints=checkpoints)
    row = dict(attempt=attempt, maximum=maximum, changed_at=time.time(),
               **error.evidence(), unfinished=unfinished,
               recovery='restart from last complete epoch; replay incomplete work')
    atomic_json(out/'reallocations'/f'{attempt}-{time.time_ns()}.json', row)
    write_status(out, state='reallocating', stage='gpu-allocation', **row)
    print(f'GPU 资源变化，重新分配 {attempt}/{maximum}: {error}', flush=True)
