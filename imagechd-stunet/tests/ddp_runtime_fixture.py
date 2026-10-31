"""Importable CPU-only targets for the distributed launcher contract tests."""
import os
from pathlib import Path
import random
import time


def echo(rank, world_size, arguments):
    import torch
    print(f'rank {rank} stdout', flush=True)
    import sys
    print(f'rank {rank} stderr', file=sys.stderr, flush=True)
    assert not torch.cuda.is_initialized()
    return dict(rank=rank, world_size=world_size, argument=arguments,
                visible=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda=False)


def fail(rank, world_size, arguments):
    if rank == 0:
        raise ValueError('rank target exploded')
    time.sleep(60)


def sleep(rank, world_size, arguments):
    Path(arguments, f'{rank}.ready').write_text(str(os.getpid()))
    time.sleep(60)


def missing(rank, world_size, arguments):
    import ddp_runtime
    ddp_runtime._write_record = lambda *a: None
    return {}


def partial(rank, world_size, arguments):
    import ddp_runtime
    def write(path, value):
        Path(path).write_text('{"rank":', encoding='utf-8')
    ddp_runtime._write_record = write
    return {}


def mismatched(rank, world_size, arguments):
    import ddp_runtime
    original = ddp_runtime._write_record
    def write(path, value):
        value['invocation_id'] = 'stale-other-invocation'
        original(path, value)
    ddp_runtime._write_record = write
    return {}


def abrupt(rank, world_size, arguments):
    os._exit(9)


def inherited_lock(rank, world_size, arguments):
    fd = int(os.environ['WHOLE_HEART_LOCK_FD'])
    os.fstat(fd)
    directory = Path(arguments['directory'])
    (directory / f'{rank}.lock').write_text(str(fd))
    if arguments.get('wait'):
        deadline = time.monotonic() + 20
        while not (directory / 'release').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('lock test release marker missing')
            time.sleep(.02)
    return {'held': True}


def train_resume(rank, world_size, arguments):
    import copy
    import numpy as np
    import torch
    import torch.distributed as dist
    from ddp_runtime import capture_rank_state, restore_rank_state, gather_rank_states
    torch.set_num_threads(1)
    random.seed(100 + rank)
    np.random.seed(200 + rank)
    torch.manual_seed(300 + rank)
    assert not torch.cuda.is_initialized()
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.25, -.5]], dtype=torch.float64))
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.8)
    x = torch.tensor([[1., 2.], [3., -1.], [-2., 4.], [.5, -.5]], dtype=torch.float64)
    y = torch.tensor([[1.], [-1.], [2.], [.5]], dtype=torch.float64)
    local_x, local_y = x.chunk(world_size)[rank], y.chunk(world_size)[rank]
    def step():
        optimizer.zero_grad()
        loss = ((model(local_x) - local_y) ** 2).mean()
        loss.backward()
        gradient = model.module.weight.grad.detach().clone()
        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss)
        optimizer.step()
        return gradient.tolist(), float(reduced_loss / world_size)
    gradient, loss = step()
    first_weight = model.module.weight.detach().tolist()
    state = capture_rank_state()
    states = gather_rank_states(state)
    assert len(states) == world_size and 'cuda' not in state
    def draws():
        return [random.random(), float(np.random.random()), float(torch.rand(()))]
    expected_draw = draws()
    restore_rank_state(state)
    assert draws() == expected_draw
    restore_rank_state(state)
    checkpoint = Path(arguments) / 'toy-checkpoint.pt'
    if rank == 0:
        torch.save(dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(),
                        ranks=states), checkpoint)
    dist.barrier()
    expected_random = draws()
    second_gradient, second_loss = step()
    expected_weight = model.module.weight.detach().clone()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.module.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    restore_rank_state(saved['ranks'][rank])
    assert draws() == expected_random
    actual_gradient, actual_loss = step()
    assert torch.equal(expected_weight, model.module.weight)
    actual_optimizer = optimizer.state_dict()
    for key, value in expected_optimizer['state'][0].items():
        assert torch.equal(value, actual_optimizer['state'][0][key])
    assert second_gradient == actual_gradient and second_loss == actual_loss
    assert not torch.cuda.is_initialized()
    return dict(gradient=gradient, loss=loss, first_weight=first_weight,
                weight=expected_weight.tolist(), second_gradient=second_gradient,
                random=expected_random, cuda=False, restored=True)
