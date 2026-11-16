import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter, sleep

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT/'tests'))

from verify_delivery_real import MetadataWorkspace
from chd_segmentation.config import DEFAULTS
from chd_segmentation.training.memory_guard import MemoryGuard, identity, alive


def request(guard):
    began = perf_counter()
    cache = [bytearray(1024**2)]
    with guard.admit(1024**2, label='waiting_next_block', release=cache.clear) as ticket:
        return dict(admitted=ticket is not None, seconds=perf_counter()-began,
                    actual_cache_released=not cache)


def waiting_child(config, owner, cancel, connection):
    config['memory_cancel'] = cancel
    with MemoryGuard(config, 'preprocess_case', owner=owner) as guard:
        connection.send(identity())
        config['resources']['memory_limit_bytes'] = connection.recv()
        with guard.admit(1024**2, label='cancel_waiting_block') as ticket:
            connection.send(dict(admitted=ticket is not None,
                                 reason=guard.boundary()['stop_reason']))
    connection.close()


def check_cancel(config, workspace):
    context = mp.get_context('spawn')
    cancel = context.Event()
    incoming, outgoing = context.Pipe()
    with MemoryGuard(config, 'preprocess') as guard:
        with guard.admit(16*1024**2, label='current_block', reserve=True):
            process = context.Process(target=waiting_child, args=(config, identity(), cancel, outgoing))
            process.start()
            outgoing.close()
            try:
                if not incoming.poll(30):
                    raise TimeoutError('本次等待worker未完成登记')
                member = incoming.recv()
                facts = guard.boundary()
                incoming.send(facts['project']['accounted_bytes']+8*1024**2)
                sleep(1.4)
                cancel.set()
                if not incoming.poll(15):
                    raise TimeoutError('取消后本次worker未退出等待')
                result = incoming.recv()
                process.join(15)
                assert process.exitcode == 0 and not alive(member)
                guard.boundary()
                sleep(.3)
                guard.boundary()
                with guard.admit(0, label='owner_after_worker_cancel', wait=False) as continued:
                    owner_continued = continued is not None
                workspace.results.append(dict(stage='cancel_waiting_worker', result=result,
                    worker_exited=True, owner_continued=owner_continued,
                    ct_voxels_read=0, optimizer_updates=0))
                assert not result['admitted'] and result['reason'] == 'cancelled'
                assert owner_continued
            finally:
                cancel.set()
                if process.is_alive():
                    process.terminate()
                    process.join(15)
                incoming.close()


def check_hysteresis(config, workspace):
    config['resources']['memory_resume_fraction'] = .02
    with MemoryGuard(config, 'preprocess') as guard, ThreadPoolExecutor(max_workers=1) as pool:
        with guard.admit(16*1024**2, label='hysteresis_block', reserve=True):
            config['resources']['memory_limit_bytes'] = guard.boundary()['project']['accounted_bytes']+8*1024**2
            future = pool.submit(request, guard)
            sleep(1.4)
            assert not future.done()
        try:
            result = future.result(timeout=4)
            assert result['admitted']
            workspace.results.append(dict(stage='small_cap_hysteresis', resumed=result, optimizer_updates=0))
        finally:
            config['resources']['memory_limit_bytes'] = 100_000_000_000


def check_supply_cancel(config, workspace):
    from chd_segmentation.models.supply import CpuReady, OrderedSupply
    context = mp.get_context('spawn')
    cancel = context.Event()
    ready = CpuReady(context, 2, Path(config['run_dir']), owner=identity(), cancel=cancel)
    supply = OrderedSupply(context, [1, 1], 1, ready, Path(config['run_dir']),
                           capacity_bytes=4*1024**2, estimates=[dict(case_id='ct_1003', workspace_bytes=4*1024**2)])
    supply.reserve(0, 'ct_1003')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(supply.reserve, 1, 'ct_1003')
        try:
            sleep(.3)
            assert not future.done()
            cancel.set()
            sleep(5.5)
            assert future.done(), '取消后病例空间等待仍未结束'
            assert isinstance(future.exception(), RuntimeError)
            workspace.results.append(dict(stage='supply_workspace_cancel', worker_exited=True, optimizer_updates=0))
        finally:
            with supply.condition:
                supply.reserved[:] = [0, 0]
                supply.condition.notify_all()


def check_blocks(config, workspace):
    from chd_segmentation.models.stunet.streaming import BlockWorkspace
    from chd_segmentation.observability.progress import Progress
    from chd_segmentation.artifacts.store import read_json
    config['logging']['heartbeat_seconds'] = 15
    with MemoryGuard(config, 'preprocess') as guard:
        progress = Progress(config, 'preprocess', total=1)
        progress.update(active=1, current_case='ct_1003')
        progress.snapshot(force=True)
        blocks = BlockWorkspace(workspace.root/'blocks', config, guard, 'ct_1003')
        try:
            blocks.report('read_decompress', 3, 7)
            path = Path(config['run_dir'])/'progress-preprocess.json'
            record = read_json(path)
            assert record['block_done'] == 3 and record['block_total'] == 7
            progress.snapshot(force=True)
            assert read_json(path)['block_done'] == 3
            progress.update(done=1, active=0, current_case=None)
            progress.close()
            assert read_json(path)['block_done'] is None
            workspace.results.append(dict(stage='structured_block_progress', done=3, total=7, optimizer_updates=0))
        finally:
            blocks.close()


def run(args):
    sys.path.insert(0, args.simulator)
    from common import clean_owned, files_under
    with MetadataWorkspace(args.work_root, args.report, clean_owned, files_under) as workspace:
        if args.only in ('hysteresis', 'supply', 'blocks'):
            config = deepcopy(DEFAULTS)
            config.update(scope='technical', work_root=str(workspace.root/'course'), run_dir=str(workspace.root/'course/preprocess'))
            config['resources'].update(ram_fraction=.95, reserve_ram_bytes=256*1024**2,
                                       memory_pressure_seconds=.2, memory_resume_seconds=.2)
            {'hysteresis': check_hysteresis, 'supply': check_supply_cancel, 'blocks': check_blocks}[args.only](config, workspace)
            return
        for name in (() if args.only == 'cancel' else ('preprocess', 'training_batch')):
            config = deepcopy(DEFAULTS)
            config.update(scope='technical', work_root=str(workspace.root/'course'),
                          run_dir=str(workspace.root/'course'/name))
            config['resources'].update(ram_fraction=.95, reserve_ram_bytes=256*1024**2,
                memory_pressure_seconds=.2, memory_resume_seconds=.2, memory_resume_fraction=.000001)
            with MemoryGuard(config, name) as guard, ThreadPoolExecutor(max_workers=1) as pool:
                with guard.admit(16*1024**2, label='actual_current_block', reserve=True) as ticket:
                    assert ticket is not None
                    payload = bytearray(b'\x01')*(4*1024**2)
                    before = guard.boundary()
                    config['resources']['memory_limit_bytes'] = before['project']['accounted_bytes']+8*1024**2
                    future = pool.submit(request, guard)
                    sleep(1.4)
                    pressure = guard.boundary()
                    blocked_before_release = not future.done()
                    del payload
                    gc.collect()
                resumed = future.result(timeout=12)
                after = guard.boundary()
                workspace.results.append(dict(stage=name, real_reservation_bytes=16*1024**2,
                    actual_payload_bytes=4*1024**2, pressure=pressure['paused'],
                    stop_requested=pressure['stop_requested'], blocked_before_release=blocked_before_release,
                    resumed=resumed, paused_after_release=after['paused'],
                    ct_voxels_read=0, optimizer_updates=0))
                assert pressure['paused'] and not pressure['stop_requested']
                assert blocked_before_release and resumed['admitted']
                assert resumed['actual_cache_released']
                assert not after['paused']
        if args.only in ('all', 'cancel'):
            config = deepcopy(DEFAULTS)
            config.update(scope='technical', work_root=str(workspace.root/'course'),
                          run_dir=str(workspace.root/'course/cancel'))
            config['resources'].update(ram_fraction=.95, reserve_ram_bytes=256*1024**2,
                memory_pressure_seconds=.2, memory_resume_seconds=.2, memory_resume_fraction=.000001)
            check_cancel(config, workspace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--simulator', default='D:/projects/test')
    parser.add_argument('--only', choices=('all', 'wait', 'cancel', 'hysteresis', 'supply', 'blocks'), default='all')
    run(parser.parse_args())
