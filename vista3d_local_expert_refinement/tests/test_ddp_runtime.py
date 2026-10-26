"""Bounded real spawn/Gloo tests; no real data or CUDA initialization."""
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddp_runtime import launch_ranks, capture_rank_state, restore_rank_state

HAS_TORCH = importlib.util.find_spec('torch') is not None


class ValidationTests(unittest.TestCase):
    def test_reject_invalid_inputs_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            for change in [dict(gpu_uuids=[]), dict(gpu_uuids=['a', 'a']),
                           dict(target='not-importable'), dict(backend='invalid'),
                           dict(timeout_seconds=0), dict(init_timeout_seconds=float('nan'))]:
                args = dict(target='ddp_runtime_fixture:echo', arguments={}, gpu_uuids=['a'],
                            log_dir=directory, backend='gloo')
                args.update(change)
                with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                    launch_ranks(**args)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_launcher_import_is_stdlib_only(self):
        import subprocess
        source = str(Path(__file__).resolve().parents[1] / '05_src')
        script = f'import sys;sys.path.insert(0,{source!r});import ddp_runtime;assert "torch" not in sys.modules'
        subprocess.run([sys.executable, '-S', '-c', script], check=True, timeout=10)

    def test_lock_wrapper_cannot_create_resource_sharer_token_outside_spawn(self):
        from ddp_runtime import _SpawnLock
        with patch('ddp_runtime.reduction.DupFd', create=True) as duplicate:
            wrapped = _SpawnLock(123)
            duplicate.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'active spawn'):
                pickle.dumps(wrapped)
            duplicate.assert_not_called()

    def test_unpicklable_arguments_fail_before_any_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((AttributeError, pickle.PicklingError)):
                launch_ranks('ddp_runtime_fixture:echo', lambda: None, ['CPU'], directory, backend='gloo')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_log_path_does_not_duplicate_the_borrowed_lock(self):
        with patch('ddp_runtime.os.dup', return_value=123) as duplicate:
            with self.assertRaises(TypeError):
                launch_ranks('ddp_runtime_fixture:echo', {}, ['CPU'], None,
                             backend='gloo', lock_fd=5)
            duplicate.assert_not_called()


@unittest.skipUnless(HAS_TORCH, 'requires CPU Torch/Gloo; no dependency installed by test')
class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.before = {p.pid for p in mp.active_children()}
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def tearDown(self):
        self.assertEqual({p.pid for p in mp.active_children()}, self.before)
        for sig, handler in self.handlers.items():
            self.assertEqual(signal.getsignal(sig), handler)

    def launch(self, target='echo', arguments=None, **kwargs):
        options = dict(backend='gloo', timeout_seconds=40, init_timeout_seconds=20)
        options.update(kwargs)
        return launch_ranks(f'ddp_runtime_fixture:{target}', arguments, ['CPU-A', 'CPU-B'],
                            self.directory / 'logs', **options)

    def test_fresh_invocations_logs_map_registry_and_parent_cuda(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        job = self.directory / 'job'
        with patch.dict(os.environ, WHOLE_HEART_JOB=str(job)):
            first = self.launch(arguments={'token': 7})
            second = self.launch(arguments={'token': 8})
        self.assertNotEqual(first[0]['invocation_id'], second[0]['invocation_id'])
        for group in (first, second):
            for rank, result in enumerate(group):
                self.assertEqual(result['rank'], rank)
                self.assertEqual(result['gpu_uuid'], ['CPU-A', 'CPU-B'][rank])
                self.assertEqual(result['result']['rank'], rank)
                log = Path(result['log_dir'])
                self.assertIn(f'rank {rank} stdout', (log / 'stdout.log').read_text())
                self.assertIn(f'rank {rank} stderr', (log / 'stderr.log').read_text())
                self.assertEqual(json.loads((log / 'result.json').read_text()), result)
        registered = [json.loads(p.read_text()) for p in (job / 'processes').glob('*.json')]
        self.assertEqual({r['pid'] for r in registered}, {r['pid'] for r in first + second})
        self.assertTrue(all(r['role'] == 'ddp-rank' for r in registered))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_gradient_loss_optimizer_and_all_rank_rng_resume(self):
        import torch
        results = self.launch('train_resume', str(self.directory))
        model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[.25, -.5]], dtype=torch.float64))
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.8)
        x = torch.tensor([[1., 2.], [3., -1.], [-2., 4.], [.5, -.5]], dtype=torch.float64)
        y = torch.tensor([[1.], [-1.], [2.], [.5]], dtype=torch.float64)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        gradient = model.weight.grad.detach().clone()
        optimizer.step()
        first = model.weight.detach().clone()
        optimizer.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        optimizer.step()
        for result in results:
            value = result['result']
            torch.testing.assert_close(torch.tensor(value['gradient'], dtype=torch.float64), gradient)
            torch.testing.assert_close(torch.tensor(value['first_weight'], dtype=torch.float64), first)
            torch.testing.assert_close(torch.tensor(value['weight'], dtype=torch.float64), model.weight)
            self.assertAlmostEqual(value['loss'], float(loss.detach()))
            self.assertTrue(value['restored'])
        self.assertNotEqual(results[0]['result']['random'], results[1]['result']['random'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_failure_and_abrupt_exit_clean_siblings(self):
        for target in ('fail', 'abrupt'):
            with self.subTest(target=target), self.assertRaises(RuntimeError):
                self.launch(target)
        traces = list((self.directory / 'logs').glob('*/rank-0/traceback.txt'))
        self.assertTrue(any('rank target exploded' in p.read_text() for p in traces))

    def test_missing_partial_and_stale_success_are_rejected(self):
        for target in ('missing', 'partial', 'mismatched'):
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'result'):
                self.launch(target)

    def test_timeout_cleans_all_ranks(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            self.launch('sleep', str(self.directory), timeout_seconds=8)
        self.assertEqual(len(list(self.directory.glob('*.ready'))), 2)
        self.assertLess(time.monotonic() - started, 15)

    def test_signal_cancellation_cleans_all_ranks(self):
        finished = threading.Event()
        def cancel_after_ready():
            deadline = time.monotonic() + 30
            while not finished.wait(.05):
                if list(self.directory.glob('*.ready')) or time.monotonic() > deadline:
                    signal.raise_signal(signal.SIGINT)
                    return
        thread = threading.Thread(target=cancel_after_ready, daemon=True)
        thread.start()
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.launch('sleep', str(self.directory))
        finally:
            finished.set()
            thread.join(timeout=2)

    def test_second_spawn_failure_cleans_first_rank(self):
        original = mp.context.SpawnProcess.start
        calls = []
        def start(process):
            calls.append(process)
            if len(calls) == 2:
                raise OSError('injected spawn failure')
            return original(process)
        with patch.object(mp.context.SpawnProcess, 'start', start):
            with self.assertRaisesRegex(OSError, 'injected spawn failure'):
                self.launch()

    @unittest.skipUnless(os.name == 'nt', 'Windows creates OS child before reducing arguments')
    def test_windows_reduction_failure_reaps_already_created_os_child(self):
        import ddp_runtime
        from multiprocessing.context import get_spawning_popen
        original = ddp_runtime._SpawnLock.__reduce__
        actual_pids = []
        def broken_reduce(wrapped):
            actual_pids.append(get_spawning_popen().pid)
            original(wrapped)  # A handle was already duplicated into this child.
            raise OSError('injected post-create reduction failure')
        job = self.directory / 'job'
        with (self.directory / 'lock').open('w') as stream, \
                patch.dict(os.environ, WHOLE_HEART_JOB=str(job)), \
                patch.object(ddp_runtime._SpawnLock, '__reduce__', broken_reduce):
            with self.assertRaisesRegex(OSError, 'post-create reduction failure'):
                self.launch(lock_fd=stream.fileno())
            os.fstat(stream.fileno())
        self.assertEqual(len(actual_pids), 1)
        registered = [json.loads(p.read_text()) for p in (job / 'processes').glob('*.json')]
        self.assertEqual([r['pid'] for r in registered], actual_pids)
        # This process never reached mp._children; check the actual OS handle.
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, actual_pids[0])
        if handle:
            try:
                exit_code = ctypes.c_ulong()
                self.assertTrue(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(exit_code)))
                self.assertNotEqual(exit_code.value, 259)  # STILL_ACTIVE
            finally:
                kernel.CloseHandle(ctypes.c_void_p(handle))

    def test_cpu_rng_roundtrip_without_cuda(self):
        import random
        import numpy as np
        import torch
        state = capture_rank_state()
        expected = [random.random(), float(np.random.random()), torch.rand(3)]
        restore_rank_state(state)
        actual = [random.random(), float(np.random.random()), torch.rand(3)]
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertNotIn('cuda', state)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cuda_rng_api_only_uses_requested_current_rank(self):
        import torch
        marker = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with patch.object(torch.cuda, 'current_device', return_value=1), \
                patch.object(torch.cuda, 'get_rng_state', return_value=marker) as get_state, \
                patch.object(torch.cuda, 'set_rng_state') as set_state, \
                patch.object(torch.cuda, 'get_rng_state_all', side_effect=AssertionError('all GPUs')), \
                patch.object(torch.cuda, 'set_rng_state_all', side_effect=AssertionError('all GPUs')):
            state = capture_rank_state(device_rank=1)
            restore_rank_state(state, device_rank=1)
            get_state.assert_called_once_with(1)
            self.assertEqual(set_state.call_args.args[1], 1)
            self.assertTrue(torch.equal(set_state.call_args.args[0], marker))
            with self.assertRaises(ValueError):
                capture_rank_state(device_rank=0)
            with self.assertRaises(ValueError):
                restore_rank_state(state, device_rank=0)
            with self.assertRaises(ValueError):
                restore_rank_state(state)
        self.assertFalse(torch.cuda.is_initialized())

    def test_gather_requires_initialized_group(self):
        from ddp_runtime import gather_rank_states
        with self.assertRaisesRegex(RuntimeError, 'initialized group'):
            gather_rank_states({})

    def test_explicit_and_environment_lock_fds_reach_windows_or_posix_children(self):
        path = self.directory / 'borrowed-descriptor'
        with path.open('w') as stream:
            self.launch('inherited_lock', dict(directory=str(self.directory)), lock_fd=stream.fileno())
            os.fstat(stream.fileno())
            with patch.dict(os.environ, WHOLE_HEART_LOCK_FD=str(stream.fileno())):
                self.launch('inherited_lock', dict(directory=str(self.directory)))
            os.fstat(stream.fileno())
        self.assertEqual(len(list(self.directory.glob('*.lock'))), 2)

    @unittest.skipUnless(os.name == 'posix', 'POSIX flock/DupFd needs Linux target acceptance')
    def test_inherited_flock_survives_parent_fd_close_and_failed_spawn(self):
        import fcntl
        path = self.directory / 'lease.lock'
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed, errors = [], []
        def close_and_check():
            deadline = time.monotonic() + 25
            while len(list(self.directory.glob('[01].lock'))) != 2:
                if time.monotonic() > deadline:
                    errors.append('ranks not ready')
                    return
                time.sleep(.02)
            os.close(fd)
            with path.open('w') as check:
                try:
                    fcntl.flock(check.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    observed.append('held')
            (self.directory / 'release').touch()
        thread = threading.Thread(target=close_and_check, daemon=True)
        thread.start()
        self.launch('inherited_lock', dict(directory=str(self.directory), wait=True), lock_fd=fd)
        thread.join(timeout=2)
        self.assertEqual(errors, [])
        self.assertEqual(observed, ['held'])
        with path.open('w') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Fail after spawn has serialized _SpawnLock, precisely where a
            # precreated resource_sharer.DupFd would leave a persistent flock.
            with patch('multiprocessing.util.spawnv_passfds', side_effect=OSError('failed')):
                with self.assertRaises(OSError):
                    self.launch(lock_fd=stream.fileno())
        with path.open('w') as check:
            fcntl.flock(check.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(len(list(self.directory.glob('*.lock'))), 3)


if __name__ == '__main__':
    unittest.main()
