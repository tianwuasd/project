"""Real spawn tests; all batches are inert CPU data."""
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
from batch_pool import EpochBatchIterator, deterministic_seed


class RandomLoader:
    def __iter__(self):
        return self

    def __next__(self):
        return {'sample': random.random()}


class PrivateLoader(RandomLoader):
    def reseed(self, seed):
        self.rng = random.Random(seed)

    def __next__(self):
        return self.rng.random()


class ResetHookLoader(PrivateLoader):
    reseed = None

    def reset_rng(self, seed):
        self.rng = random.Random(seed)


class SlowFirstLoader(RandomLoader):
    def __init__(self, marker_dir, first_seed, await_seed=None):
        self.marker_dir, self.first_seed = marker_dir, first_seed
        self.await_seed=await_seed

    def reseed(self, seed):
        self.seed = seed

    def __next__(self):
        if self.seed == self.first_seed:
            deadline = time.monotonic() + 10
            while not (Path(self.marker_dir,str(self.await_seed)).exists() if self.await_seed is not None
                       else list(Path(self.marker_dir).iterdir())):
                if time.monotonic() > deadline:
                    raise RuntimeError('second worker never finished')
                time.sleep(.01)
        value = super().__next__()
        Path(self.marker_dir, str(self.seed)).write_text(str(time.perf_counter_ns()))
        return value


class RecordedLoader(RandomLoader):
    def __init__(self, marker_dir):
        self.marker_dir = marker_dir

    def reseed(self, seed):
        self.seed = seed

    def __next__(self):
        Path(self.marker_dir, str(self.seed)).write_text(str(os.getpid()))
        return super().__next__()


class BrokenLoader(RandomLoader):
    def __next__(self):
        raise ValueError('toy loader exploded')


class ExitLoader(RandomLoader):
    def __next__(self):
        os._exit(7)


class WarmThenSleepLoader(RandomLoader):
    def __init__(self, warm_seeds):
        self.warm_seeds = warm_seeds

    def reseed(self, seed):
        self.seed = seed

    def __next__(self):
        if self.seed not in self.warm_seeds:
            time.sleep(30)
        return super().__next__()


class EmptyLoader(RandomLoader):
    def __next__(self):
        raise StopIteration


class UnpicklableLoader(RandomLoader):
    def __next__(self):
        return lambda: None


class UnpicklableInputLoader(RandomLoader):
    def __init__(self):
        self.not_picklable = lambda: None


class TensorLoader(RandomLoader):
    def __next__(self):
        import numpy as np
        import torch
        if torch.cuda.is_initialized():
            raise AssertionError('CUDA initialized')
        return {'data': torch.rand((2, 3)), 'array': np.random.random((2, 3))}


class VariableTensorLoader(TensorLoader):
    def __init__(self): self.calls=0
    def __next__(self):
        import torch
        self.calls+=1
        return torch.ones(4 if self.calls==1 else 1024)


class StructuredArrayLoader(RandomLoader):
    def __next__(self):
        import numpy as np
        return np.array([(3,1.25),(7,2.5)],dtype=[('label','<i4'),('value','<f4')])


class DefaultDictArrayLoader(RandomLoader):
    def __next__(self):
        from collections import defaultdict
        import numpy as np
        return defaultdict(int,data=np.array([1,2,3],dtype=np.int32))


class PinBatch:
    def pin_memory(self):
        self.pinned_pid = os.getpid()
        return self


class PinLoader(RandomLoader):
    def __next__(self):
        return {'data': [PinBatch()]}


class OptionalRNGLoader(RandomLoader):
    def __next__(self):
        value = [random.random()]
        if importlib.util.find_spec('numpy'):
            import numpy as np
            value.append(float(np.random.random()))
        if importlib.util.find_spec('torch'):
            import torch
            if torch.cuda.is_initialized():
                raise AssertionError('worker initialized CUDA')
            value.append(float(torch.rand(())))
        return value


class LockLoader(RandomLoader):
    def __init__(self, marker_dir):
        self.marker_dir = marker_dir

    def __next__(self):
        fd = int(os.environ['WHOLE_HEART_LOCK_FD'])
        stat = os.fstat(fd)
        Path(self.marker_dir, str(os.getpid())).write_text(json.dumps({
            'inode': stat.st_ino, 'pgid': os.getpgrp(), 'fd': fd}))
        return stat.st_ino


def delayed_lock_worker(*_args):
    """Stay before _worker_main's detach until parent registration fails."""
    Path(os.environ['BATCH_POOL_BOOTSTRAP_MARKER']).write_text(str(os.getpid()))
    time.sleep(30)


class BatchPoolTests(unittest.TestCase):
    def collect(self, loader, count=8, **settings):
        iterator = EpochBatchIterator(loader, timeout_seconds=20, **settings)
        try:
            return [next(iterator) for _ in range(count)]
        finally:
            iterator.close()

    def assert_no_new_children(self, before):
        remaining = {p.pid for p in mp.active_children()} - before
        self.assertEqual(remaining, set())

    def test_seed_stable_and_coordinates_distinct(self):
        seed = deterministic_seed(13, 4, 1, 'train', 9)
        self.assertEqual(seed, 6127380649356115927)  # Checkpoint-compatible seed version 1.
        self.assertEqual(seed, deterministic_seed(13, 4, 1, 'train', 9))
        coordinates = [(14, 4, 1, 'train', 9), (13, 5, 1, 'train', 9),
                       (13, 4, 2, 'train', 9), (13, 4, 1, 'val', 9),
                       (13, 4, 1, 'train', 10)]
        self.assertTrue(all(seed != deterministic_seed(*c) for c in coordinates))

    def test_real_spawn_matches_serial_and_repeats(self):
        expected = self.collect(OptionalRNGLoader(), workers=0, seed=31, epoch=2)
        self.assertEqual(expected, self.collect(OptionalRNGLoader(), workers=2, seed=31, epoch=2))
        self.assertEqual(expected, self.collect(OptionalRNGLoader(), workers=2, seed=31, epoch=2))

    def test_private_rng_explicit_hooks(self):
        for loader in (PrivateLoader, ResetHookLoader):
            with self.subTest(loader=loader.__name__):
                self.assertEqual(self.collect(loader(), workers=0), self.collect(loader(), workers=2))

    def test_completes_out_of_order_but_yields_in_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            first_seed = deterministic_seed(7, 0, 0, 'train', 0)
            second_seed = deterministic_seed(7, 0, 0, 'train', 1)
            actual = self.collect(SlowFirstLoader(temp, first_seed), count=4, workers=2, seed=7)
            self.assertEqual(actual, self.collect(RandomLoader(), count=4, workers=0, seed=7))
            self.assertLess(int(Path(temp, str(second_seed)).read_text()),
                            int(Path(temp, str(first_seed)).read_text()))

    def test_finite_prefetch_during_consumer_pause(self):
        with tempfile.TemporaryDirectory() as temp:
            iterator = EpochBatchIterator(RecordedLoader(temp), workers=2, prefetch=3, timeout_seconds=20)
            try:
                next(iterator)
                time.sleep(.4)
                self.assertLessEqual(len(list(Path(temp).iterdir())), 4)  # one consumed plus <=3 outstanding
                self.assertEqual(iterator.state_dict()['next_index'], 1)
            finally:
                iterator.close()

    def test_total_prefetch_slots_keep_fast_worker_busy_behind_slow_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            first=deterministic_seed(0,0,0,'train',0)
            fourth=deterministic_seed(0,0,0,'train',3)
            iterator=EpochBatchIterator(SlowFirstLoader(temp,first,fourth),workers=2,prefetch=4,phase_steps=4,timeout_seconds=20)
            try:
                next(iterator)
                self.assertEqual(len(list(Path(temp).iterdir())),4)
                self.assertEqual(iterator.observations()['queue_depth_peak'],4)
                self.assertLess(int(Path(temp,str(fourth)).read_text()),int(Path(temp,str(first)).read_text()))
            finally:iterator.close()

    def test_exact_resume_ignores_prefetched_work_and_worker_count(self):
        iterator = EpochBatchIterator(PrivateLoader(), workers=2, seed=99, epoch=3,
                                      rank=1, stage='validation', prefetch=4, timeout_seconds=20)
        try:
            prefix = [next(iterator) for _ in range(3)]
            state = iterator.state_dict()
            expected_tail = [next(iterator) for _ in range(5)]
        finally:
            iterator.close()
        resumed = EpochBatchIterator(PrivateLoader(), workers=0)
        try:
            resumed.load_state_dict(state)
            self.assertEqual(state['next_index'], 3)
            self.assertEqual(expected_tail, [next(resumed) for _ in range(5)])
            resumed.set_epoch(3)
            self.assertEqual(prefix, [next(resumed) for _ in range(3)])
            resumed.set_epoch(3, start_index=3)
            self.assertEqual(next(resumed), expected_tail[0])
            resumed.set_epoch(4)
            self.assertNotEqual(next(resumed), prefix[0])
        finally:
            resumed.close()

    def test_active_epoch_reset_preserves_spawn_pids_and_sequence(self):
        before = {p.pid for p in mp.active_children()}
        iterator = EpochBatchIterator(RandomLoader(), workers=2, timeout_seconds=20)
        try:
            expected = next(iterator)
            old_workers = list(iterator._processes)
            iterator.set_epoch(0)
            self.assertTrue(all(process.is_alive() for process in old_workers))
            self.assertEqual(next(iterator), expected)
            self.assertEqual(iterator._processes, old_workers)
            state = iterator.state_dict()
            next_expected = next(iterator)
            iterator.load_state_dict(state)
            self.assertEqual(next(iterator), next_expected)
        finally:
            iterator.close()
        self.assert_no_new_children(before)

    def test_phase_bound_pause_and_replay_preserve_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            iterator = EpochBatchIterator(RecordedLoader(temp), workers=2, prefetch=8,
                                          phase_steps=3, worker_threads=2, timeout_seconds=20)
            try:
                first = next(iterator)
                workers = list(iterator._processes)
                state = iterator.state_dict()
                iterator.pause()
                count = len(list(Path(temp).iterdir()))
                time.sleep(.15)
                self.assertEqual(len(list(Path(temp).iterdir())), count)
                self.assertEqual(iterator.observations()['pending_batches'], 0)
                self.assertEqual(iterator.observations()['ready_batches'], 0)
                self.assertEqual(iterator.state_dict(), state)
                tail = [next(iterator) for _ in range(2)]
                with self.assertRaises(StopIteration): next(iterator)
                self.assertLessEqual(len(list(Path(temp).iterdir())), 3)
                iterator.load_state_dict(state)
                self.assertEqual([next(iterator) for _ in range(2)], tail)
                iterator.set_epoch(0)
                self.assertEqual(next(iterator), first)
                self.assertEqual(iterator._processes, workers)
                self.assertTrue(all(p.is_alive() for p in workers))
            finally: iterator.close()

    def test_outer_prepared_cursor_rewind_matches_consumed_checkpoint(self):
        iterator = EpochBatchIterator(PrivateLoader(), workers=2, prefetch=4, timeout_seconds=20)
        try:
            expected = [next(iterator) for _ in range(5)]
            pids = iterator.observations()['worker_pids']
            iterator.rewind(2)  # outer queue prepared 5 but the model consumed only 2
            self.assertEqual(iterator.state_dict()['next_index'],2)
            self.assertEqual([next(iterator) for _ in range(3)],expected[2:])
            self.assertEqual(iterator.observations()['worker_pids'],pids)
            with self.assertRaises(ValueError):iterator.rewind(99)
        finally:iterator.close()

    def test_phase_limit_change_is_explicit_and_cannot_skip_consumed_work(self):
        iterator=EpochBatchIterator(PrivateLoader(),phase_steps=5)
        try:
            next(iterator)
            with self.assertRaises(ValueError):iterator.set_phase_steps(0)
            iterator.set_phase_steps(2)
            next(iterator)
            with self.assertRaises(StopIteration):next(iterator)
            iterator.set_epoch(1)
            iterator.set_phase_steps(1)
            next(iterator)
            with self.assertRaises(StopIteration):next(iterator)
        finally:iterator.close()

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'tensor runtime')
    def test_reusable_shared_transport_and_bounded_pipe_fallback(self):
        import torch
        for budget in (0, 4096):
            with self.subTest(budget=budget):
                iterator = EpochBatchIterator(TensorLoader(), workers=1, prefetch=2,
                    worker_threads=2, max_shared_bytes=budget, timeout_seconds=30)
                try:
                    first = next(iterator); saved = first['data'].clone()
                    next(iterator); next(iterator)
                    iterator.pause()
                    obs = iterator.observations()
                    self.assertTrue(torch.equal(first['data'], saved))
                    self.assertEqual(obs['workers'][0]['torch_threads'], 2)
                    self.assertLessEqual(obs['shared_allocated_bytes'], budget)
                    if budget:
                        self.assertGreater(obs['shared_batches'], 1)
                        self.assertEqual(obs['pipe_batches'], 0)
                        self.assertGreater(obs['shared_reuses'], 0)
                    else:
                        self.assertGreater(obs['pipe_batches'], 1)
                        self.assertIn('shared_budget', obs['fallback_reasons'])
                finally: iterator.close()
                self.assertEqual(iterator.observations()['shared_allocated_bytes'], 0)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'tensor runtime')
    def test_shared_allocation_failure_and_larger_batch_have_explicit_pipe_fallback(self):
        from multiprocessing import shared_memory
        with patch.object(shared_memory,'SharedMemory',side_effect=OSError('injected allocation fault')):
            iterator=EpochBatchIterator(TensorLoader(),workers=1,prefetch=1,phase_steps=1,timeout_seconds=30)
            try:
                next(iterator);iterator.pause()
                self.assertEqual(iterator.observations()['pipe_batches'],1)
                self.assertTrue(any('injected allocation fault' in reason for reason in iterator.observations()['fallback_reasons']))
                self.assertEqual(iterator.observations()['shared_allocated_bytes'],0)
            finally:iterator.close()
        iterator=EpochBatchIterator(VariableTensorLoader(),workers=1,prefetch=1,phase_steps=2,timeout_seconds=30)
        try:
            first=next(iterator);second=next(iterator);iterator.pause()
            self.assertEqual((len(first),len(second)),(4,1024))
            self.assertTrue((first==1).all())
            self.assertIn('shared_slot_capacity',iterator.observations()['fallback_reasons'])
            self.assertEqual(iterator.observations()['shared_allocated_bytes'],16)
        finally:iterator.close()

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy runtime')
    def test_structured_numpy_fields_and_values_survive_transport(self):
        import numpy as np
        for budget in (4096,0):
            iterator=EpochBatchIterator(StructuredArrayLoader(),workers=1,prefetch=1,
                phase_steps=1,max_shared_bytes=budget,timeout_seconds=30)
            try:
                batch=next(iterator)
                self.assertEqual(batch.dtype.names,('label','value'))
                np.testing.assert_array_equal(batch['label'],[3,7])
                np.testing.assert_array_equal(batch['value'],[1.25,2.5])
                self.assertEqual(iterator.observations()['pipe_batches'],1)
                self.assertIn('structured_or_subarray_dtype',iterator.observations()['fallback_reasons'])
            finally:iterator.close()

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy runtime')
    def test_defaultdict_factory_and_missing_key_survive_transport(self):
        from collections import defaultdict
        import numpy as np
        for budget in (4096,0):
            iterator=EpochBatchIterator(DefaultDictArrayLoader(),workers=1,prefetch=1,
                phase_steps=1,max_shared_bytes=budget,timeout_seconds=30)
            try:
                batch=next(iterator)
                self.assertIsInstance(batch,defaultdict)
                self.assertIs(batch.default_factory,int)
                self.assertEqual(batch['missing'],0)
                np.testing.assert_array_equal(batch['data'],[1,2,3])
                self.assertEqual(iterator.observations()['pipe_batches'],1)
                self.assertIn('unsupported_container_subclass',iterator.observations()['fallback_reasons'])
            finally:iterator.close()

    def test_pause_timeout_closes_all_owned_workers(self):
        warm=[deterministic_seed(0,0,0,'train',0)]
        iterator=EpochBatchIterator(WarmThenSleepLoader(warm),workers=1,prefetch=1,timeout_seconds=20)
        next(iterator)
        iterator.timeout_seconds=.2
        with self.assertRaisesRegex(TimeoutError,'sequence=1'):iterator.pause()
        self.assertTrue(iterator.observations()['closed'])
        self.assertEqual(iterator.observations()['worker_pids'],[])

    def test_completed_prefetch_does_not_timeout_when_consumer_pauses(self):
        iterator = EpochBatchIterator(RandomLoader(), workers=1, timeout_seconds=20)
        try:
            next(iterator)
            # Initial optional-library imports are already complete. The next
            # batch finishes promptly; only the consumer pauses past its limit.
            iterator.timeout_seconds = .3
            time.sleep(.5)
            next(iterator)
        finally:
            iterator.close()

    def test_serial_preserves_parent_rng_streams(self):
        random.seed(18)
        before = random.getstate()
        numpy_state = torch_state = None
        if importlib.util.find_spec('numpy'):
            import numpy as np
            numpy_state = np.random.get_state()
        if importlib.util.find_spec('torch'):
            import torch
            torch_state = torch.get_rng_state().clone()
        self.collect(OptionalRNGLoader(), workers=0)
        self.assertEqual(before, random.getstate())
        if numpy_state is not None:
            after = np.random.get_state()
            self.assertEqual(numpy_state[0], after[0])
            self.assertTrue((numpy_state[1] == after[1]).all())
            self.assertEqual(numpy_state[2:], after[2:])
        if torch_state is not None:
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_serial_exception_also_preserves_parent_rng(self):
        before = random.getstate()
        iterator = EpochBatchIterator(BrokenLoader())
        with self.assertRaisesRegex(ValueError, 'toy loader exploded'):
            next(iterator)
        self.assertEqual(before, random.getstate())

    def test_failure_has_child_traceback_and_cleans_all_children(self):
        before = {p.pid for p in mp.active_children()}
        iterator = EpochBatchIterator(BrokenLoader(), workers=2, stage='validation', timeout_seconds=20)
        with self.assertRaisesRegex(RuntimeError, r'(?s)pid=.*validation.*sequence=0.*Traceback.*toy loader exploded'):
            next(iterator)
        iterator.close()
        self.assert_no_new_children(before)

    def test_abrupt_exit_is_reported_and_children_cleaned(self):
        before = {p.pid for p in mp.active_children()}
        iterator = EpochBatchIterator(ExitLoader(), workers=2, timeout_seconds=20)
        with self.assertRaisesRegex(RuntimeError, 'pid=.*sequence='):
            next(iterator)
        self.assert_no_new_children(before)

    def test_timeout_is_bounded_and_cleans_all_children(self):
        before = {p.pid for p in mp.active_children()}
        warm = [deterministic_seed(0, 0, 0, 'train', index) for index in (0, 1)]
        iterator = EpochBatchIterator(WarmThenSleepLoader(warm), workers=2, timeout_seconds=20)
        try:
            next(iterator)
            next(iterator)
            iterator.timeout_seconds = .5
            start = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, 'pid=.*train.*sequence=2'):
                next(iterator)
            self.assertLess(time.monotonic() - start, 6)
        finally:
            iterator.close()
        self.assert_no_new_children(before)

    def test_unpicklable_result_has_useful_failure(self):
        iterator = EpochBatchIterator(UnpicklableLoader(), workers=1, timeout_seconds=20)
        with self.assertRaisesRegex(RuntimeError, r'(?s)sequence=0.*Traceback.*pickle'):
            next(iterator)

    def test_startup_and_registration_failure_clean_owned_children(self):
        import task_worker
        before = {p.pid for p in mp.active_children()}
        iterator = EpochBatchIterator(UnpicklableInputLoader(), workers=2)
        with self.assertRaises((AttributeError, TypeError, pickle.PicklingError)):
            next(iterator)
        self.assert_no_new_children(before)
        with patch.dict(os.environ, {'WHOLE_HEART_MANAGED': '1'}), patch.object(
                task_worker, 'register_child', side_effect=RuntimeError('registration failed')):
            iterator = EpochBatchIterator(RandomLoader(), workers=2)
            with self.assertRaisesRegex(RuntimeError, 'registration failed'):
                next(iterator)
        self.assert_no_new_children(before)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                         'optional Torch/NumPy CPU batch test')
    def test_cpu_tensor_batches_survive_pool_close_and_never_seed_cuda(self):
        import torch
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CUDA seed called')):
            serial = self.collect(TensorLoader(), count=2, workers=0)
        parallel = self.collect(TensorLoader(), count=2, workers=2)
        for expected, actual in zip(serial, parallel):
            self.assertTrue(torch.equal(expected['data'], actual['data']))
            self.assertTrue((expected['array'] == actual['array']).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_empty_iterator_finishes_and_close_is_idempotent(self):
        before = {p.pid for p in mp.active_children()}
        for workers in (0, 2):
            iterator = EpochBatchIterator(EmptyLoader(), workers=workers, timeout_seconds=20)
            self.assertIs(iter(iterator), iterator)
            with self.assertRaises(StopIteration):
                next(iterator)
            iterator._finish()
            iterator.close()
            with self.assertRaises(StopIteration):
                next(iterator)
        self.assert_no_new_children(before)

    def test_pinning_only_when_requested_and_in_parent(self):
        for workers in (0, 2):
            batch = self.collect(PinLoader(), count=1, workers=workers)[0]
            self.assertFalse(hasattr(batch['data'][0], 'pinned_pid'))
            batch = self.collect(PinLoader(), count=1, workers=workers, pin_memory=True)[0]
            self.assertEqual(batch['data'][0].pinned_pid, os.getpid())

    def test_managed_registration_uses_actual_child_pid(self):
        import task_worker
        with patch.dict(os.environ, {'WHOLE_HEART_MANAGED': '1'}), patch.object(task_worker, 'register_child') as register:
            iterator = EpochBatchIterator(RandomLoader(), workers=2, timeout_seconds=20)
            try:
                next(iterator)
                self.assertEqual(register.call_count, 2)
                self.assertTrue(all(call.args[0].pid != os.getpid() for call in register.call_args_list))
                self.assertTrue(all('batch' in call.args[1] for call in register.call_args_list))
            finally:
                iterator.close()

    def test_invalid_settings_and_state_do_not_start_children(self):
        for settings in ({'workers': -1}, {'workers': 1.5}, {'prefetch': 0},
                         {'timeout_seconds': 0}, {'timeout_seconds': float('inf')}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                EpochBatchIterator(RandomLoader(), **settings)
        iterator = EpochBatchIterator(RandomLoader())
        state = iterator.state_dict()
        for value in (-1, 1.2):
            with self.assertRaises(ValueError):
                iterator.set_epoch(0, start_index=value)
            with self.assertRaises(ValueError):
                iterator.load_state_dict(dict(state, next_index=value))
        with self.assertRaises(ValueError):
            iterator.load_state_dict(dict(state, version=999))
        iterator.close()

    def test_lock_transfer_is_deferred_until_spawn_serialization(self):
        import batch_pool
        from multiprocessing import reduction
        with patch.object(reduction, 'DupFd', create=True) as duplicate:
            transfer = batch_pool._SpawnLockFd(1234)
            duplicate.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'spawn serialization'):
                pickle.dumps(transfer)
            duplicate.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX flock and descriptor inheritance')
    def test_registration_failure_before_worker_detach_releases_lock(self):
        import batch_pool
        import fcntl
        import task_worker
        before = {process.pid for process in mp.active_children()}
        with tempfile.TemporaryDirectory() as temp:
            lock_path = Path(temp, 'workflow.lock')
            marker = Path(temp, 'before-detach')
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            iterator = EpochBatchIterator(RandomLoader(), workers=1, timeout_seconds=20)
            def fail_registration(*_args):
                deadline = time.monotonic() + 10
                while not marker.exists():
                    if time.monotonic() >= deadline:
                        raise AssertionError('child did not reach delayed bootstrap')
                    time.sleep(.01)
                raise RuntimeError('registration failed before worker detach')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.dict(os.environ, {'WHOLE_HEART_LOCK_FD': str(fd),
                                             'WHOLE_HEART_MANAGED': '1',
                                             'BATCH_POOL_BOOTSTRAP_MARKER': str(marker)}), patch.object(
                        batch_pool, '_worker_main', delayed_lock_worker), patch.object(
                        task_worker, 'register_child', side_effect=fail_registration):
                    with self.assertRaisesRegex(RuntimeError, 'registration failed before worker detach'):
                        next(iterator)
                self.assert_no_new_children(before)
                os.close(fd)
                fd = None
                probe = os.open(lock_path, os.O_RDWR)
                try:
                    # Parent is still alive. No abandoned transfer may retain
                    # its flock after the only worker has been joined.
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(probe)
            finally:
                iterator.close()
                if fd is not None:
                    os.close(fd)

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX flock and descriptor inheritance')
    def test_each_spawn_child_holds_shared_lock_and_parent_process_group(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temp:
            lock_path = Path(temp, 'workflow.lock')
            marker_dir = Path(temp, 'markers')
            marker_dir.mkdir()
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            iterator = None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                inode = os.fstat(fd).st_ino
                with patch.dict(os.environ, {'WHOLE_HEART_LOCK_FD': str(fd)}):
                    iterator = EpochBatchIterator(LockLoader(str(marker_dir)), workers=2, timeout_seconds=20)
                    self.assertEqual(next(iterator), inode)
                    deadline = time.monotonic() + 10
                    while len(list(marker_dir.iterdir())) < 2 and time.monotonic() < deadline:
                        time.sleep(.02)
                    records = [json.loads(p.read_text()) for p in marker_dir.iterdir()]
                    self.assertEqual(len(records), 2)
                    self.assertTrue(all(r['inode'] == inode and r['pgid'] == os.getpgrp() for r in records))
                    os.close(fd)
                    fd = None
                    probe = os.open(lock_path, os.O_RDWR)
                    try:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # One child closing cannot release the other child's shared open-file lock.
                        iterator._processes[0].terminate()
                        iterator._processes[0].join(3)
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        iterator.close()
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(probe)
            finally:
                if iterator is not None:
                    iterator.close()
                if fd is not None:
                    os.close(fd)


if __name__ == '__main__':
    unittest.main()
