from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import shutil
import unittest

import numpy as np

from dataset1282_analysis.artifact_retention import (
    release_branch_arrays, release_input_cache, release_shared_prediction_cache,
    validate_retained_files)
from dataset1282_analysis.contracts import atomic_json, digest, sha256_file
from dataset1282_analysis.xu_sample_cache import ArrayCache
from dataset1282_analysis.observability.status import operation_lock


class RetentionLiveTests(unittest.TestCase):
    def setUp(self):
        home = Path('D:/projects/back/t17/dataset1282-retention')
        home.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(dir=home)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def stage(self):
        for name in ('branches/all128/ct.npy', 'cases/ct.npy', 'cases/ct.nii.gz', 'complete.json'):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        return {path.relative_to(self.root).as_posix(): sha256_file(path)
                for path in self.root.rglob('*') if path.is_file()}

    def test_release_keeps_results_and_original_receipt_and_is_idempotent(self):
        expected = self.stage()
        required = ('complete.json', 'cases/ct.npy', 'cases/ct.nii.gz')
        report = release_branch_arrays(self.root, expected, required)
        self.assertGreater(report['removed_bytes'], 0)
        self.assertFalse((self.root/'branches/all128/ct.npy').exists())
        self.assertEqual(validate_retained_files(self.root, expected, required), expected)
        self.assertEqual(release_branch_arrays(self.root, expected, required)['removed_bytes'], 0)
        (self.root/'cases/ct.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_retained_files(self.root, expected, required)

    def test_unrecorded_missing_and_changed_cache_prevent_cleanup(self):
        expected = self.stage()
        path = self.root/'branches/all128/ct.npy'
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            release_branch_arrays(self.root, expected)
        self.assertTrue(path.exists())
        self.assertFalse((self.root/'.runtime/retention.json').exists())
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            validate_retained_files(self.root, expected, ())

    def test_release_intent_survives_interruption_and_cannot_release_actual_results(self):
        expected = self.stage()
        ledger = self.root/'.runtime/retention.json'
        atomic_json(ledger, dict(schema=1, receipt_digest=digest(expected),
                    released={'branches/all128/ct.npy': expected['branches/all128/ct.npy']}))
        self.assertEqual(validate_retained_files(self.root, expected, ()), expected)
        release_branch_arrays(self.root, expected)
        atomic_json(ledger, dict(schema=1, receipt_digest=digest(expected),
                    released={'cases/ct.npy': expected['cases/ct.npy']}))
        with self.assertRaisesRegex(ValueError, 'Invalid released'):
            validate_retained_files(self.root, expected, ())

    def test_required_branch_array_stays_required(self):
        expected = self.stage()
        required = ('branches/all128/ct.npy',)
        release_branch_arrays(self.root, expected, required)
        self.assertTrue((self.root/required[0]).exists())
        (self.root/required[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            validate_retained_files(self.root, expected, required)

    def test_real_array_builds_run_concurrently_across_cache_instances(self):
        from concurrent.futures import ThreadPoolExecutor
        barrier = threading.Barrier(2, timeout=10)
        first = ArrayCache(self.root/'.input-cache',max_bytes=20000)
        second = ArrayCache(self.root/'.input-cache',max_bytes=20000)
        def build(value):
            array = np.full((10,10,10),value,dtype=np.float32)
            barrier.wait()
            return array
        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(first.get, {'case':'left'}, lambda:build(1))
            right = pool.submit(second.get, {'case':'right'}, lambda:build(2))
            np.testing.assert_array_equal(left.result(timeout=15),np.ones((10,10,10)))
            np.testing.assert_array_equal(right.result(timeout=15),np.full((10,10,10),2))
        self.assertEqual(len(list(first.root.glob('*.npy'))),2)
        self.assertLessEqual(sum(path.stat().st_size for path in first.root.iterdir()),20000)

    def test_parallel_capacity_pressure_returns_exact_arrays_with_bounded_storage(self):
        from concurrent.futures import ThreadPoolExecutor
        from dataset1282_analysis.artifact_retention import cache_lock_home
        from dataset1282_analysis.contracts import read_json
        barrier = threading.Barrier(2,timeout=10)
        cache = ArrayCache(self.root/'.input-cache',max_bytes=8500)
        def build(value):
            values = np.full((10,10,10),value,dtype=np.float32)
            barrier.wait()
            return values
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(cache.get,{'case':str(value)},lambda value=value:build(value))
                       for value in (1,2)]
            for value,future in enumerate(futures,1):
                np.testing.assert_array_equal(future.result(timeout=15),np.full((10,10,10),value))
        self.assertLessEqual(sum(path.stat().st_size for path in cache.root.iterdir()),8500)
        self.assertEqual(read_json(cache_lock_home(cache.root)/'reservations.json'),{})

    def test_lru_bound_exact_values_checksum_and_oversized_bypass(self):
        cache = ArrayCache(self.root/'.input-cache', max_bytes=13000)
        values = np.arange(1000, dtype=np.float32).reshape(10,10,10)
        for name in ('first', 'second', 'first', 'third'):
            np.testing.assert_array_equal(cache.get({'name': name}, lambda: values), values)
        self.assertTrue((cache.root/(digest({'name':'first'})+'.npy')).exists())
        self.assertFalse((cache.root/(digest({'name':'second'})+'.npy')).exists())
        self.assertLessEqual(sum(path.stat().st_size for path in cache.root.iterdir()),13000)
        path = cache.root/(digest({'name':'first'})+'.npy')
        damaged = values.copy()
        damaged.flat[0] = -3
        np.save(path,damaged,allow_pickle=False)
        with self.assertRaisesRegex(ValueError, '缓存'):
            cache.get({'name':'first'},lambda: values)
        bypass = ArrayCache(self.root/'oversized', max_bytes=100)
        np.testing.assert_array_equal(bypass.get({'name':'big'},lambda: values),values)
        self.assertEqual(list(bypass.root.iterdir()), [])

    def test_active_cache_is_not_deleted_and_cleanup_is_narrow(self):
        cache = ArrayCache(self.root/'.input-cache')
        started, finish = threading.Event(), threading.Event()
        def build():
            started.set()
            if not finish.wait(10):
                raise RuntimeError('Timed out waiting for cache cleanup assertion')
            return np.ones((2,2,2),dtype=np.float32)
        def worker():
            result.set_result(cache.get({'case':'ct'},build))
        from concurrent.futures import Future
        result = Future()
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(started.wait(10))
            with self.assertRaises(RuntimeError):
                release_input_cache(self.root)
            self.assertTrue(cache.root.exists())
        finally:
            finish.set()
            thread.join(10)
        self.assertTrue(result.done())
        other = self.root/'source.npy'
        np.save(other,np.ones((2,2,2)),allow_pickle=False)
        release_input_cache(self.root)
        self.assertTrue(other.exists())
        self.assertFalse(cache.root.exists())

    def test_shared_cache_verifies_complete_entry_and_clears_only_cache(self):
        root = self.root/'prediction-cache'
        identity = dict(checkpoint='frozen', branch='all128')
        entry = root/'entries'/digest(identity)
        entry.mkdir(parents=True)
        source = self.root/'external.npy'
        np.save(source,np.ones((2,2,2),dtype=np.uint8),allow_pickle=False)
        shutil.copyfile(source,entry/'ct.npy')
        files = {'ct':sha256_file(source)}
        atomic_json(entry/'complete.json', dict(checkpoint='frozen', files=files))
        record = dict(identity=identity,files=files)
        atomic_json(entry/'manifest.json', dict(record=record,digest=digest(record)))
        with operation_lock(root/'locks'/entry.name, 'live-prediction'):
            with self.assertRaises(RuntimeError):
                release_shared_prediction_cache(root)
            self.assertTrue((entry/'ct.npy').exists())
        (entry/'ct.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            release_shared_prediction_cache(root)
        self.assertTrue(entry.exists())
        shutil.copyfile(source,entry/'ct.npy')
        report = release_shared_prediction_cache(root)
        self.assertEqual(report['entries'],1)
        self.assertFalse(entry.exists())
        self.assertEqual(sha256_file(source),files['ct'])
        self.assertEqual(release_shared_prediction_cache(root)['entries'],0)
