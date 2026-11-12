import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from chd_stunet.observability.runlog import write_json
from chd_stunet.preprocessing.shared_cache import MODE, begin_view, commit_view, data_directory, lease, ready_view, reader_claim


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


class SharedCacheTests(unittest.TestCase):
    def setUp(self):
        self.started = time.monotonic()
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ['CHD_SHARED_TEST_ROOT'])
        self.root = Path(self.temporary.name)
        self.samples = [(ROOT/'configs/preprocessing'/name).read_bytes()
                        for name in ('stunet-s.json', 'stunet-s-partial.json')]
        write_json(self.root/'signature.json', dict(identity='technical-storage-test', method={'storage_mode': MODE}))
        for fold in (0, 1):
            write_json(self.root/f'fold_{fold}/preprocessing_plan.json', {'fold': fold})

    def tearDown(self):
        files = [p for p in self.root.rglob('*') if p.is_file()]
        print(json.dumps(dict(test=self.id(), scope='file_management_only', samples=2,
                              files=len(files), logical_bytes=sum(p.stat().st_size for p in files),
                              seconds=time.monotonic()-self.started, training_updates=0,
                              memory_peak='not_measured', gpu_peak='not_measured')))
        self.temporary.cleanup()

    def fill(self, fold, cid='sample_a'):
        contract = digest({'run': 'technical-storage-test', 'plan': {'fold': fold}})
        begin_view(self.root, fold, [cid], contract)
        directory = data_directory(self.root, fold, check=False)
        directory.mkdir(exist_ok=True)
        hashes = {}
        for index, suffix in enumerate(('.b2nd', '_seg.b2nd', '.pkl')):
            # 此处仅测试文件归属及生命周期，不把样本字节解释为医学数组。
            path = directory/(cid+suffix)
            path.write_bytes(self.samples[(index+fold) % 2])
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(self.root/f'fold_{fold}/completion/{cid}.json',
                   dict(case_id=cid, fold=fold, success=True, contract=contract, file_sha256=hashes))
        commit_view(self.root, fold, [cid], contract)

    def test_one_directory_and_wrong_fold_refusal(self):
        self.fill(0)
        self.assertEqual(data_directory(self.root, 0), self.root/'shared_data')
        with self.assertRaises(ValueError):
            data_directory(self.root, 1)
        self.fill(1)
        self.assertEqual(data_directory(self.root, 1), self.root/'shared_data')
        self.assertEqual(len(list((self.root/'shared_data').iterdir())), 3)
        self.assertFalse((self.root/'fold_0/data').exists())
        with self.assertRaises(ValueError):
            ready_view(self.root, 0)

    def test_building_is_not_available_and_missing_file_blocks_commit(self):
        contract = digest({'run': 'technical-storage-test', 'plan': {'fold': 0}})
        begin_view(self.root, 0, ['sample_a'], contract)
        with self.assertRaises(ValueError):
            ready_view(self.root, 0)
        with self.assertRaises(FileNotFoundError):
            commit_view(self.root, 0, ['sample_a'], contract)

    def test_training_lease_prevents_switch(self):
        self.fill(0)
        before = (self.root/'shared_cache.json').read_bytes()
        with lease(self.root, fold=0):
            with self.assertRaises(RuntimeError):
                with lease(self.root, enabled=True):
                    self.fail('不能在读取期间改写共享缓存')
        self.assertEqual((self.root/'shared_cache.json').read_bytes(), before)

    def test_child_reader_prevents_switch_without_supervisor(self):
        self.fill(0)
        with reader_claim(self.root, 0):
            with self.assertRaises(RuntimeError):
                with lease(self.root, enabled=True):
                    self.fail('独立读取进程仍持有缓存')
        with lease(self.root, enabled=True):
            pass
        self.assertEqual(list((self.root/'.cache-readers').iterdir()), [])

    def test_reader_must_reject_a_different_cache_generation(self):
        self.fill(0)
        generation = ready_view(self.root, 0)['generation']
        self.fill(1)
        self.fill(0)
        with self.assertRaises(ValueError):
            with reader_claim(self.root, 0, generation=generation):
                self.fail('孤立供数进程不能读取已经重新构建的缓存')

    def test_writer_child_keeps_ownership_after_parent_lease_ends(self):
        contract = digest({'run': 'technical-storage-test', 'plan': {'fold': 0}})
        claim = None
        with lease(self.root, enabled=True) as token:
            begin_view(self.root, 0, ['sample_a'], contract)
            claim = reader_claim(self.root, 0, writer_token=token, contract=contract)
            claim.__enter__()
        try:
            with self.assertRaises(RuntimeError):
                with lease(self.root, enabled=True):
                    self.fail('写入子进程仍持有缓存')
        finally:
            claim.__exit__(None, None, None)
        with lease(self.root, enabled=True):
            with self.assertRaises(ValueError):
                with reader_claim(self.root, 0, writer_token=token, contract=contract):
                    self.fail('旧写入任务不能进入新的构建过程')

    def test_read_only_lease_preserves_stale_claims_and_all_files(self):
        self.fill(0)
        with lease(self.root, enabled=True):
            pass
        stale = self.root/'.cache-readers'/('a'*32+'.lock')
        stale.write_bytes(self.samples[0])
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with lease(self.root, read_only=True):
            self.assertTrue(stale.is_file())
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_forced_parent_exit_preserves_live_child_protection(self):
        self.fill(0)
        child = None
        parent = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--parent', str(self.root)])
        try:
            deadline = time.monotonic()+15
            while not (self.root/'child-ready.json').is_file() and time.monotonic() < deadline:
                self.assertIsNone(parent.poll())
                time.sleep(.02)
            self.assertTrue((self.root/'child-ready.json').is_file())
            record = json.loads((self.root/'child-process.json').read_text(encoding='utf-8'))
            child = psutil.Process(record['pid'])
            self.assertEqual(child.create_time(), record['born'])
            parent.terminate()
            parent.wait(timeout=10)
            self.assertTrue(child.is_running())
            with self.assertRaises(RuntimeError):
                with lease(self.root, enabled=True):
                    self.fail('父进程退出后，真实子进程仍在读取')
        finally:
            (self.root/'child-stop').write_bytes(b'0')
            if parent.poll() is None:
                parent.terminate()
            parent.wait(timeout=10)
            if child is None and (self.root/'child-process.json').is_file():
                record = json.loads((self.root/'child-process.json').read_text(encoding='utf-8'))
                child = psutil.Process(record['pid'])
                self.assertEqual(child.create_time(), record['born'])
            if child is not None:
                child.wait(timeout=10)
        with lease(self.root, enabled=True):
            pass

    def test_switch_releases_only_registered_obsolete_files(self):
        self.fill(0)
        self.fill(1, 'sample_b')
        self.assertEqual({p.name for p in (self.root/'shared_data').iterdir()},
                         {'sample_b.b2nd', 'sample_b_seg.b2nd', 'sample_b.pkl'})
        (self.root/'shared_data/unknown.bin').write_bytes(self.samples[0])
        with self.assertRaises(ValueError):
            begin_view(self.root, 0, ['sample_a'], digest({'run': 'technical-storage-test', 'plan': {'fold': 0}}))
        self.assertTrue((self.root/'shared_data/unknown.bin').is_file())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--parent':
        root = Path(sys.argv[2])
        with lease(root, fold=0):
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--child', str(root)])
            write_json(root/'child-process.json', dict(pid=process.pid, born=psutil.Process(process.pid).create_time()))
            while not (root/'child-stop').is_file():
                time.sleep(.02)
    elif len(sys.argv) == 3 and sys.argv[1] == '--child':
        root = Path(sys.argv[2])
        with reader_claim(root, 0):
            write_json(root/'child-ready.json', dict(pid=os.getpid()))
            while not (root/'child-stop').is_file():
                time.sleep(.02)
    else:
        unittest.main()
