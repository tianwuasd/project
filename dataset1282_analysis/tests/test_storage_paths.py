import json
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import unittest
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dataset1282_analysis.contracts import atomic_json, sha256_file
from dataset1282_analysis.storage_paths import assert_output_location, validate_input_output_paths
from dataset1282_analysis.storage_migration import migration_plan, apply_migration, canonical_path, load_relocation
from dataset1282_analysis.observability.status import operation_lock

BASE = Path('D:/projects/back/t17/d1282-storage') if sys.platform == 'win32' else ROOT / 'local-validation/storage-tests'
BASE.mkdir(parents=True, exist_ok=True)


class StorageTests(unittest.TestCase):
    def test_nas_lexical_and_overlap(self):
        for path in ('/data_nas/out', '/data_nas', '/data_nas/data/../out', '/other/../data_nas/output'):
            with self.assertRaises(ValueError):
                assert_output_location(path)
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            path = Path(temporary)
            self.assertEqual(assert_output_location(path), path.resolve())
            with self.assertRaises(ValueError):
                validate_input_output_paths(path, {'work_root': path / 'outputs'})
            with self.assertRaises(ValueError):
                validate_input_output_paths(path / 'data', {'work_root': path})

    def test_copy_preserves_contract_and_canonical_identity(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            base = Path(temporary); source = base / 'old'; target = base / 'new'; source.mkdir()
            (source / 'prepared').mkdir(); (source / 'experiments').mkdir()
            atomic_json(source / 'experiments/run.json', {'prepared': str(source / 'prepared'), 'argv': [str(source)]})
            (source / 'prepared/data.bin').write_bytes(bytes(range(256)))
            (source / 'experiments/checkpoint.pt').write_bytes(b'checkpoint')
            atomic_json(source / 'experiments/request.json', {'gpu': 0})
            (source / 'prediction-cache').mkdir()
            atomic_json(source / 'prediction-cache/complete.json', {'temporary_cache': True})
            before = sha256_file(source / 'experiments/run.json')
            plan = migration_plan(source, target); atomic_json(base / 'plan.json', plan)
            result = apply_migration(base / 'plan.json', base / 'home')
            self.assertTrue(result['source_preserved'])
            self.assertTrue(source.exists())
            self.assertEqual(sha256_file(target / 'experiments/run.json'), before)
            self.assertEqual(canonical_path(target / 'prepared'), str(source / 'prepared'))
            self.assertEqual(load_relocation(target)['source'], str(source))
            (target / 'experiments/checkpoint.pt').write_bytes(b'resumed checkpoint')
            atomic_json(target / 'experiments/request.json', {'gpu': 1})
            (target / 'prediction-cache/complete.json').unlink()
            self.assertEqual(canonical_path(target / 'prepared'), str(source / 'prepared'))
            (target / 'experiments/run.json').write_text('{}')
            with self.assertRaises(ValueError):
                canonical_path(target / 'prepared')

    def test_changed_source_and_live_locks_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            base = Path(temporary); source = base / 'source'; source.mkdir()
            (source / 'data.bin').write_bytes(b'abc')
            plan = migration_plan(source, base / 'target'); atomic_json(base / 'plan.json', plan)
            (source / 'data.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                apply_migration(base / 'plan.json', base / 'home')
            with operation_lock(source, 'training'):
                with self.assertRaises(RuntimeError):
                    migration_plan(source, base / 'target')
            self.assertFalse((base / 'target').exists())

    def test_mirror_rejects_local_auth_configuration(self):
        from dataset1282_analysis.maintenance import _git
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            cache = Path(temporary) / 'cache.git'
            subprocess.run(['git', 'init', '--bare', str(cache)], check=True, capture_output=True)
            self.assertTrue(_git(cache, 'rev-parse', '--is-bare-repository', public_only=True).strip() == b'true')
            subprocess.run(['git', '--git-dir', str(cache), 'config', 'http.extraHeader', 'X-Test: secret'], check=True)
            with self.assertRaises(ValueError):
                _git(cache, 'rev-parse', '--is-bare-repository', public_only=True)

    def test_partial_clone_capability_with_real_git_daemon(self):
        from dataset1282_analysis.maintenance import _git
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            base = Path(temporary); server = base / 'server.git'; cache = base / 'client.git'
            for path in (server, cache):
                subprocess.run(['git', 'init', '--bare', str(path)], check=True, capture_output=True)
            subprocess.run(['git', '--git-dir', str(server), 'config', 'uploadpack.allowFilter', 'true'], check=True)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            subprocess.run(['git', '--git-dir', str(cache), 'config', 'remote.origin.url',
                            f'git://127.0.0.1:{port}/server.git'], check=True)
            daemon = subprocess.Popen(['git', 'daemon', '--reuseaddr', '--export-all', '--listen=127.0.0.1',
                                       '--port=' + str(port), '--base-path=' + str(base), str(base)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                while True:
                    with socket.socket() as probe:
                        ready = probe.connect_ex(('127.0.0.1', port)) == 0
                    if ready:
                        break
                    if daemon.poll() is not None or time.monotonic() >= deadline:
                        self.fail('Git daemon did not become ready')
                    time.sleep(.05)
                _git(cache, 'ls-remote', '--heads', 'origin', 'main', public_only=True, require_filter=True)
                subprocess.run(['git', '--git-dir', str(server), 'config', 'uploadpack.allowFilter', 'false'], check=True)
                with self.assertRaises(ValueError):
                    _git(cache, 'ls-remote', '--heads', 'origin', 'main', public_only=True, require_filter=True)
            finally:
                daemon.terminate(); daemon.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
