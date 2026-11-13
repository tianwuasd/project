import json
import os
from pathlib import Path
import sys
import time
import unittest
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import update_aaoca_code as updater


@unittest.skipUnless(os.environ.get('AAOCA_CODE_UPDATE_TEST_ROOT'), '需要指定已批准的源码测试目录')
class CodeUpdateTests(unittest.TestCase):
    def test_two_real_source_directories(self):
        started = time.monotonic()
        root = Path(os.environ['AAOCA_CODE_UPDATE_TEST_ROOT']).resolve()
        commit = os.environ.get('AAOCA_CODE_UPDATE_TEST_COMMIT', 'f5ff0889bd8bc58d230d15ca764eb5cb5308b61f')
        snapshot, data = updater.Download('gh-proxy', root).snapshot(commit)
        first, second = root / 'first/AAOCA', root / 'second/AAOCA'
        first.mkdir(parents=True, exist_ok=True)
        second.mkdir(parents=True, exist_ok=True)
        for target in (first, second):
            updater.apply_snapshot(target, snapshot, data)
        original = data['README.md']
        (first / 'README.md').write_bytes(original + b'\nLocal edit\n')
        unknown = first / '.local-note.txt'
        unknown.write_text('private local note', encoding='utf-8')
        result = updater.update(first, root, 'gh-proxy', commit)
        self.assertEqual((first / 'README.md').read_bytes(), original)
        self.assertEqual(unknown.read_text(encoding='utf-8'), 'private local note')
        self.assertEqual(result['commit'], commit)
        self.assertFalse(list(root.glob('.aaoca-code-update-*')))
        self.assertFalse((first / updater.LOCK).exists())

        stale = second / 'retired-code.py'
        stale.write_bytes(data['tools/coarse_segmentation/aaoca_seg/__init__.py'])
        receipt_path = second / updater.RECEIPT
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        retired = {'path': stale.name, 'sha': updater.blob_hash(stale.read_bytes()),
                   'size': stale.stat().st_size, 'mode': '100644'}
        receipt['files'].append(retired)
        receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
        original_stale = stale.read_bytes()
        stale.write_bytes(original_stale + b'\n# Local edit\n')
        with self.assertRaisesRegex(ValueError, '本地修改'):
            updater.apply_snapshot(second, snapshot, data)
        self.assertTrue(stale.read_bytes().endswith(b'# Local edit\n'))
        stale.write_bytes(original_stale)
        result = updater.apply_snapshot(second, snapshot, data)
        self.assertEqual(result['removed_retired_files'], 1)
        self.assertFalse(stale.exists())
        self.assertEqual(updater.apply_snapshot(second, snapshot, data)['updated'], 0)

        before = {path.relative_to(second): path.read_bytes() for path in second.rglob('*') if path.is_file()}
        corrupted = dict(data)
        corrupted['README.md'] = original + b'changed'
        with self.assertRaisesRegex(ValueError, '哈希不符'):
            updater.apply_snapshot(second, snapshot, corrupted)
        self.assertEqual(before, {path.relative_to(second): path.read_bytes() for path in second.rglob('*') if path.is_file()})

        documents, held = second / 'docs', second / '.held-docs'
        documents.replace(held)
        documents.write_bytes(b'local file occupies source directory')
        blocked = {path.relative_to(second): path.read_bytes() for path in second.rglob('*') if path.is_file()}
        try:
            with self.assertRaisesRegex(ValueError, '父路径'):
                updater.apply_snapshot(second, snapshot, data)
            self.assertEqual(blocked, {path.relative_to(second): path.read_bytes() for path in second.rglob('*') if path.is_file()})
        finally:
            documents.unlink()
            held.replace(documents)

        restore_test = 'not_measured_on_this_platform'
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            (second / 'AGENTS.md').write_bytes(data['AGENTS.md'] + b'\nLocal edit\n')
            (second / 'README.md').write_bytes(original + b'\nLocal edit\n')
            before_restore = {path.relative_to(second): path.read_bytes() for path in second.rglob('*') if path.is_file()}
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            kernel.CreateFileW.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.CreateFileW(str(second / 'README.md'), 0x80000000, 3, None, 3, 128, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            try:
                with self.assertRaises(PermissionError):
                    updater.apply_snapshot(second, snapshot, data)
            finally:
                kernel.CloseHandle(handle)
            self.assertEqual(before_restore, {path.relative_to(second): path.read_bytes() for path in second.rglob('*') if path.is_file()})
            updater.apply_snapshot(second, snapshot, data)
            restore_test = 'actual_Windows_deny_delete_handle_mid_apply_restore_passed'

        for path in ('../outside.py', '/absolute.py', '.git/config', 'runtime/model.py', 'scan.nii.gz'):
            with self.assertRaises(ValueError):
                updater.safe_name(path)
        with self.assertRaises(subprocess.CalledProcessError):
            updater.update(first, root, 'gh-proxy', '0' * 40)
        self.assertFalse(list(root.glob('.aaoca-code-update-*')))
        self.assertFalse((first / updater.LOCK).exists())
        self.assertEqual((first / 'README.md').read_bytes(), original)
        self.assertEqual([path for path in first.rglob('.aaoca-code-*')
                          if path.name != updater.RECEIPT], [])
        report = {'real_source_directories': 2, 'commit': commit, 'files_per_snapshot': len(data),
                  'source_bytes': sum(map(len, data.values())), 'seconds': time.monotonic() - started,
                  'checks': ['real_mirror_download', 'hash_verified_apply', 'local_edit_replaced',
                             'unknown_file_preserved', 'retired_source_removed', 'repeat_update',
                             'modified_retired_source_preserved', 'parent_file_obstruction_rejected',
                             'corruption_rejected_before_apply', 'unsafe_paths_rejected',
                             'real_download_failure_cleaned'], 'mid_apply_restore': restore_test,
                  'CT_read': 0, 'model_runs': 0}
        (root / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    unittest.main()
