import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from aaoca_seg.operations import PROBE, check_device
from aaoca_seg.resources import available_devices, canonical_gpu_uuid, child_env, eligible_devices, select_devices

UUID_A = '00000000-0000-4000-8000-00000000000a'
UUID_B = '00000000-0000-4000-8000-00000000000b'


class GpuUuidTests(unittest.TestCase):
    def test_prefixed_and_bare_uuid_have_one_identity(self):
        for value in (UUID_A, UUID_B):
            for spelling in (value, value.upper(), 'GPU-' + value):
                with self.subTest(spelling=spelling):
                    self.assertEqual(canonical_gpu_uuid(spelling), 'GPU-' + value)

    def test_invalid_or_mig_identity_rejected(self):
        for value in (None, '', 'GPU-', 'GPU-1234', 'MIG-' + UUID_A, 'x' * 36):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_gpu_uuid(value)

    def test_normalized_probe_identity_can_be_selected_and_bound(self):
        info = {'devices': [{'index': 7, 'uuid': canonical_gpu_uuid(UUID_A)}]}
        selected = select_devices(info, [7])
        self.assertEqual(selected[0]['uuid'], 'GPU-' + UUID_A)
        self.assertEqual(child_env(2, selected[0])['CUDA_VISIBLE_DEVICES'], 'GPU-' + UUID_A)
        with self.assertRaises(ValueError):
            select_devices(info, [0])

    def test_saved_identity_accepts_both_spellings_and_rejects_another_gpu(self):
        for saved, current in ((UUID_A, 'GPU-' + UUID_A), ('GPU-' + UUID_A, UUID_A)):
            config = {'device': 'gpu:0', 'gpu_confirmed': True, 'gpu_uuid': saved}
            info = {'cuda_available': True, 'devices': [{'index': 0, 'uuid': current}]}
            check_device(config, info)
            info['devices'][0]['uuid'] = 'GPU-' + UUID_B
            with self.assertRaisesRegex(ValueError, '映射已变化'):
                check_device(config, info)
        check_device({'device': 'cpu'}, {})

    def test_missing_allocated_gpu_rejected_before_occupancy_query(self):
        selected = [{'index': 7, 'uuid': 'GPU-' + UUID_A}]
        current = {'devices': [{'index': 0, 'uuid': 'GPU-' + UUID_B}]}
        with self.assertRaisesRegex(ValueError, '可见范围/分配已变化'):
            available_devices(selected, current)

    def test_busy_gpu_with_enough_memory_and_reserve_is_eligible(self):
        selected = [{'index': 7, 'uuid': 'GPU-' + UUID_A}, {'index': 6, 'uuid': 'GPU-' + UUID_B}]
        current = {'devices': [{'index': 0, 'uuid': UUID_A}, {'index': 1, 'uuid': UUID_B}]}
        snapshot = {'GPU-' + UUID_A: {'free_mib': 12 * 1024, 'utilization': 99},
                    'GPU-' + UUID_B: {'free_mib': 9 * 1024, 'utilization': 99}}
        occupied = {'GPU-' + UUID_A, 'GPU-' + UUID_B}
        self.assertEqual(eligible_devices(selected, current, snapshot, occupied), [])
        self.assertEqual(eligible_devices(selected, current, snapshot, occupied, 8, allow_shared=True, reserve_gib=2),
                         [{'index': 0, 'uuid': 'GPU-' + UUID_A}])
        snapshot['GPU-' + UUID_B]['free_mib'] = 10 * 1024
        self.assertEqual(len(eligible_devices(selected, current, snapshot, occupied, 8, allow_shared=True, reserve_gib=2)), 2)

    @unittest.skipUnless(os.environ.get('AAOCA_GPU_CHECK') == '1', '需要显式启用一张真实 GPU 的只读检查')
    def test_real_probe_selection_and_uuid_isolated_child(self):
        env = dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', '-c', PROBE], env=env,
                                capture_output=True, check=True, timeout=60)
        info = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(info['cuda_available'])
        self.assertTrue(info['devices'])
        selected = select_devices(info, [info['devices'][0]['index']])
        uid = selected[0]['uuid']
        self.assertTrue(uid.startswith('GPU-'))
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                               capture_output=True, check=True, timeout=10)
        system_uuids = {row[0].strip() for row in csv.reader(query.stdout.decode().splitlines())}
        self.assertIn(uid, system_uuids)
        free = available_devices(selected, info, .05, allow_shared=True, reserve_gib=.05)
        self.assertTrue(free)
        self.assertTrue(all(device['uuid'] == uid for device in free))
        isolated = {**child_env(1, selected[0]), 'PYTHONPATH': str(SOURCE),
                    'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run([sys.executable, '-B', '-c', PROBE], env=isolated,
                                capture_output=True, check=True, timeout=60)
        child = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual([(device['index'], device['uuid']) for device in child['devices']], [(0, uid)])


if __name__ == '__main__':
    unittest.main()
