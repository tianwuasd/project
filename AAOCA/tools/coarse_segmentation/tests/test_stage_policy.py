from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaoca_seg.resources import GIB
from aaoca_seg.scheduler import phase_disk, phase_memory, stage_capacity


class StagePolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'prepare_workers': 1, 'export_workers': 1, 'prefetch': 2}

    def test_ready_and_preparing_cases_share_one_bound(self):
        pending = [{'phase': 'infer', 'prepared': 'a/prepared.json'}, {'phase': 'infer', 'prepared': 'b/prepared.json'}]
        self.assertFalse(stage_capacity({'phase': 'prepare'}, pending, [], self.settings))
        self.assertTrue(stage_capacity({'phase': 'prepare'}, pending[:1], [], self.settings))
        self.assertFalse(stage_capacity({'phase': 'prepare'}, [], [{'phase': 'prepare'}], self.settings))

    def test_gpu_and_export_do_not_use_prepare_worker_cap(self):
        active = [{'phase': 'prepare'}]
        self.assertTrue(stage_capacity({'phase': 'infer'}, [], active, self.settings))
        self.assertTrue(stage_capacity({'phase': 'merge'}, [], active, self.settings))
        self.assertFalse(stage_capacity({'phase': 'verify'}, [], [{'phase': 'merge'}], self.settings))

    def test_stage_memory_and_disk_are_independent(self):
        item = {'phase': 'infer', 'memory': 12*GIB, 'prepare_memory': 2*GIB,
                'disk_bytes': 6*GIB, 'prepare_disk': 2*GIB, 'export_disk': 3*GIB}
        self.assertEqual((phase_memory(item), phase_disk(item)), (12*GIB, 6*GIB))
        item['phase'] = 'prepare'
        self.assertEqual((phase_memory(item), phase_disk(item)), (2*GIB, 2*GIB))
        item['phase'] = 'verify'
        self.assertEqual((phase_memory(item), phase_disk(item)), (GIB, 1024*1024))
        item['phase'] = 'merge'
        self.assertEqual(phase_disk(item), 3*GIB)


if __name__ == '__main__':
    unittest.main()
