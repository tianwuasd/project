import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dataset1282_analysis.contracts import read_json
from dataset1282_analysis.resources import GIB, snapshot
from dataset1282_analysis.xu_capacity import admission, specification


class CapacityPrefetchTests(unittest.TestCase):
    def test_formal_grids_use_bounded_prefetch_with_actual_smoke_networks(self):
        import copy
        from dataclasses import replace
        import torch
        from dataset1282_analysis.xu_capacity import _samples_3d
        from dataset1282_analysis.xu_models import branch_spec, Xu3DSegmenter
        from dataset1282_analysis.xu_training import _epoch

        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        keys = ('XU_PREFETCH', 'XU_PREFETCH_BYTES', 'XU_PREFETCH_WORKERS')
        previous = {key: os.environ.get(key) for key in keys}
        def restore():
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(restore)
        os.environ.update(XU_PREFETCH_BYTES=str(GIB), XU_PREFETCH_WORKERS='2')
        for branch in ('crop64', 'crop128'):
            with self.subTest(branch=branch):
                spec = replace(branch_spec(branch, 'smoke'), batch=4)
                names, groups, load = _samples_3d(spec, 'marginal')
                self.assertEqual(load.source_voxels(names[0]), spec.side**3)
                image, label = load(names[0], branch)
                self.assertEqual(image.shape, (spec.side,)*3)
                self.assertEqual(label.shape, image.shape)
                self.assertEqual(len(names), 4)
                self.assertEqual(groups[names[1]], [10])
                initial = Xu3DSegmenter(spec)
                before = initial.head.weight.detach().clone()
                outcomes = []
                for enabled in ('0', '1'):
                    os.environ['XU_PREFETCH'] = enabled
                    model = copy.deepcopy(initial)
                    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                    rng = torch.Generator().manual_seed(2023)
                    stats = {}
                    loss = _epoch(model, names, groups, load, branch, 'cpu', rng,
                                  optimizer, stats)
                    self.assertGreater((model.head.weight.detach()-before).abs().max().item(), 0)
                    outcomes.append((loss, copy.deepcopy(model.state_dict()), rng.get_state()))
                self.assertTrue(stats['requested'])
                self.assertTrue(stats['applied'])
                self.assertEqual(stats['prefetch_batches'], 1)
                self.assertGreater(stats['workers_applied'], 0)
                self.assertLessEqual(stats['reserved_peak_bytes'], GIB)
                self.assertEqual(outcomes[0][0], outcomes[1][0])
                for name, value in outcomes[0][1].items():
                    torch.testing.assert_close(value, outcomes[1][1][name], rtol=0, atol=0)
                self.assertTrue(torch.equal(outcomes[0][2], outcomes[1][2]))

    def test_formal_cpu_step_with_prefetch_updates_weights_and_records_supply(self):
        spec = specification('3d64')
        denied, facts = admission(spec, snapshot()['available_memory_bytes'])
        if denied:
            self.skipTest(f'正式 64³ CPU 容量检查内存不足: {facts}')
        root = Path(__file__).resolve().parents[1]
        private = root / 'local-validation'
        private.mkdir(exist_ok=True)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('XU_')}
        env.update(XU_PREFETCH='1', XU_PREFETCH_BYTES=str(GIB),
                   XU_PREFETCH_WORKERS='2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4')
        with tempfile.TemporaryDirectory(dir=private) as directory:
            result = subprocess.run(
                [sys.executable, '-B', str(root / 'scripts/xu_capacity.py'),
                 '--stage', '3d64', '--device', 'cpu', '--output', directory],
                cwd=root, env=env, capture_output=True, text=True, encoding='utf-8',
                timeout=600)
            report = read_json(Path(directory) / 'capacity.json')
        self.assertEqual(result.returncode, 0, report)
        self.assertEqual(report['state'], 'passed')
        self.assertEqual(report['optimizer_steps'], 1)
        self.assertEqual(report['shape'], [4, 1, 64, 64, 64])
        self.assertGreater(report['head_max_abs_update'], 0)
        supply = report['supply']
        self.assertTrue(supply['requested'])
        self.assertTrue(supply['applied'])
        self.assertEqual(supply['prefetch_batches'], 1)
        self.assertEqual(supply['budget_bytes'], GIB)
        self.assertGreater(supply['workers_applied'], 0)
        self.assertLessEqual(supply['reserved_peak_bytes'], GIB)
