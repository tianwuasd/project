import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import blosc2

from dataset1282_analysis.contracts import atomic_json, read_json, sha256_file
from dataset1282_analysis.xu_auto_init import (auto_eligible, auto_proximal_target,
                                             build_auto_targets)
from dataset1282_analysis.xu_geometry import CaseGeometry, PreparedCache
from dataset1282_analysis.xu_targets import TargetStore, build_targets, register, run
from test_xu_targets import MAPPING, on_disk_phantom


TEST_ROOT = Path('D:/projects/back/t17/auto-proximal')


class AutomaticProximalTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_anisotropic_mm_union_seed_components_and_unknown(self):
        shape = (50, 32, 32)
        seg = np.zeros(shape, np.int8)
        seg[0] = -1
        seg[2:7, 10:15, 10:15] = MAPPING['RV']
        seg[2:7, 20:25, 20:25] = MAPPING['LV']
        seg[7:45, 11:13, 11:13] = MAPPING['AO']
        seg[7:45, 21:23, 21:23] = MAPPING['PA']
        seg[7:10, 16:18, 16:18] = MAPPING['AO']
        for i, key in enumerate(('LA', 'RA', 'MYO')):
            seg[3:5, 2+i, 2:4] = MAPPING[key]
        geo = CaseGeometry(shape[::-1], (1., 1., 2.), (0, 1, 2),
                           tuple((0, n) for n in shape), shape)
        y, qc = auto_proximal_target(seg, geo, MAPPING, 5., 20.)
        self.assertEqual(qc['realized_radius_mm'], 6.)
        self.assertFalse(qc['reviewed_ostium'])
        self.assertEqual(qc['vessels']['AO']['ventricular_root_class'], MAPPING['RV'])
        self.assertEqual(qc['vessels']['PA']['ventricular_root_class'], MAPPING['LV'])
        self.assertEqual(qc['vessels']['AO']['root_distance_mm'], 2.)
        self.assertEqual(y[10, 11, 11], 6)
        self.assertEqual(y[11, 11, 11], 0)
        self.assertEqual(y[8, 16, 16], 0)
        self.assertTrue(np.all(y[seg == -1] == -1))
        self.assertTrue(np.all(np.isin(seg[y == 6], [MAPPING['AO'], MAPPING['PA']])))
        for record in qc['vessels'].values():
            self.assertLessEqual(record['maximum_selected_seed_distance_mm'], 6.)
            self.assertLess(record['selected_fraction'], 1.)
        for radius, margin in ((0., 20.), (30., -1.), (np.nan, 20.), (30., np.inf)):
            with self.assertRaises(ValueError):
                auto_proximal_target(seg, geo, MAPPING, radius, margin)

    def test_real_store_resume_tampering_and_existing_targets_unchanged(self):
        cache = on_disk_phantom(self.base / 'prepared')
        original = self.base / 'reviewed'
        build_targets(cache, SimpleNamespace(width_mm=1., reviews=None, resume=False), original)
        before = {str(p.relative_to(original)): sha256_file(p) for p in original.rglob('*') if p.is_file()}
        output = self.base / 'automatic'
        args = SimpleNamespace(radius_mm=30., margin_percent=20., resume=False)
        report = build_auto_targets(cache, args, output)
        self.assertEqual(report['available'], {'blood': 0, 'init': 2})
        self.assertEqual(set(report['qc']), {'train_a', 'dev_b'})
        store = TargetStore(output, cache)
        self.assertFalse(store.has('train_a', 'blood'))
        y = store.read('train_a', 'init')
        self.assertTrue(np.all(y[cache.read_label('train_a') == -1] == -1))
        self.assertEqual(set(np.unique(y)), set(range(-1, 7)))
        for branch in ('init64', 'init128'):
            side = int(branch[4:])
            self.assertEqual(store.read('train_a', 'init', shape=(side,) * 3).shape, (side,) * 3)
        args.resume = True
        self.assertEqual(build_auto_targets(cache, args, output), report)
        self.assertEqual(before, {str(p.relative_to(original)): sha256_file(p)
                                 for p in original.rglob('*') if p.is_file()})
        TargetStore(original, cache).read('train_a', 'blood')
        args.radius_mm = 31.
        with self.assertRaisesRegex(ValueError, 'identical'):
            build_auto_targets(cache, args, output)
        args.radius_mm = 30.
        with (output / 'cases/train_a/init.b2nd').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            build_auto_targets(cache, args, output)
        with self.assertRaisesRegex(ValueError, 'digest'):
            TargetStore(output, cache).read('train_a', 'init')

    def test_partial_eligibility_cli_and_provenance(self):
        meta = dict(present_classes=list(range(1, 11)), unknown_classes=[], known_absent_classes=[])
        self.assertTrue(auto_eligible(meta, MAPPING))
        for field in ('unknown_classes', 'known_absent_classes'):
            partial = dict(meta, present_classes=[x for x in range(1, 11) if x != MAPPING['AO']],
                           **{field: [MAPPING['AO']]})
            self.assertFalse(auto_eligible(partial, MAPPING))
        cache = on_disk_phantom(self.base / 'prepared')
        dataset = cache.run_dir / 'preprocessed/Dataset282_CHDSource'
        supervision = read_json(dataset / 'partial_supervision.json')
        supervision['dev_b']['present_classes'].remove(MAPPING['AO'])
        supervision['dev_b']['unknown_classes'] = [MAPPING['AO']]
        supervision['dev_b']['ambiguous_background_allowed'] = [0, MAPPING['AO']]
        atomic_json(dataset / 'partial_supervision.json', supervision)
        contract = read_json(dataset / 'training_contract.json')
        contract['artifacts']['partial_supervision.json'] = sha256_file(dataset / 'partial_supervision.json')
        atomic_json(dataset / 'training_contract.json', contract)
        qc_path = cache.run_dir / 'qc/dev_b.json'
        qc = read_json(qc_path)
        qc['supervision'] = supervision['dev_b']
        atomic_json(qc_path, qc)
        seg_path = dataset / 'cached/dev_b_seg.b2nd'
        volume = cache.read_label('train_a').copy()
        volume[volume == MAPPING['AO']] = -1
        array = blosc2.asarray(volume[None], urlpath=str(seg_path), mode='w')
        del array
        marker_path = cache.run_dir / 'commits/process-dev_b.json'
        marker = read_json(marker_path)
        for path in (qc_path, seg_path):
            marker['artifacts'][path.relative_to(cache.run_dir).as_posix()] = sha256_file(path)
        atomic_json(marker_path, marker)
        cache = PreparedCache(cache.run_dir, allow_technical_subset=True)
        parser = argparse.ArgumentParser()
        register(parser.add_subparsers(dest='command'))
        args = parser.parse_args(['build-auto-targets', '--prepared', str(cache.run_dir),
                                  '--output', str(self.base / 'automatic')])
        report = run(cache, args, args.output)
        self.assertEqual(report['available']['init'], 1)
        self.assertFalse(TargetStore(args.output, cache).has('dev_b', 'init'))
        manifest = read_json(args.output / 'targets.json')
        manifest['contract']['policy'] = 'changed'
        atomic_json(args.output / 'targets.json', manifest)
        atomic_json(args.output / 'target-request.json', manifest['contract'])
        with self.assertRaisesRegex(ValueError, 'policy'):
            TargetStore(args.output, cache)

    def test_actual_observation_required_after_resampling(self):
        cache = on_disk_phantom(self.base / 'prepared')
        path = cache.run_dir / 'preprocessed/Dataset282_CHDSource/cached/dev_b_seg.b2nd'
        volume = cache.read_label('dev_b').copy()
        volume[volume == MAPPING['AO']] = 0
        array = blosc2.asarray(volume[None], urlpath=str(path), mode='w')
        del array
        marker_path = cache.run_dir / 'commits/process-dev_b.json'
        marker = read_json(marker_path)
        marker['artifacts'][path.relative_to(cache.run_dir).as_posix()] = sha256_file(path)
        atomic_json(marker_path, marker)
        cache = PreparedCache(cache.run_dir, allow_technical_subset=True)
        output = self.base / 'automatic'
        report = build_auto_targets(cache, SimpleNamespace(radius_mm=30., margin_percent=20., resume=False), output)
        self.assertEqual(report['available']['init'], 1)
        self.assertIn('AO', report['qc']['dev_b']['skipped']['init'])
        self.assertFalse((output / 'cases/dev_b/init.b2nd').exists())

    def test_real_cpu_training_both_sizes(self):
        from dataset1282_analysis.xu_init import _epoch, _model
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        cache = on_disk_phantom(self.base / 'prepared')
        output = self.base / 'automatic'
        build_auto_targets(cache, SimpleNamespace(radius_mm=30., margin_percent=20., resume=False), output)
        store = TargetStore(output, cache)
        book = {'boxes': {n: ((0, 20),) * 3 for n in ('train_a', 'dev_b')}}
        for branch in ('init64', 'init128'):
            with self.subTest(branch=branch):
                model = _model(branch, 'smoke')
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                before = model.head.weight.detach().clone()
                loss = _epoch(model, ['train_a'], cache, store, book, torch.device('cpu'),
                              torch.Generator().manual_seed(7), optimizer)
                validation = _epoch(model, ['dev_b'], cache, store, book, torch.device('cpu'))
                self.assertTrue(np.isfinite([loss, validation]).all())
                self.assertFalse(torch.equal(before, model.head.weight))


if __name__ == '__main__':
    unittest.main()
