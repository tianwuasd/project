"""Synthetic paired-cache contract and segmentation grid tests; no model download."""
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aaoca_seg import executor, scheduler
from aaoca_seg.images import inspect_image
from aaoca_seg.preprocessed import candidates, digest, load, validate_case
from aaoca_seg.schema import LABELS
from aaoca_seg.service_state import file_lock, read_json, write_json
from aaoca_seg.terminal import Panel
from aaoca_seg.workbench import Workbench


class PreprocessedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'preprocessed/20261023-synthetic'
        self.work = Workbench(self.base / 'seg-runtime')
        self.key = 'abc123'
        folder = self.root / 'cases' / self.key
        folder.mkdir(parents=True)
        (self.root / '.preprocess.lock').write_bytes(b'0')
        self.hu_data = np.linspace(-1600, 1900, 512, dtype=np.float32).reshape(8, 8, 8)
        self.affine = np.array([[.6, 0, 0, 10], [0, .7, 0, -20], [0, 0, .4, 30], [0, 0, 0, 1.]])
        for name, data, desc in [('image_hu.nii.gz', self.hu_data, b'aaoca_preprocessing HU; resampled; unclipped'),
                                 ('image.nii.gz', np.clip(self.hu_data, -1000, 1000)/1000, b'aaoca_preprocessing v1; normalized [-1,1]; NOT HU')]:
            image = nib.Nifti1Image(data, self.affine)
            image.header.set_xyzt_units('mm')
            image.header['descrip'] = desc
            nib.save(image, folder / name)
        self.config = {'hu_companion': 'resampled_unclipped_HU_v1', 'input_units': 'HU',
                       'formula': 'clip(HU, -1000, 1000) / 1000'}
        self.record_path = folder / 'record.json'
        self.record = {'schema': 2, 'status': 'complete', 'case_id': self.key,
                       'config': self.config, 'config_sha256': '1'*64,
                       'output': f'cases/{self.key}/image.nii.gz',
                       'hu_output': f'cases/{self.key}/image_hu.nii.gz',
                       'output_sha256': digest(folder / 'image.nii.gz'),
                       'hu_output_sha256': digest(folder / 'image_hu.nii.gz'),
                       'output_shape': [8, 8, 8], 'output_affine': self.affine.tolist()}
        write_json(self.record_path, self.record)
        write_json(self.root / 'dataset.json', {'config': self.config, 'config_sha256': '1'*64})
        write_json(self.root / 'progress.json', {'state': 'completed', 'total': 1, 'pending': 0, 'failed': 0})
        write_json(self.root / 'manifest.json', [{'case_id': self.key, 'state': 'completed',
                   'output': self.record['output'], 'hu_output': self.record['hu_output']}])

    def test_select_completed_pairs_and_freeze_provenance(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        case = load(self.root)[0]
        validate_case(case)
        job = self.work.create('预处理分割', ['plan'], preprocessed=self.root)
        request = read_json(job / 'request.json')
        self.assertEqual(request['preprocessing_source'], str(self.root))
        self.assertEqual(request['cases'][0]['image'], str(self.root / self.record['hu_output']))
        self.assertEqual(request['cases'][0]['preprocessing'], case['preprocessing'])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_legacy_normalized_only_cache_is_not_inverted(self):
        write_json(self.root / 'dataset.json', {'config': {'input_units': 'HU'}})
        with self.assertRaisesRegex(ValueError, '未截断 HU'):
            load(self.root)

    def test_direct_normalized_file_rejected_even_if_manifest_claims_hu(self):
        with self.assertRaisesRegex(ValueError, '不能把'):
            inspect_image(self.root / self.record['output'])

    def test_incomplete_or_active_source_is_rejected(self):
        with file_lock(self.root / '.preprocess.lock'):
            with self.assertRaises(ValueError): load(self.root)
        runtime = self.root.with_name(self.root.name + '.runtime')
        write_json(runtime / 'worker.json', {'state': 'checking'})
        with self.assertRaises(ValueError): load(self.root)
        write_json(runtime / 'worker.json', {'state': 'completed'})
        write_json(self.root / 'progress.json', {'state': 'completed_with_errors', 'total': 1})
        with self.assertRaises(ValueError): load(self.root)

    def test_missing_corrupt_and_changed_records_are_rejected(self):
        case = load(self.root)[0]
        normalized = self.root / self.record['output']
        original = normalized.read_bytes()
        normalized.write_bytes(original + b'corruption')
        with self.assertRaisesRegex(ValueError, 'SHA256'): validate_case(case)
        normalized.write_bytes(original)
        self.record_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, '记录'): validate_case(case)
        write_json(self.record_path, self.record)
        normalized.unlink()
        with self.assertRaises(ValueError): load(self.root)

    def test_geometry_mismatch_rejected_even_with_new_hash(self):
        path = self.root / self.record['output']
        old = nib.load(path)
        image = nib.Nifti1Image(old.get_fdata(dtype=np.float32), np.diag([1., 1., 1., 1.]), old.header)
        nib.save(image, path)
        self.record['output_sha256'] = digest(path)
        write_json(self.record_path, self.record)
        with self.assertRaisesRegex(ValueError, '网格'): validate_case(load(self.root)[0])

    def test_record_path_escape_and_duplicate_manifest_are_rejected(self):
        self.record['hu_output'] = '../../outside.nii.gz'
        write_json(self.record_path, self.record)
        with self.assertRaises(ValueError): load(self.root)
        self.record['hu_output'] = f'cases/{self.key}/image_hu.nii.gz'
        write_json(self.record_path, self.record)
        rows = read_json(self.root / 'manifest.json')
        write_json(self.root / 'manifest.json', rows*2)
        write_json(self.root / 'progress.json', {'state': 'completed', 'total': 2})
        with self.assertRaises(ValueError): load(self.root)

    def test_output_nested_in_producer_is_rejected(self):
        with self.assertRaises(ValueError):
            self.work.create('坏路径', ['plan'], {**self.work.settings, 'results': str(self.root / 'seg')}, preprocessed=self.root)
        self.assertEqual(self.work.entries(), [])

    def test_parent_historical_input_keeps_paired_contract(self):
        parent = self.work.create('上游', ['plan'], preprocessed=self.root)
        child = self.work.create('下游', ['plan'], input_job=parent.name)
        self.assertEqual(read_json(child / 'request.json')['cases'][0]['preprocessing'],
                         read_json(parent / 'request.json')['cases'][0]['preprocessing'])

    def test_menu_default_preprocessed_selection_can_cancel_without_submission(self):
        panel = Panel(self.work.root, ask=Mock(side_effect=['合成任务', '', str(self.root.parent), '1', '1', '', 'no']), emit=lambda _: None)
        panel.new()
        self.assertEqual(self.work.entries(), [])
        self.assertFalse((self.work.root / 'seg-settings.json').exists())
        self.assertEqual(candidates(self.root.parent), [self.root])

    def test_runtime_inside_preprocessing_is_rejected_without_writing_source(self):
        work = Workbench(self.root / 'seg-runtime')
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with self.assertRaises(ValueError):
            work.create('nested runtime', ['plan'], preprocessed=self.root)
        self.assertEqual(before, {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_invalid_progress_metadata_is_diagnosed_without_menu_crash(self):
        write_json(self.root / 'progress.json', [])
        panel = Panel(self.work.root, ask=Mock(side_effect=[str(self.root), '1']), emit=lambda _: None)
        with self.assertRaises(ValueError):
            panel.choose_preprocessed()

    def test_readonly_producer_lock_allows_consumer_validation(self):
        lock = self.root / '.preprocess.lock'
        previous = lock.stat().st_mode
        lock.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            validate_case(load(self.root)[0])
        finally:
            lock.chmod(previous)

    def test_full_cpu_validation_precedes_gpu_wait(self):
        job = self.work.create('核验', ['plan'], preprocessed=self.root)
        (self.root / self.record['hu_output']).write_bytes(b'corrupt')
        probe = Mock(side_effect=AssertionError('GPU must not be queried'))
        result = scheduler.run(job, allocation={'cpu_limit': 2, 'case_slots': 1, 'threads': 1,
                              'memory_bytes': 8*2**30, 'io_slots': 1}, probe=probe, backend={'versions': {}})
        self.assertEqual(result['failed'], 1)
        probe.assert_not_called()

    def test_mock_model_reads_unclipped_hu_and_labels_align_to_normalized_ct(self):
        case = load(self.root)[0]
        attempt = self.base / 'seg-output/cases/abc123/attempt-1'
        def fake_model(command, log, stop, registry):
            source = nib.load(command[command.index('-i')+1])
            np.testing.assert_array_equal(source.get_fdata(), self.hu_data)
            root = Path(command[command.index('-o')+1]); root.mkdir(parents=True, exist_ok=True)
            task = command[command.index('--task')+1]
            for label, name, _, model in LABELS:
                if model == task:
                    data = np.zeros(source.shape, dtype=np.uint8)
                    data[label % 7, 3, 4] = 1
                    nib.save(nib.Nifti1Image(data, source.affine, source.header), root / (name + '.nii.gz'))
        with patch.object(executor, 'execute_model', side_effect=fake_model):
            executor.infer(case, attempt, 1, 'cpu', self.base / 'stop', {'synthetic': True})
        cp = attempt / 'inference.json'
        destination = self.base / 'merged'
        result = executor.merge(case, cp, destination)
        labels = nib.load(destination / 'aaoca_11_labels.nii.gz')
        reference = nib.load(case['preprocessing']['normalized_image'])
        self.assertEqual(labels.shape, reference.shape)
        np.testing.assert_array_equal(labels.affine, reference.affine)
        self.assertEqual(result['label_reference_image'], str(self.root / self.record['output']))
        self.assertTrue(executor.completed(destination / 'result.json', case, cp))


if __name__ == '__main__':
    unittest.main()
