"""Portable preprocessing tests. All images and identifiers are synthetic."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import map_coordinates
    import psutil
except ImportError as exc:
    raise unittest.SkipTest('Install requirements-preprocess.txt to run optional preprocessing tests') from exc

from aaoca_preprocessing.config import Config
from aaoca_preprocessing.image import normalize_hu, output_grid, inspect_image
from aaoca_preprocessing.io import RunLock, discover, file_hash
from aaoca_preprocessing.pipeline import plan, run


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / 'input'
        self.out = self.root / 'output'
        self.src.mkdir()

    def image(self, name='case_a.nii.gz', data=None, affine=None, units='mm'):
        if data is None:
            data = np.linspace(-1600, 1800, 210, dtype=np.float32).reshape(5, 6, 7)
        image = nib.Nifti1Image(data, np.diag([.7, .9, 1.3, 1]) if affine is None else affine)
        image.header.set_xyzt_units(units)
        path = self.src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image, path)
        return path

    def record(self):
        return json.loads(next((self.out / 'cases').glob('*/record.json')).read_text())

    def test_hu_endpoints_and_nonfinite(self):
        source = np.array([-2000, -1000, -500, 0, 500, 1000, 1800], dtype=np.float64)
        actual = normalize_hu(source)
        np.testing.assert_array_equal(actual, [-1, -1, -.5, 0, .5, 1, 1])
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(source[0], -2000)
        for value in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                normalize_hu(np.array([value]))

    def test_grid_preserves_all_cell_corners_and_centre(self):
        affine = np.array([[.7, -.2, .1, 34], [.2, .8, 0, -18], [0, 0, 1.5, 12], [0, 0, 0, 1]])
        shape, target, mapping = output_grid((7, 9, 11), affine, (.5, .5, .5))
        for corner in np.ndindex(2, 2, 2):
            a = np.array(corner) * np.array([7, 9, 11]) - .5
            b = np.array(corner) * np.array(shape) - .5
            np.testing.assert_allclose(affine @ np.r_[a, 1], target @ np.r_[b, 1])
        np.testing.assert_allclose(affine @ np.r_[(np.array([7, 9, 11])-1)/2, 1],
                                   target @ np.r_[(np.array(shape)-1)/2, 1])
        np.testing.assert_allclose(target, affine @ mapping)

    def test_native_run_preserves_source_and_reuses_verified_output(self):
        source = self.image()
        before = file_hash(source)
        result = run(self.src, self.out, Config())
        self.assertEqual((result['completed'], result['failed']), (1, 0))
        record = self.record()
        image = nib.load(self.out / record['output'])
        np.testing.assert_array_equal(image.get_fdata(), normalize_hu(nib.load(source).get_fdata()))
        self.assertEqual(image.get_data_dtype(), np.dtype('float32'))
        self.assertEqual(image.header.get_xyzt_units()[0], 'mm')
        self.assertEqual(file_hash(source), before)
        self.assertEqual(run(self.src, self.out, Config())['reused'], 1)

    def test_nifti_scaling_applied_once(self):
        source = self.image(data=np.arange(24, dtype=np.int16).reshape(2, 3, 4))
        image = nib.load(source)
        image.header.set_slope_inter(100, -1000)
        nib.save(image, source)
        expected = normalize_hu(nib.load(source).get_fdata())
        self.assertEqual(run(self.src, self.out, Config())['failed'], 0)
        actual = nib.load(self.out / self.record()['output'])
        np.testing.assert_array_equal(actual.get_fdata(), expected)
        self.assertEqual((actual.dataobj.slope, actual.dataobj.inter), (1, 0))

    def test_oblique_flip_resampling_matches_independent_sampler(self):
        x, y, z = np.indices((5, 7, 9))
        data = (300*x + 23*y - 55*z - 700).astype(np.float32)
        affine = np.array([[-.8, .15, 0, 23], [-.2, -.9, 0, 42], [0, 0, 1.6, -4], [0, 0, 0, 1]])
        self.image(data=data, affine=affine)
        for block, output in [(1, self.out), (5, self.root / 'other')]:
            result = run(self.src, output, Config(target_spacing=(.5, .5, .5), chunk_depth=block))
            self.assertEqual(result['failed'], 0)
        a = nib.load(self.out / self.record()['output'])
        positions = np.indices(a.shape).reshape(3, -1)
        mapped = (np.linalg.inv(affine) @ a.affine @ np.vstack([positions, np.ones(positions.shape[1])]))[:3]
        expected = normalize_hu(map_coordinates(data, mapped, order=1, mode='nearest', prefilter=False))
        np.testing.assert_allclose(a.get_fdata().reshape(-1), expected, atol=2e-6)
        b = nib.load(next((self.root / 'other' / 'cases').glob('*/image.nii.gz')))
        np.testing.assert_array_equal(a.get_fdata(), b.get_fdata())
        self.assertEqual(nib.aff2axcodes(a.affine), ('R', 'A', 'S'))
        self.assertEqual(int(a.header['qform_code']), 0)  # shear is represented by sform

    def test_invalid_geometry_units_and_dimensions(self):
        for units, data in [('unknown', np.zeros((2, 3, 4))), ('mm', np.zeros((2, 3, 4, 2)))]:
            with self.assertRaises(ValueError):
                inspect_image(self.image(units=units, data=data), Config())
        source = self.image()
        image = nib.load(source)
        image.set_qform(np.diag([1, 1, 2, 1]), code=1)
        nib.save(image, source)
        with self.assertRaisesRegex(ValueError, 'qform'):
            inspect_image(source, Config())

    def test_failure_isolation_and_invalid_pixels(self):
        self.image('good.nii.gz')
        self.image('bad.nii.gz', data=np.full((2, 3, 4), np.nan, dtype=np.float32))
        result = run(self.src, self.out, Config())
        self.assertEqual((result['completed'], result['failed']), (1, 1))
        self.assertEqual(len(list((self.out / 'cases').glob('*/record.json'))), 1)
        self.assertEqual(result['state'], 'completed_with_errors')

    def test_changed_source_and_parameters_preserve_previous_results(self):
        source = self.image()
        run(self.src, self.out, Config())
        image = self.out / self.record()['output']
        before = file_hash(image)
        self.image(data=np.zeros((5, 6, 7), dtype=np.float32))
        self.assertEqual(run(self.src, self.out, Config())['failed'], 1)
        self.assertEqual(file_hash(image), before)
        with self.assertRaises(ValueError):
            run(self.src, self.out, Config(target_spacing=(1, 1, 1)))
        self.assertTrue(source.exists())

    def test_corrupt_output_is_not_reused_or_overwritten(self):
        self.image()
        run(self.src, self.out, Config())
        image = self.out / self.record()['output']
        image.write_bytes(b'corrupt')
        result = run(self.src, self.out, Config())
        self.assertEqual((result['reused'], result['failed']), (0, 1))
        self.assertEqual(image.read_bytes(), b'corrupt')

    def test_manifest_filters_dataset_and_rejects_escape_or_duplicates(self):
        source = self.image()
        manifest = self.root / 'manifest.json'
        rows = [{'type': 'AAOCA', 'destination': source.name, 'sha256': file_hash(source)},
                {'type': 'TAPVC', 'destination': 'not-present.nii.gz'}]
        manifest.write_text(json.dumps(rows))
        self.assertEqual(len(discover(self.src, manifest)), 1)
        for bad in [[rows[0], rows[0]], [{'type': 'AAOCA', 'destination': '../outside.nii.gz'}]]:
            manifest.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                discover(self.src, manifest)

    def test_discovery_excludes_derived_and_label_images(self):
        self.image('subject_a/native/ct.nii.gz')
        self.image('subject_a/native/研究版_目标0.5mm/ct.nii.gz')
        self.image('subject_a/native/ct_label.nii.gz')
        self.image('subject_a/native/ct_mask.nii.gz')
        self.assertEqual(len(discover(self.src)), 1)

    def test_input_output_overlap_and_memory_guard(self):
        self.image()
        for output in [self.src, self.src / 'result', self.root]:
            with self.assertRaises(ValueError):
                run(self.src, output, Config())
        result = run(self.src, self.out, Config(max_case_gib=1e-7))
        self.assertEqual(result['failed'], 1)
        self.assertFalse(list((self.out / 'cases').glob('*/image.nii.gz')))

    def test_plan_is_readonly_and_reports_size(self):
        self.image()
        result = plan(self.src, Config())
        self.assertEqual((result['total'], result['invalid']), (1, 0))
        self.assertGreater(result['output_bytes_uncompressed'], 0)
        self.assertFalse(self.out.exists())

    def test_lock_releases_without_deleting_lock_file(self):
        self.out.mkdir()
        with RunLock(self.out):
            with self.assertRaises(RuntimeError):
                with RunLock(self.out):
                    pass
        with RunLock(self.out):
            self.assertTrue((self.out / '.preprocess.lock').exists())

    def test_config_validation(self):
        for kwargs in [{'target_spacing': (0, 1, 1)}, {'target_spacing': (1, 2)},
                       {'target_spacing': (float('nan'), 1, 1)}, {'workers': 0},
                       {'chunk_depth': 0}, {'max_case_gib': -1}]:
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_cli_from_other_directory(self):
        self.image()
        entry = Path(__file__).resolve().parents[1] / 'preprocess_aaoca.py'
        for arguments in [['--help'], ['plan', '--input', str(self.src)],
                          ['run', '--input', str(self.src), '--output', str(self.out)],
                          ['status', '--output', str(self.out)]]:
            result = subprocess.run([sys.executable, str(entry), *arguments], cwd=self.root,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_parallel_cases_have_independent_records_and_can_resume(self):
        self.image('a.nii.gz')
        self.image('b.nii')
        self.image('c.nii.gz')
        result = run(self.src, self.out, Config(workers=2))
        self.assertEqual((result['completed'], result['failed'], result['pending']), (3, 0, 0))
        self.assertEqual(run(self.src, self.out, Config(workers=1))['reused'], 3)

    def test_failed_save_leaves_no_completed_case_and_retry_succeeds(self):
        self.image()
        with patch('aaoca_preprocessing.image.validate_written', side_effect=ValueError('simulated read-back failure')):
            result = run(self.src, self.out, Config())
        self.assertEqual(result['failed'], 1)
        self.assertFalse(list((self.out / 'cases').iterdir()))
        self.assertFalse(list((self.out / '.staging').iterdir()))
        self.assertFalse(list((self.out / '.scratch').iterdir()))
        self.assertEqual(run(self.src, self.out, Config())['completed'], 1)

    def test_source_hash_from_manifest_is_enforced(self):
        source = self.image()
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps([{'input': source.name, 'sha256': '0' * 64}]))
        result = run(self.src, self.out, Config(), manifest)
        self.assertEqual(result['failed'], 1)
        self.assertFalse(list((self.out / 'cases').iterdir()))

    def test_truncated_gzip_is_not_published(self):
        source = self.image()
        source.write_bytes(source.read_bytes()[:-8])
        result = run(self.src, self.out, Config())
        self.assertEqual(result['failed'], 1)
        self.assertFalse(list((self.out / 'cases').iterdir()))

    def test_normalized_input_is_rejected(self):
        self.image()
        run(self.src, self.out, Config())
        generated = self.out / self.record()['output']
        with self.assertRaisesRegex(ValueError, 'already normalized'):
            inspect_image(generated, Config())

    def test_native_orientation_permutation_is_lossless(self):
        data = np.arange(60, dtype=np.float32).reshape(3, 4, 5) * 20 - 500
        affine = np.array([[0, -.9, 0, 23], [0, 0, 1.2, -4], [.7, 0, 0, 11], [0, 0, 0, 1]])
        source = self.image(data=data, affine=affine)
        run(self.src, self.out, Config())
        output = nib.load(self.out / self.record()['output'])
        canonical = nib.as_closest_canonical(nib.load(source))
        np.testing.assert_array_equal(output.get_fdata(), normalize_hu(canonical.get_fdata()))
        np.testing.assert_allclose(output.affine, canonical.affine, atol=1e-6)

    def test_unrelated_output_files_are_preserved(self):
        self.image()
        self.out.mkdir()
        unrelated = self.out / 'notes.txt'
        unrelated.write_text('keep me')
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            run(self.src, self.out, Config())
        self.assertEqual(unrelated.read_text(), 'keep me')

    def test_disk_shortage_records_failed_status(self):
        self.image()
        with patch('aaoca_preprocessing.pipeline.shutil.disk_usage', return_value=type('Space', (), {'free': 0})()):
            with self.assertRaisesRegex(ValueError, 'disk'):
                run(self.src, self.out, Config())
        status = json.loads((self.out / 'progress.json').read_text())
        self.assertEqual(status['state'], 'failed')
        self.assertFalse(list((self.out / 'cases').iterdir()))

    def test_source_change_during_processing_is_not_published(self):
        source = self.image()
        from aaoca_preprocessing.pipeline import preprocess_image

        def changed(*args, **kwargs):
            result = preprocess_image(*args, **kwargs)
            source.write_bytes(source.read_bytes() + b'changed')
            return result

        with patch('aaoca_preprocessing.pipeline.preprocess_image', side_effect=changed):
            result = run(self.src, self.out, Config())
        self.assertEqual(result['failed'], 1)
        self.assertIn('changed while processing', result['errors'][0]['error'])
        self.assertFalse(list((self.out / 'cases').iterdir()))


if __name__ == '__main__':
    unittest.main()
