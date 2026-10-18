"""Synthetic-only regression checks; no real identifiers or credentials."""
import tempfile
import unittest
from pathlib import Path
import zipfile
import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pyzipper

from archive import extract_labels, safe_members, sha256
from prepare import inventory, process_pair, read_volume, reusable, run, RunLock


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.out = self.root / 'output'
        self.source.mkdir()
        (self.out / 'extracted_labels').mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, fractional=False, shifted=False):
        values = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
        image = self.source / 'synthetic_A_image.nii.gz'
        label = self.out / 'extracted_labels' / 'synthetic_A_label.nii.gz'
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(values, affine), image)
        labels = np.zeros(values.shape, dtype=np.float32)
        labels[1:, :, :] = 0.5 if fractional else 2
        if shifted:
            affine[0, 3] = 5
        nib.save(nib.Nifti1Image(labels, affine), label)
        return image, label, inventory(self.source, label.parent)[0]

    def test_full_pipeline_and_resume_hash_validation(self):
        image, label, row = self.pair()
        summary = run(self.source, self.out, workers=1)
        self.assertEqual(summary['passed'], 1)
        self.assertEqual(summary['label_voxel_counts'], {'0': 20, '2': 40})
        self.assertEqual(sha256(image), sha256(next((self.out / 'native/images').glob('*.nii.gz'))))
        self.assertEqual(sha256(label), sha256(next((self.out / 'native/labels').glob('*.nii.gz'))))
        again = run(self.source, self.out, workers=1, rehash=True)
        self.assertEqual(again['reused'], 1)
        self.assertFalse(again['normalized'])

    def test_fractional_labels_fail_without_publication(self):
        _, _, row = self.pair(fractional=True)
        result = process_pair(row, self.out)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('integer', ';'.join(result['errors']))
        self.assertFalse((self.out / 'native').exists())

    def test_grid_mismatch_preserved_in_review_collection(self):
        _, _, row = self.pair(shifted=True)
        result = process_pair(row, self.out)
        self.assertEqual(result['status'], 'review_required')
        self.assertIn('IMAGE_LABEL_AFFINE_MISMATCH', result['grid_issues'])
        self.assertTrue((self.out / 'review_required/images').exists())
        self.assertFalse((self.out / 'native').exists())

    def test_truncated_gzip_is_rejected(self):
        image, _, _ = self.pair()
        image.write_bytes(image.read_bytes()[:-6])
        with self.assertRaises(Exception):
            read_volume(image)

    def test_incomplete_pairing(self):
        _, label, _ = self.pair()
        label.unlink()
        row = inventory(self.source, label.parent)[0]
        self.assertEqual(row['pairing_errors'], ['MISSING_LABEL'])

    def test_changed_source_invalidates_resume(self):
        image, _, row = self.pair()
        result = process_pair(row, self.out)
        self.assertTrue(reusable(row, result, rehash=True))
        image.write_bytes(image.read_bytes() + b'changed')
        self.assertFalse(reusable(row, result))

    def test_existing_different_output_is_not_overwritten(self):
        _, _, row = self.pair()
        target = self.out / 'native/images' / (row['case_id'] + '_0000.nii.gz')
        target.parent.mkdir(parents=True)
        target.write_bytes(b'keep this existing result')
        result = process_pair(row, self.out)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(target.read_bytes(), b'keep this existing result')

    def test_project_launcher_and_operating_system_lock(self):
        self.pair()
        project = self.root / 'project'
        project.mkdir()
        (project / 'project.local.json').write_text(json.dumps({'chd_reference_root': str(self.root)}), encoding='utf-8')
        (self.out / 'extraction.json').write_text('{"files": [], "total": 0}', encoding='utf-8')
        command = [sys.executable, str(Path(__file__).with_name('prepare.py')), '--project-root', str(project),
                   '--source', str(self.source), '--output', str(self.out), '--workers', '1']
        first = subprocess.run(command, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr.decode(errors='replace'))
        with RunLock(self.out / '.run.lock'):
            second = subprocess.run(command, capture_output=True)
            self.assertNotEqual(second.returncode, 0)

    def test_duplicate_names_not_silently_selected(self):
        image, label, _ = self.pair()
        other = self.source / 'second'
        other.mkdir()
        (other / image.name).write_bytes(image.read_bytes())
        self.assertIn('DUPLICATE_IMAGE_NAME', inventory(self.source, label.parent)[0]['pairing_errors'])

    def test_archive_paths_and_duplicate_basenames(self):
        for name in ['../outside.nii.gz', '/outside.nii.gz', 'C:/outside.nii.gz', '..\\outside.nii.gz']:
            with self.assertRaises(ValueError):
                safe_members([zipfile.ZipInfo(name)])
        with self.assertRaises(ValueError):
            safe_members([zipfile.ZipInfo('a/test.nii.gz'), zipfile.ZipInfo('b/TEST.nii.gz')])

    def test_aes_password_and_resumable_extraction(self):
        archive = self.root / 'labels.zip'
        synthetic_password = b'test-only-key'
        with pyzipper.AESZipFile(archive, 'w', encryption=pyzipper.WZ_AES) as z:
            z.setpassword(synthetic_password)
            z.writestr('labels/synthetic_label.nii.gz', b'synthetic payload')
        with self.assertRaises(RuntimeError):
            extract_labels(archive, self.out, b'wrong-test-key')
        self.assertFalse((self.out / 'extracted_labels/synthetic_label.nii.gz').exists())
        result = extract_labels(archive, self.out, synthetic_password)
        self.assertEqual(result['total'], 1)
        resumed = extract_labels(archive, self.out)
        self.assertEqual(result['files'], resumed['files'])


if __name__ == '__main__':
    unittest.main()
