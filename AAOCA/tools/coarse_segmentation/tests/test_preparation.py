import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaoca_seg.preparation import prepare, source_stat, validate_prepared
from aaoca_seg.pipeline import sha256
from aaoca_seg.executor import restore_empty_mask_units
from aaoca_seg.schema import LABELS


@unittest.skipUnless(os.environ.get('AAOCA_SEG_TEST_ROOT'), '需要显式设置已批准的测试目录')
class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AAOCA_SEG_TEST_ROOT'])
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.nii.gz'
        image = nib.Nifti1Image(np.arange(9*10*11, dtype=np.float32).reshape(9, 10, 11),
                               np.diag([-.7, .8, 1.2, 1.]))
        image.header.set_xyzt_units('mm')
        nib.save(image, self.source)
        self.case = {'case_id': 'sample', 'image': str(self.source), 'source_stat': source_stat(self.source)}
        self.attempt = self.root / 'case/attempt-a'
        self.stop = self.root / 'stop'

    def tearDown(self):
        self.temp.cleanup()

    def test_lossless_bytes_geometry_and_cached_reuse(self):
        before = sha256(self.source)
        record = prepare(self.case, self.attempt, self.stop)
        raw = Path(record['image'])
        with gzip.open(self.source, 'rb') as stream:
            self.assertEqual(raw.read_bytes(), stream.read())
        np.testing.assert_array_equal(nib.load(raw).get_fdata(), nib.load(self.source).get_fdata())
        np.testing.assert_array_equal(nib.load(raw).affine, nib.load(self.source).affine)
        modified = raw.stat().st_mtime_ns
        self.assertEqual(prepare(self.case, self.attempt, self.stop), record)
        self.assertEqual(raw.stat().st_mtime_ns, modified)
        self.assertEqual(sha256(self.source), before)

    def test_changed_cache_rejected(self):
        record = prepare(self.case, self.attempt, self.stop)
        with Path(record['image']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, '路径或哈希'):
            validate_prepared(self.case, self.attempt / 'prepared.json')

    def test_changed_receipt_rejected(self):
        prepare(self.case, self.attempt, self.stop)
        receipt = self.attempt / 'prepared.json'
        record = json.loads(receipt.read_text(encoding='utf-8'))
        record['geometry']['shape'][0] += 1
        receipt.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '记录完整性'):
            validate_prepared(self.case, receipt)

    def test_changed_source_rejected(self):
        prepare(self.case, self.attempt, self.stop)
        with self.source.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, '原始输入发生变化'):
            validate_prepared(self.case, self.attempt / 'prepared.json')

    def test_stop_removes_partial_and_never_commits_prepared(self):
        self.stop.touch()
        with self.assertRaises(KeyboardInterrupt):
            prepare(self.case, self.attempt, self.stop)
        self.assertFalse((self.attempt / 'input.partial.nii').exists())
        self.assertFalse((self.attempt / 'prepared.json').exists())

    def test_interrupted_partial_is_removed_before_real_preparation(self):
        self.attempt.mkdir(parents=True)
        (self.attempt / 'input.partial.nii').write_bytes(b'interrupted')
        record = prepare(self.case, self.attempt, self.stop)
        self.assertTrue(Path(record['image']).is_file())
        self.assertFalse((self.attempt / 'input.partial.nii').exists())

    def empty_masks(self):
        source = nib.load(self.source)
        for _, name, _, task in LABELS:
            folder = self.attempt / task
            folder.mkdir(parents=True, exist_ok=True)
            image = nib.Nifti1Image(np.zeros(source.shape, dtype=np.uint8), source.affine)
            image.header.set_xyzt_units('mm')
            nib.save(image, folder / f'{name}.nii.gz')
        _, name, _, task = LABELS[0]
        target = self.attempt / task / f'{name}.nii.gz'
        image = nib.load(target)
        image.header.set_xyzt_units('unknown')
        nib.save(image, target)
        (self.attempt / f'{task}.log').write_text(
            'INFO: Crop is empty. Returning empty segmentation.\n', encoding='utf-8')
        return target

    def test_confirmed_empty_model_output_recovers_units_without_changing_grid(self):
        target = self.empty_masks()
        before = nib.load(target)
        expected = before.affine.copy()
        changed = restore_empty_mask_units(self.source, self.attempt)
        self.assertEqual(len(changed), 1)
        actual = nib.load(target)
        self.assertEqual(actual.header.get_xyzt_units()[0], 'mm')
        np.testing.assert_array_equal(actual.affine, expected)
        self.assertFalse(np.any(actual.get_fdata()))

    def test_unknown_unit_nonempty_mask_is_rejected(self):
        target = self.empty_masks()
        image = nib.load(target)
        data = np.asanyarray(image.dataobj).copy()
        data[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(data, image.affine, image.header), target)
        with self.assertRaisesRegex(ValueError, '非空预测'):
            restore_empty_mask_units(self.source, self.attempt)


if __name__ == '__main__':
    unittest.main()
