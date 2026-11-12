import os
import json
from pathlib import Path
import tempfile
import sys
import unittest

import nibabel as nib
import numpy as np
import slicerio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aaoca_seg.segmentation import export_segmentation
from aaoca_seg.schema import LABELS
from aaoca_seg.executor import commit_editable
from aaoca_seg.pipeline import sha256


@unittest.skipUnless(os.environ.get('AAOCA_SEG_TEST_ROOT'), '需要显式设置格式测试的专用目录')
class EditableFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AAOCA_SEG_TEST_ROOT'])
        self.root = Path(self.temp.name)
        self.affine = np.array([[-.7, .1, 0, 11], [0, .8, .05, -7], [0, 0, 1.2, 21], [0, 0, 0, 1]])
        self.source = self.root / 'format-sample.nii.gz'
        image = nib.Nifti1Image(np.arange(9 * 10 * 11, dtype=np.float32).reshape(9, 10, 11), self.affine)
        image.header.set_xyzt_units('mm')
        image.header['descrip'] = b'SYNTHETIC_SOURCE_METADATA'
        nib.save(image, self.source)

    def tearDown(self):
        self.temp.cleanup()

    def masks(self, overlap):
        for label, name, _, task in LABELS:
            folder = self.root / 'masks' / task
            folder.mkdir(parents=True, exist_ok=True)
            data = np.zeros((9, 10, 11), dtype=np.uint8)
            data[1, label % 10, label // 10] = 1
            if overlap:
                data[4, 4, 4] = 1
            image = nib.Nifti1Image(data, self.affine)
            image.header.set_xyzt_units('mm')
            nib.save(image, folder / f'{name}.nii.gz')

    def test_overlaps_names_colors_and_oblique_geometry_with_independent_reader(self):
        self.masks(True)
        path, report = export_segmentation(self.source, self.root / 'masks', self.root / 'output')
        decoded = slicerio.read_segmentation(str(path))
        self.assertEqual(len(decoded['segments']), 11)
        self.assertEqual(report['layers'], 11)
        self.assertTrue(report['all_voxels_roundtrip_equal'])
        expected_lps = np.diag([-1., -1., 1., 1.]) @ self.affine
        np.testing.assert_allclose(decoded['ijkToLPS'], expected_lps, atol=1e-6)
        for segment, (label, name, _, _) in zip(decoded['segments'], LABELS):
            self.assertEqual(segment['name'], name)
            self.assertEqual(segment['labelValue'], label)
            self.assertEqual(decoded['voxels'][segment['layer'], 4, 4, 4], label)
            self.assertEqual(len(segment['color']), 3)
        self.assertNotIn(b'SYNTHETIC_SOURCE_METADATA', path.read_bytes())

    def test_disjoint_structures_use_one_layer(self):
        self.masks(False)
        path, report = export_segmentation(self.source, self.root / 'masks', self.root / 'output')
        decoded = slicerio.read_segmentation(str(path))
        self.assertEqual(report['layers'], 1)
        self.assertEqual(decoded['voxels'].shape, (9, 10, 11))
        for label, _, _, _ in LABELS:
            self.assertEqual(decoded['voxels'][1, label % 10, label // 10], label)

    def test_changed_geometry_rejected_without_final_output(self):
        self.masks(False)
        _, name, _, task = LABELS[0]
        path = self.root / 'masks' / task / f'{name}.nii.gz'
        original = nib.load(path)
        shifted = self.affine.copy()
        shifted[0, 3] += 1
        nib.save(nib.Nifti1Image(np.asanyarray(original.dataobj), shifted, original.header), path)
        with self.assertRaisesRegex(ValueError, '空间网格'):
            export_segmentation(self.source, self.root / 'masks', self.root / 'output')
        self.assertFalse((self.root / 'output/aaoca_11.seg.nrrd').exists())

    def test_result_pointer_committed_before_cleanup_failure(self):
        self.masks(True)
        attempt = self.root / 'case/attempt-a'
        path, report = export_segmentation(self.source, self.root / 'masks', attempt)
        # 本检查仅使用已写出的格式样本核验文件提交顺序。
        record = {'mask_root': str(attempt), 'artifacts': {str(Path(task) / f'{name}.nii.gz'): sha256(self.root / 'masks' / task / f'{name}.nii.gz') for _, name, _, task in LABELS}}
        for _, name, _, task in LABELS:
            folder = attempt / task
            folder.mkdir(exist_ok=True)
            (folder / f'{name}.nii.gz').write_bytes((self.root / 'masks' / task / f'{name}.nii.gz').read_bytes())
        _, name, _, task = LABELS[0]
        (attempt / task / f'{name}.nii.gz').write_bytes(b'changed-format-test-file')
        result = {'output_format': 'slicer-seg-v1', 'qc': report, 'artifacts': {str(path): sha256(path)}}
        with self.assertRaisesRegex(ValueError, '已经改变'):
            commit_editable(attempt, result, record)
        pointer = json.loads((attempt.parent / 'latest-result.json').read_text(encoding='utf-8'))
        self.assertEqual(pointer['directory'], str(attempt.resolve()))
        self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
