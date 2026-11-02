import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class AnalysisTests(unittest.TestCase):
    def test_metadata_dates_precision_and_controls_are_preserved_or_flagged(self):
        from dataset1282_analysis.metadata import FIELDS,read_metadata
        import openpyxl
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'metadata.xlsx';w=openpyxl.Workbook();s=w.active;s.title='dicom2nii_info';s.append(FIELDS)
            r=dict(idx='a',PatientID='1234567890123456',PatientName=r'name\u001B',PatientBirthDate='20210101',AcquisitionDate='20200101')
            s.append([r.get(k) for k in FIELDS]);r.update(idx='b',PatientID=123,PatientBirthDate='not-date')
            s.append([r.get(k) for k in FIELDS]);w.save(path)
            rows=read_metadata(path)
            self.assertEqual(rows[0]['PatientID'],'1234567890123456')
            self.assertEqual(rows[0]['PatientName'],r'name\u001B')
            self.assertIsNone(rows[0]['age_at_acquisition_years'])
            self.assertIn('negative_acquisition_age',rows[0]['_issues'])
            self.assertIn('PatientID_was_numeric_or_nontext',rows[1]['_issues'])
            self.assertIn('PatientBirthDate_invalid',rows[1]['_issues'])

    def test_strict_link_preserves_unmatched_and_duplicate_ids(self):
        self.assertTrue((ROOT / 'src/dataset1282_analysis/metadata.py').exists(), 'metadata linker missing')
        from dataset1282_analysis.metadata import link_metadata, summarize_metadata
        from dataset1282_analysis.contracts import CaseInput
        cases = [CaseInput('A','key1','native','a','b',{}), CaseInput('B','key2','native','c','d',{})]
        rows = [{'idx':'key1','PatientID':'1234567890123456'}, {'idx':'key3','PatientID':'1234567890123456'}]
        linked = link_metadata(cases, rows)
        self.assertEqual(linked['case_rows'][0]['status'], 'matched')
        self.assertEqual(linked['case_rows'][1]['status'], 'unmatched')
        self.assertEqual(linked['unmatched_excel_indices'], [1])
        self.assertEqual(summarize_metadata(rows)['distinct_patient_ids'], 1)
        self.assertEqual(summarize_metadata(rows)['record_count'], 2)
        ambiguous = link_metadata(cases, rows + [dict(rows[0])])
        self.assertEqual(ambiguous['case_rows'][0]['status'], 'ambiguous')

    def test_manifest_uses_canonical_path_and_rejects_traversal(self):
        self.assertTrue((ROOT / 'src/dataset1282_analysis/inputs.py').exists(), 'manifest reader missing')
        from dataset1282_analysis.inputs import load_cases
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [{'case_id':'A', 'source_key':'source', 'collection':'native', 'outputs':[]}]
            (root/'manifest.json').write_text(json.dumps(rows))
            case = load_cases({'data_root':d})[0]
            self.assertEqual(Path(case.image), root/'native/images/A_0000.nii.gz')
            rows[0]['case_id'] = '../escape'
            (root/'manifest.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError): load_cases({'data_root':d})

    def test_streamed_voxels_scaling_nonfinite_hash_and_chunk_invariance(self):
        self.assertTrue((ROOT / 'src/dataset1282_analysis/voxels.py').exists(), 'voxel analysis missing')
        import numpy as np
        import nibabel as nib
        from dataset1282_analysis.voxels import analyze_image
        from dataset1282_analysis.contracts import sha256_file
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'image.nii.gz'
            x = np.arange(120, dtype=np.float64).reshape(4,5,6)
            x[0,0,0] = np.nan; x[1,0,0] = np.inf
            image = nib.Nifti1Image(x, np.eye(4)); image.header.set_slope_inter(2, 5); nib.save(image,path)
            a = analyze_image(path, {'chunk_slices':1,'percentile_samples':100})
            b = analyze_image(path, {'chunk_slices':4,'percentile_samples':100})
            expected = x[np.isfinite(x)] * 2 + 5
            self.assertEqual(a['nonfinite_count'],2)
            self.assertEqual(a['voxel_count'],120)
            self.assertAlmostEqual(a['mean'],expected.mean(),places=10)
            self.assertAlmostEqual(a['std'],expected.std(),places=10)
            self.assertAlmostEqual(a['mean'],b['mean'],places=10)
            self.assertEqual(a['percentiles'],b['percentiles'])
            self.assertEqual(a['sha256'],sha256_file(path))
            self.assertEqual(len(a['slices']),6)
            payload = path.read_bytes(); path.write_bytes(payload[:-20])
            with self.assertRaises((EOFError,OSError,ValueError)): analyze_image(path, {'chunk_slices':2,'percentile_samples':100})

    def test_geometry_separates_array_match_and_physical_uncertainty(self):
        self.assertTrue((ROOT / 'src/dataset1282_analysis/geometry.py').exists(), 'geometry analysis missing')
        import numpy as np
        import nibabel as nib
        from dataset1282_analysis.geometry import inspect_geometry
        with tempfile.TemporaryDirectory() as d:
            a,b = Path(d)/'a.nii.gz',Path(d)/'b.nii.gz'
            affine=np.eye(4); shifted=affine.copy(); shifted[:3,3]=1
            nib.save(nib.Nifti1Image(np.zeros((4,5,6),dtype=np.int8),affine),a)
            nib.save(nib.Nifti1Image(np.zeros((4,5,6),dtype=np.int8),shifted),b)
            result=inspect_geometry(a,b)
            self.assertEqual(result['array_geometry_status'],'match')
            self.assertEqual(result['physical_geometry_status'],'unverified')
            self.assertEqual(result['origin_delta_header_units'],[1.,1.,1.])
            self.assertFalse(result['physical_volume_supported'])

    def test_labels_preserve_invalid_values_and_connected_components(self):
        self.assertTrue((ROOT / 'src/dataset1282_analysis/labels.py').exists(), 'label analysis missing')
        import numpy as np
        import nibabel as nib
        from dataset1282_analysis.labels import analyze_label
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'label.nii.gz'
            x=np.zeros((8,8,8),dtype=np.float64); x[0,0,0]=1; x[4:6,4:6,4:6]=1; x[2,2,2]=2.5; x[7,7,7]=np.nan
            nib.save(nib.Nifti1Image(x,np.eye(4)),p)
            result=analyze_label(p,{'connectivity':1})
            self.assertEqual(result['noninteger_count'],1)
            self.assertEqual(result['nonfinite_count'],1)
            one=next(c for c in result['classes'] if c['value']==1)
            self.assertEqual(one['voxels'],9)
            self.assertEqual(one['components'],2)
            self.assertEqual(one['largest_component_voxels'],8)
            self.assertTrue(one['touches_boundary'])
            self.assertTrue(any(c['value']==2.5 for c in result['classes']))
            self.assertNotIn('volume_ml',one)


if __name__ == '__main__': unittest.main()
