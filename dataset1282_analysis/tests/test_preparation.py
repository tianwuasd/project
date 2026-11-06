import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dataset1282_analysis.contracts import CaseInput, digest, sha256_file
from dataset1282_analysis.metadata import FIELDS
from dataset1282_analysis.preparation import SOURCES, allocate, apply_excel_geometry, select_cases, select_excel_pilot


def row(idx, patient, device):
    return dict(idx=idx, PatientID=patient, ManufacturerModelName=device,
                AcquisitionDate='2020-06-01', PatientBirthDate='2020-01-01')


class CohortTests(unittest.TestCase):
    def test_excel_pilot_only_opens_source_train_and_reserves_independent_patients(self):
        import nibabel as nib
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            file = str(Path(folder) / 'fixture.nii.gz')
            nib.save(nib.Nifti1Image(np.zeros((8, 9, 7), dtype='uint8'), np.eye(4)), file)
            cases, rows, records = [], [], {}
            for n in range(11):
                name = str(n)
                role = 'train' if n < 9 else 'source_test' if n == 9 else 'target_primary'
                path = file if role == 'train' else str(Path(folder) / 'must_not_open')
                cases.append(CaseInput(name, name, 'native', path, path, {'image':'a', 'label':'b'}))
                rows.append(dict(idx=name, Cols=8, Rows=9, valid_z=6 if n == 8 else 7,
                                 PixelSpacingX=2.5 if n == 6 else .25, PixelSpacingY=.5,
                                 calc_zSpacing=.625, _excel_row=n+2))
                records[name] = dict(role=role, patient=str(max(n-1, 0)), device=SOURCES[int(n >= 4)])
            selected, screen = select_excel_pilot(cases, {'cases':records}, rows, 1200 * 2**20 + 40*504)
            self.assertEqual((screen['checked'], screen['eligible']), (9,8))
            self.assertEqual(sorted(v['role'] for v in selected.values()), ['pilot_probe']*2 + ['train']*4)
            self.assertEqual(len({records[n]['patient'] for n in selected}), 6)
            self.assertEqual(screen['cases']['6']['capacity'], 'exceeds_local_reservation')
            self.assertNotIn('6', selected)
            self.assertEqual(selected['7']['role'], 'pilot_probe')
            self.assertNotIn('9', screen['cases'])
            self.assertNotIn('10', screen['cases'])

    def test_excel_spacing_rejects_missing_increment_and_slice_count_mismatch(self):
        import nibabel as nib
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            file = str(Path(folder) / 'A.nii.gz')
            nib.save(nib.Nifti1Image(np.zeros((5, 4, 7), dtype='uint8'), np.eye(4)), file)
            cases = [CaseInput('A', 'key', 'native', file, '', {})]
            selected = {'A': {'image': file, 'review': {'spacing_source': 'excel', 'excel_grid_verified': True,
                         'directions_ras': np.eye(3).tolist(), 'origin_ras_mm': [0, 0, 0]}}}
            row = dict(idx='key', PixelSpacingX=.4, PixelSpacingY=.6, calc_zSpacing=None,
                       SliceThickness=1.25, Cols=5, Rows=4, valid_z=7, _excel_row=2)
            with self.assertRaisesRegex(ValueError, 'slice increment'):
                apply_excel_geometry(selected, cases, [row])
            row.update(calc_zSpacing=.625, valid_z=6)
            with self.assertRaisesRegex(ValueError, 'dimensions differ'):
                apply_excel_geometry(selected, cases, [row])

    def test_patient_embargoes_and_order_independent_allocation(self):
        rows = [row(str(i), str(i), SOURCES[0]) for i in range(30)]
        rows += [row('duplicate', '0', SOURCES[0]), row('target', '1', 'iCT 256'),
                 row('missingfile', '2', 'Brilliance iCT'), row('ultra', '3', 'LightSpeed Ultra')]
        cases = [CaseInput(r['idx'], r['idx'], 'native', '', '', {}) for r in rows if r['idx'] != 'missingfile']
        a = allocate(cases, rows)
        self.assertEqual(a, allocate(list(reversed(cases)), list(reversed(rows))))
        c = a['cases']
        self.assertEqual(c['0']['role'], c['duplicate']['role'])
        self.assertEqual(c['target']['role'], 'target_primary')
        for key in ['1', '2']:
            self.assertEqual(c[key]['role'], 'target_patient_other_scan')
        self.assertEqual(c['3']['role'], 'other_device')
        for key in ['target', '1', '2', 'ultra']:
            with self.assertRaisesRegex(ValueError, 'ineligible'):
                select_cases(cases, a, {'cohort_digest': digest(a), 'cases': {key: {}}})

    def test_review_is_bound_to_cohort_and_originals(self):
        cases = [CaseInput('A', 'A', 'review_required', '', '', {'image': 'a', 'label': 'b'})]
        a = {'cases': {'A': {'role': 'train'}}}
        good = dict(evidence='fixture', index_alignment_verified=True, patient_group_verified=True,
                    preoperative_verified=True, full_supervision_verified=True)
        with self.assertRaisesRegex(ValueError, 'frozen cohort'):
            select_cases(cases, a, {'cohort_digest': 'bad', 'cases': {}})
        with self.assertRaisesRegex(ValueError, 'bound'):
            select_cases(cases, a, {'cohort_digest': digest(a), 'cases': {'A': good}})


@unittest.skipUnless(os.environ.get('DATASET1282_NNUNET_PYTHON'), 'set research interpreter for numerical integration')
class PhysicalPipelineTests(unittest.TestCase):
    def test_only_training_affects_spacing_and_validation_returns_to_original_grid(self):
        import nibabel as nib
        import numpy as np
        import openpyxl
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / 'source'
            for sub in ('images', 'labels'):
                (data / 'native' / sub).mkdir(parents=True)
            excel = base / 'metadata.xlsx'
            wb = openpyxl.Workbook()
            sheet = wb.active
            sheet.title = 'dicom2nii_info'
            sheet.append(FIELDS)
            manifest = []
            for i in range(20):
                name = f'C{i:02d}'
                record = row(name, 'P' + name, SOURCES[0])
                record.update(PixelSpacingX=.75, PixelSpacingY=1., calc_zSpacing=2., SliceThickness=3.,
                              Cols=52, Rows=48, valid_z=40)
                sheet.append([record.get(f) for f in FIELDS])
                manifest.append(dict(case_id=name, source_key=name, collection='native', outputs=[]))
            wb.save(excel)
            # Oblique-axis assignment with a flip and nonzero origin.
            affine = np.array([[0,-1,0,32],[.75,0,0,1],[0,0,2,5],[0,0,0,1.]])
            shape = (52, 48, 40)
            xyz = np.indices(shape)
            mask = ((xyz[0]-22)**2 + (xyz[1]-21)**2 + (xyz[2]-19)**2) < 15**2
            image = np.where(mask, 20 + xyz[0] + xyz[2], 0).astype(np.int16)
            seg = np.where(mask, 1 + xyz[0] % 10, 0).astype(np.uint8)
            for item in manifest:
                name = item['case_id']
                for group, suffix, arr in [('images','_0000',image), ('labels','',seg)]:
                    path = data / 'native' / group / (name + suffix + '.nii.gz')
                    nib.save(nib.Nifti1Image(arr, affine), path)
                    item['outputs'].append(dict(path=str(path), sha256=sha256_file(path)))
            (data / 'manifest.json').write_text(json.dumps(manifest))
            cfg = base / 'config.json'
            cfg.write_text(json.dumps(dict(data_root=str(data), metadata_excel=str(excel),
                                           environment=sys.prefix, work_root=str(base / 'work'))))
            cmd = [sys.executable, '-B', str(ROOT / 'run.py'), '--config', str(cfg),
                   'prepare', '--run', 'example']
            subprocess.run(cmd, check=True, capture_output=True)
            run = base / 'work/prepared/example'
            cohort = json.loads((run / 'cohort.json').read_text(encoding='utf8'))
            items = {}
            for role in ('train', 'development'):
                name = next(k for k, r in cohort['cases'].items() if r['role'] == role)
                pair = next(i['outputs'] for i in manifest if i['case_id'] == name)
                matrix = affine.copy()
                if role == 'development':
                    matrix[:3,:3] *= 1.5
                items[name] = dict(source_sha256=dict(image=pair[0]['sha256'], label=pair[1]['sha256']),
                                   affine_ras_mm=matrix.tolist(), evidence='synthetic coordinate fixture',
                                   preoperative_verified=True, patient_group_verified=True,
                                   index_alignment_verified=True, full_supervision_verified=True)
                if role == 'train':
                    del items[name]['affine_ras_mm']
                    items[name].update(spacing_source='excel', excel_grid_verified=True,
                                       directions_ras=[[0,-1,0],[1,0,0],[0,0,1]], origin_ras_mm=[32,1,5])
            review = base / 'review.json'
            review.write_text(json.dumps(dict(cohort_digest=digest(cohort), cases=items)))
            result = subprocess.run(cmd + ['--review', str(review), '--research-python',
                                    os.environ['DATASET1282_NNUNET_PYTHON']], capture_output=True)
            if result.returncode:
                self.fail((run/'physical/preparation.log').read_text(errors='replace') + repr(result.stderr))
            out = run / 'physical'
            complete = json.loads((out / 'complete.json').read_text(encoding='utf8'))
            self.assertEqual(sorted(complete['spacing_plan_axes']), [.75, 1, 2])
            self.assertEqual(complete['units'], 'mm')
            self.assertEqual(complete['fit_cases'], [next(iter(items))])
            staged = json.loads((out / 'qc' / (complete['fit_cases'][0] + '.json')).read_text())
            self.assertTrue(np.allclose(staged['working_affine'], affine))
            prep = out / 'preprocessed/Dataset282_CHDSource'
            splits = json.loads((prep / 'splits_final.json').read_text())[0]
            self.assertEqual(set(splits['train'] + splits['val']), set(items))
            check = json.loads((out / 'qc' / (splits['val'][0] + '.json')).read_text())
            self.assertEqual(check['roundtrip']['classes']['1']['original_voxels'], int((seg == 1).sum()))
            self.assertTrue(all(0.85 < v['dice'] <= 1 for v in check['roundtrip']['classes'].values()))
