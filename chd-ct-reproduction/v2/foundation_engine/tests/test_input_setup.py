"""Synthetic source discovery, immutable default input and weight acquisition."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import imagechd_input_setup as setup
import stunet_weights as weights


def cohort(raw, count=9):
    import nibabel as nib
    import numpy as np
    raw.mkdir()
    rows = []
    for i in range(count):
        cid = 'ct_' + str(1000 + i)
        affine = np.eye(4)
        label = np.tile(np.arange(8, dtype=np.uint8), (8, 8, 1))
        image = (label.astype(np.int16) * 20 + i)
        if i == count - 1:
            label = label.copy(); label[label == 1] = 0
        for name, data in [('image', image), ('label', label)]:
            nib.save(nib.Nifti1Image(data, affine), raw / (cid + '_' + name + '.nii.gz'))
        values = {'A':'idx','B':str(i),'I':'PixelSpacing','J':'.8','K':'.9','L':'calculate_z_thick','M':'1.2'}
        cells = ''.join(f'<c r="{col}{i+1}" t="inlineStr"><is><t>{value}</t></is></c>' for col, value in values.items())
        rows.append(f'<row r="{i+1}">{cells}</row>')
    with ZipFile(raw / 'imagechd_dataset_image_info.xlsx', 'w') as archive:
        archive.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet r:id="r1"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels','<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(rows)+'</sheetData></worksheet>')


class InputSetup(unittest.TestCase):
    def test_raw_source_unchanged_and_frozen_descriptor_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp)/'raw', Path(tmp)/'input'
            cohort(raw)
            before = {p.name: setup.file_hash(p) for p in raw.iterdir()}
            result = setup.build_auto(raw, out)
            self.assertEqual(8, sum(c['status']=='eligible' for c in result['cases']))
            self.assertEqual([1], result['cases'][-1]['missing_label_ids'])
            self.assertFalse(result['patient_identity_verified'])
            target = out/'imagechd_input.json'; stamp=target.stat().st_mtime_ns
            self.assertEqual(result, setup.build_auto(raw, out))
            self.assertEqual(stamp, target.stat().st_mtime_ns)
            self.assertEqual(before, {p.name:setup.file_hash(p) for p in raw.iterdir()})
            # A changed source must never rewrite an existing frozen descriptor.
            existing=target.read_bytes()
            (raw/'ct_1000_image.nii.gz').write_bytes((raw/'ct_1001_image.nii.gz').read_bytes())
            with self.assertRaisesRegex(ValueError,'重复'):
                setup.build_auto(raw,out)
            self.assertEqual(existing,target.read_bytes())

    def test_matches_original_six_path_folds(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from chd_ct_v2.imagechd.folds import make_plan
        ids=['ct_'+str(1000+i) for i in range(97)]
        plan=make_plan(dict(cases=[dict(case_id=x,patient_id=x,split='train') for x in ids]),42)
        roles=setup.frozen_roles(ids)
        for fold in plan['folds']:
            for role, cases in fold['cases'].items():
                self.assertEqual(set(cases),{cid for cid in ids if roles[cid][str(fold['index'])]==role})

    def test_existing_unrelated_output_rejected_before_reading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw,out=Path(tmp)/'raw',Path(tmp)/'out'
            raw.mkdir();out.mkdir();(out/'private.txt').write_text('keep')
            with self.assertRaises(FileExistsError):setup.build_auto(raw,out)
            with self.assertRaises(ValueError):setup.build_auto(raw,raw/'nested')

    def test_failed_publish_leaves_no_partial_target_and_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw,out=Path(tmp)/'raw',Path(tmp)/'input';cohort(raw)
            with patch.object(setup.os,'rename',side_effect=OSError('interrupted publish')):
                with self.assertRaises(OSError):setup.build_auto(raw,out)
            self.assertFalse(out.exists())
            self.assertEqual({'raw'},{p.name for p in Path(tmp).iterdir()})
            self.assertEqual(9,len(setup.build_auto(raw,out)['cases']))


class WeightAcquisition(unittest.TestCase):
    def test_configured_external_weight_path_remains_explicit(self):
        from server_profile import weight_location
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            external=root/'read-only-weights/small.model'
            profile=dict(runtime=str(root/'default'),pretrained_weights=str(external))
            self.assertEqual(str(external),weight_location(profile,root/'custom'))

    def test_verified_reuse_and_bad_existing_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as tmp,patch('urllib.request.build_opener',side_effect=AssertionError('network')):
            target=Path(tmp)/'weight';target.write_bytes(b'valid')
            with patch.object(weights,'WEIGHT_BYTES',5),patch.object(weights,'WEIGHT_SHA256',hashlib.sha256(b'valid').hexdigest()):
                self.assertEqual(target,weights.ensure_weights(target))
                target.write_bytes(b'wrong')
                with self.assertRaises(ValueError):weights.ensure_weights(target)
                self.assertEqual(b'wrong',target.read_bytes())

    def test_direct_then_runtime_proxy_and_no_partial_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'weights'/'small.model'
            env={'HTTP_PROXY':'http://do-not-use','WHOLE_HEART_ENV_PREFIX':'unused'};original=env.copy()
            response=io.BytesIO(b'valid')
            opener=unittest.mock.Mock();opener.open.side_effect=[OSError('direct failed'),response]
            with patch.object(weights,'WEIGHT_BYTES',5),patch.object(weights,'WEIGHT_SHA256',hashlib.sha256(b'valid').hexdigest()),\
                 patch('runtime_setup._install_routes',return_value=iter([('direct',None),('proxy','http://runtime')])),\
                 patch('urllib.request.build_opener',return_value=opener):
                weights.ensure_weights(target,env)
            self.assertEqual(b'valid',target.read_bytes());self.assertEqual(original,env)
            names={p.name for p in target.parent.iterdir()}
            self.assertIn(target.name,names)
            self.assertLessEqual(names,{target.name,target.name+'.lock'})

    def test_failed_or_cancelled_download_never_publishes_final(self):
        for error in [ValueError('corrupt'),KeyboardInterrupt()]:
            with tempfile.TemporaryDirectory() as tmp:
                target=Path(tmp)/'small.model';opener=unittest.mock.Mock()
                opener.open.side_effect=error
                with patch('runtime_setup._install_routes',return_value=iter([('direct',None)])),\
                     patch('urllib.request.build_opener',return_value=opener):
                    with self.assertRaises((RuntimeError,KeyboardInterrupt)):weights.ensure_weights(target,{})
                self.assertEqual([],list(Path(tmp).iterdir()))


if __name__=='__main__':unittest.main()
