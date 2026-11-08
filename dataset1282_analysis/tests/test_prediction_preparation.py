from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from openpyxl import load_workbook

from dataset1282_analysis.contracts import atomic_json, digest, read_json, sha256_file
from dataset1282_analysis.inputs import load_cases
from dataset1282_analysis.metadata import FIELDS, read_metadata
from dataset1282_analysis.preparation import allocate
from dataset1282_analysis.prediction.heldout import GROUPS, HeldoutCache, plan_inputs, prepare_case
from dataset1282_analysis.xu_geometry import PreparedCache

ROOT = Path(__file__).resolve().parents[1]


class FrozenPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location('short_fixture',ROOT/'scripts/short_test.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.scratch = tempfile.TemporaryDirectory(prefix='prediction-test-')
        cls.base = Path(cls.scratch.name).resolve()
        cls.addClassCleanup(cls.scratch.cleanup)
        config_path,cls.policy,cohort_path,_ = module.fixture(cls.base)
        cls.config = read_json(config_path)
        book = load_workbook(cls.config['metadata_excel'])
        sheet = book.active
        for row,device in ((16,'iCT 256'),(17,'Brilliance iCT')):
            sheet.cell(row,FIELDS.index('ManufacturerModelName')+1,device)
        book.save(cls.config['metadata_excel'])
        book.close()
        cohort = allocate(load_cases(cls.config),read_metadata(cls.config['metadata_excel']))
        cohort['inputs'] = dict(manifest=sha256_file(Path(cls.config['data_root'])/'manifest.json'),
                                excel=sha256_file(cls.config['metadata_excel']))
        atomic_json(cohort_path,cohort)
        selected = [next(n for n,r in cohort['cases'].items() if r['role']==role and r['device']==device)
                    for role,device in (('train','LightSpeed VCT'),('train','SOMATOM Definition Flash'),
                                        ('development','LightSpeed VCT'))]
        command = [sys.executable,'-B',str(ROOT/'run.py'),'--config',str(config_path),'prepare',
                   '--workflow',str(cls.policy),'--frozen-cohort',str(cohort_path),
                   '--research-python',sys.executable,'--run','test','--workers','1']
        for name in selected:
            command += ['--case',name]
        with (cls.base/'prepare.log').open('w',encoding='utf-8') as stream:
            result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                    env=dict(os.environ),timeout=240)
        if result.returncode:
            raise RuntimeError((cls.base/'prepare.log').read_text(encoding='utf-8')[-6000:])
        cls.training = cls.base/'work/prepared/test/hu-marginal'
        cls.cache = PreparedCache(cls.training,allow_technical_subset=True)

    def test_frozen_groups_ct_only_normalization_and_resume(self):
        rows = plan_inputs(self.cache,self.config,self.policy,GROUPS)
        self.assertEqual(set(r['role'] for r in rows.values()),set(GROUPS))
        out = self.base/'prediction'
        plans = self.training/'preprocessed/Dataset282_CHDSource'
        inputs = [self.training/'fit.json',plans/'nnUNetPlans.json',plans/'dataset.json']
        before = {str(p):sha256_file(p) for p in inputs}
        job = dict(prepared=str(self.training),cases=rows,plans=sha256_file(plans/'nnUNetPlans.json'),
                   dataset=sha256_file(plans/'dataset.json'))
        atomic_json(out/'run.json',job)
        cache = HeldoutCache(out)
        for name,row in rows.items():
            # 删除合成标签，确认准备与预测供数阶段没有打开标签。
            Path(row['label']).unlink()
            first = prepare_case(out,name)
            self.assertEqual(first,prepare_case(out,name))
            image = cache.read_image(name)
            self.assertTrue(np.isfinite(image).all())
            self.assertEqual(image.dtype,np.float32)
            geo = cache.geometry(name)
            self.assertEqual(geo.native_shape_xyz,tuple(row['shape']))
            self.assertEqual(geo.spacing_xyz,tuple(row['spacing']))
            source = self.cache.case_ids('train')[0]
            # 合成 CT 的像素完全相同，冻结归一化后也应相同。
            np.testing.assert_allclose(image,self.cache.read_image(source),atol=1e-6,rtol=0)
        self.assertEqual(before,{str(p):sha256_file(p) for p in inputs})
        name = next(iter(rows))
        marker = out/'prepared'/name/'geometry.json'
        geometry = read_json(marker)
        geometry['spacing_xyz'][0] *= 2
        atomic_json(marker,geometry)
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            cache.read_image(name)

    def test_changed_cohort_and_raw_image_rejected(self):
        with self.assertRaisesRegex(ValueError,'allocation changed'):
            policy = read_json(self.policy)
            name = next(iter(self.cache._records))
            policy['excluded_cases'] = [dict(case_id=name,reason='synthetic exclusion',
                                            source_sha256=self.cache._records[name]['sha256'])]
            changed = self.base/'changed-policy.json'
            atomic_json(changed,policy)
            plan_inputs(self.cache,self.config,changed,GROUPS)
        rows = plan_inputs(self.cache,self.config,self.policy,GROUPS)
        name = next(iter(rows))
        rows[name]['sha256']['image'] = '0'*64
        plans = self.training/'preprocessed/Dataset282_CHDSource'
        out = self.base/'changed-image'
        atomic_json(out/'run.json',dict(prepared=str(self.training),cases=rows,
                    plans=sha256_file(plans/'nnUNetPlans.json'),dataset=sha256_file(plans/'dataset.json')))
        with self.assertRaisesRegex(ValueError,'CT changed'):
            prepare_case(out,name)


if __name__ == '__main__':
    unittest.main()
