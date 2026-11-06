import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from dataset1282_analysis.contracts import atomic_json, sha256_file, CaseInput
from dataset1282_analysis.preprocessing.workflow import shared_geometry, derive_cohort
from dataset1282_analysis.preparation import allocate
from dataset1282_analysis.preprocessing.queue import committed


class WorkflowTests(unittest.TestCase):
    def test_runtime_versions_are_checked_before_work(self):
        import json
        from dataset1282_analysis.preprocessing.runtime import numerical_runtime
        payload={'packages':{'nnunetv2':'2.8.1','numpy':'test'},'official_modules':{'p':'sha'}}
        with patch('dataset1282_analysis.preprocessing.runtime.subprocess.run',
                   return_value=SimpleNamespace(returncode=0,stdout=json.dumps(payload))) as run:
            self.assertEqual(numerical_runtime('research-python'),payload)
            self.assertIn('-I',run.call_args.args[0])
        payload['packages']['nnunetv2']='changed'
        with patch('dataset1282_analysis.preprocessing.runtime.subprocess.run',
                   return_value=SimpleNamespace(returncode=0,stdout=json.dumps(payload))):
            with self.assertRaises(ValueError): numerical_runtime('research-python')

    def test_pressure_stops_owned_job_and_carries_reduced_budget(self):
        from dataset1282_analysis.preprocessing import queue
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic_json(root/'request.json',{'synthetic':'pressure'})
            state={'launches':0,'pressure':False}
            def spawn(*a,**k):
                state['launches']+=1
                number=state['launches']
                return SimpleNamespace(pid=100+number,poll=lambda:None if number==1 else 0)
            def sample():
                available=16*2**30
                if state['launches']==1 and not state['pressure']:
                    available=3*2**30
                    state['pressure']=True
                return dict(cpu_available=8,cpu_ids=list(range(8)),available_memory_bytes=available)
            policy=dict(cpu_fraction=.8,ram_fraction=.8,reserve_gib=4,max_workers=2,read_concurrency=2,resource_wait_seconds=2)
            job=dict(id='test',kind='stage',memory=2**20)
            with patch.object(queue,'snapshot',side_effect=sample), \
                 patch.object(queue.subprocess,'Popen',side_effect=spawn), \
                 patch.object(queue.psutil,'Process',return_value=SimpleNamespace(create_time=lambda:1)), \
                 patch.object(queue,'sample_owned'), patch.object(queue,'register_workers'), \
                 patch.object(queue,'stop_owned') as stop, \
                 patch.object(queue,'committed',side_effect=lambda *a:state['launches']>=2), \
                 patch.object(queue.time,'sleep'):
                result=queue.run_jobs(root,[job],'python',policy,root,{})
            self.assertEqual(stop.call_count,1)
            self.assertEqual(result['memory_recoveries'],1)
            self.assertEqual(policy['_effective_workers'],1)
            self.assertEqual(policy['_memory_factor'],.75)
            self.assertEqual(state['launches'],2)

    def cohort_fixture(self):
        rows=[dict(idx=f'case{i}',PatientID=str(i//2),ManufacturerModelName='LightSpeed VCT',
                   PatientBirthDate='2020-01-01',AcquisitionDate=f'2021-01-{i+1:02}',_excel_row=i+2)
              for i in range(8)]
        rows.append(dict(idx='excel_alias',PatientID='12345',ManufacturerModelName='iCT 256',
                         PatientBirthDate='2020-01-01',AcquisitionDate='2021-01-01',_excel_row=10))
        cases=[CaseInput(f'case{i}',f'case{i}','native','image','label',dict(image='i',label='l')) for i in range(8)]
        cases.append(CaseInput('alias','ct_12345_norm','review_required','image','label',dict(image='i',label='l')))
        identities=dict(manifest='m',excel='e')
        frozen=allocate(cases,rows)
        frozen['inputs']=identities
        policy=dict(patient_aliases=[dict(case_id='alias',source_key='ct_12345_norm',excel_idx='excel_alias',
                    patient_id='12345',excel_row=10,workbook_sha256='e',source_sha256=dict(image='i',label='l'))],
                    excluded_cases=[dict(case_id='case0',reason='test conflict',source_sha256=dict(image='i',label='l'))])
        return cases,rows,frozen,policy,identities

    def test_alias_exclusion_index_and_frozen_roles(self):
        args=self.cohort_fixture()
        derived,links=derive_cohort(*args)
        self.assertEqual(derived['cases']['alias']['role'],'target_primary')
        self.assertEqual(links['alias'],8)
        self.assertFalse(derived['cases']['case0']['included'])
        self.assertTrue(derived['cases']['case1']['index_scan'])
        self.assertTrue(derived['cases']['case2']['index_scan'])
        self.assertFalse(derived['cases']['case3']['index_scan'])
        for i in range(8):
            self.assertEqual(derived['cases'][f'case{i}']['role'],args[2]['cases'][f'case{i}']['role'])

    def test_stale_policy_and_cohort_are_rejected(self):
        args=self.cohort_fixture()
        args[3]['patient_aliases'][0]['patient_id']='wrong'
        with self.assertRaises(ValueError): derive_cohort(*args)
        args=self.cohort_fixture()
        args[2]['cases']['case2']['role']='source_test'
        with self.assertRaises(ValueError): derive_cohort(*args)

    def test_placeholder_geometry_without_voxel_translation(self):
        image=SimpleNamespace(ndim=3,shape=(20,30,40),affine=np.eye(4))
        label=SimpleNamespace(shape=image.shape,affine=np.eye(4))
        label.affine[:3,3]=1
        np.testing.assert_equal(shared_geometry(image,label,[.3,.4,.6]),np.diag([.3,.4,.6,1]))
        label.affine[0,3]=2
        with self.assertRaises(ValueError): shared_geometry(image,label,[.3,.4,.6])

    def test_commit_integrity_includes_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            artifact=root/'case_seg.b2nd'
            artifact.write_bytes(b'expected')
            job={'id':'process-case','kind':'process'}
            atomic_json(root/'commits/process-case.json',dict(contract='x',job=job,
                        artifacts={'case_seg.b2nd':sha256_file(artifact)}))
            self.assertTrue(committed(root,job,'x'))
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError): committed(root,job,'x')


if __name__=='__main__': unittest.main()
