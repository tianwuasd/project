from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class RuntimeTests(unittest.TestCase):
    def test_private_credit_rejects_child_pid_reuse_after_memory_sample(self):
        from dataset1282_analysis.engine.runner import sample_owned
        root=SimpleNamespace(pid=10,create_time=lambda:1,children=lambda recursive:[child],
                             memory_info=lambda:SimpleNamespace(rss=5),memory_full_info=lambda:SimpleNamespace(uss=5))
        child=SimpleNamespace(pid=11,create_time=lambda:2,memory_info=lambda:SimpleNamespace(rss=700),memory_full_info=lambda:SimpleNamespace(uss=600))
        reused=SimpleNamespace(create_time=lambda:3)
        def process(pid): return root if pid==10 else reused
        item={'process':SimpleNamespace(pid=10),'born':1,'peak_rss':0}
        with patch('dataset1282_analysis.engine.runner.psutil.Process',side_effect=process):sample_owned(item)
        self.assertEqual(item['private_resident'],5)
        self.assertNotIn(11,item.get('owned',{}))

    def test_interrupted_official_run_resumes_complete_cases_only(self):
        from dataset1282_analysis.smoke import synthetic_fixture
        from dataset1282_analysis.config import resolve_config
        from dataset1282_analysis.engine import runner
        from dataset1282_analysis.contracts import read_json
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data,excel=synthetic_fixture(base/'input')
            config=resolve_config({'data_root':str(data),'metadata_excel':str(excel),'work_root':str(base/'work'),'max_workers':1},None,ROOT)
            event=runner.event
            def stop_after_first(directory,name,**fields):
                event(directory,name,**fields)
                if name=='case_done':raise KeyboardInterrupt('synthetic controlled interruption')
            with patch.object(runner,'event',side_effect=stop_after_first):
                with self.assertRaises(KeyboardInterrupt):runner.run_analysis(config,ROOT,base/'home',run_id='interrupted')
            resumed=runner.run_analysis(config,ROOT,base/'home',run_id='interrupted',resume=True)
            continuous=runner.run_analysis(config,ROOT,base/'home',run_id='continuous')
            self.assertEqual(resumed['reused'],1)
            self.assertEqual((resumed['complete'],resumed['failed']),(continuous['complete'],continuous['failed']))
            for name in ('SYN_000','SYN_001'):
                a=read_json(base/f'work/runs/interrupted/cases/{name}.json')
                b=read_json(base/f'work/runs/continuous/cases/{name}.json')
                self.assertEqual(a['image'],b['image']);self.assertEqual(a['label'],b['label'])

    def test_worker_peak_memory_is_reported_from_actual_worker(self):
        from dataset1282_analysis.smoke import synthetic_fixture
        from dataset1282_analysis.engine.worker import analyze_case
        from dataset1282_analysis.config import resolve_config
        from dataset1282_analysis.inputs import load_cases
        with tempfile.TemporaryDirectory() as d:
            data,excel=synthetic_fixture(Path(d)/'inputs')
            config=resolve_config({'data_root':str(data),'metadata_excel':str(excel),'work_root':str(Path(d)/'work')},None,ROOT)
            result=analyze_case({'case':load_cases(config)[0].to_dict(),'config':config,'contract':'test','stage':'full','metadata_link_status':'matched'})
            self.assertGreater(result.get('peak_process_rss_bytes',0),10*1024**2)

    def test_synthetic_official_workflow_and_resume(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/engine/runner.py').exists(),'official runner missing')
        from dataset1282_analysis.smoke import synthetic_fixture
        from dataset1282_analysis.config import resolve_config
        from dataset1282_analysis.engine.runner import run_analysis
        from dataset1282_analysis.contracts import read_json
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); data,excel=synthetic_fixture(base/'inputs')
            config=resolve_config({'data_root':str(data),'metadata_excel':str(excel),'work_root':str(base/'work')},None,ROOT)
            result=run_analysis(config,ROOT,base/'home',run_id='test',case_ids=None)
            self.assertEqual(result['total'],3)
            self.assertEqual(result['complete'],2)
            self.assertEqual(result['failed'],1)
            second=run_analysis(config,ROOT,base/'home',run_id='test',resume=True)
            self.assertEqual(second['reused'],2)
            self.assertEqual(second['failed'],1)
            status=read_json(base/'work/runs/test/status.json')
            self.assertEqual(status['state'],'completed_with_failures')

    def test_admission_does_not_double_charge_resident_bytes(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/resources.py').exists(),'resource admission missing')
        from dataset1282_analysis.resources import admit_case
        stage={'cpu_slots':4,'memory_bytes':1000,'workers':4}
        fresh={'cpu_slots':2,'memory_bytes':550}
        active=[{'reserved':600,'private_resident':450,'threads':1}]
        self.assertTrue(admit_case(stage,fresh,active,400)['allowed'])
        self.assertFalse(admit_case(stage,fresh,active,401)['allowed'])
        self.assertFalse(admit_case(stage,{'cpu_slots':2,'memory_bytes':499},active,400)['allowed'])
        self.assertFalse(admit_case(stage,{'cpu_slots':2,'memory_bytes':None},active,1)['allowed'])
        self.assertFalse(admit_case(stage,{'cpu_slots':0,'memory_bytes':9999},[],1)['allowed'])

    def test_resume_verifies_content_and_complete_marker(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/engine/recovery.py').exists(),'recovery implementation missing')
        from dataset1282_analysis.engine.recovery import reusable_result, commit_result
        from dataset1282_analysis.contracts import sha256_file
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); a=root/'a'; b=root/'b'; a.write_bytes(b'abc');b.write_bytes(b'xyz')
            result={'status':'complete','contract':'v1','input_paths':{'image':str(a),'label':str(b)},
                    'source_sha256':{'image':sha256_file(a),'label':sha256_file(b)}}
            p=root/'case.json'
            commit_result(p,result)
            self.assertTrue(reusable_result(p,'v1'))
            self.assertFalse(reusable_result(p,'v2'))
            a.write_bytes(b'xyz')
            self.assertFalse(reusable_result(p,'v1'))

    def test_complete_result_cannot_be_forged_by_filename(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/engine/recovery.py').exists(),'recovery implementation missing')
        from dataset1282_analysis.engine.recovery import reusable_result
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'case.json';p.write_text('{"status":"complete","contract":"v1"}')
            self.assertFalse(reusable_result(p,'v1'))


if __name__=='__main__': unittest.main()
