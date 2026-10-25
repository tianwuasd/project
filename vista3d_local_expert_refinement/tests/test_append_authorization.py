from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_runtime as jobs
from server_data import write_json,digest

class AppendTests(unittest.TestCase):
    def test_cpu_only_job_cannot_append_gpu_work_without_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'job';code=job/'code';code.mkdir(parents=True)
            plan=dict(name='cpu',directory=str(job),runtime=str(root/'runtime'),workspace=str(job/'workspace'),code=str(code),
                      tasks=['prepare'],methods=['B0'],folds=['ct_holdG'],gpus=[],gpu_count=0)
            write_json(job/'plan.json',plan);write_json(job/'code_manifest.json',{})
            write_json(job/'state.json',dict(status='completed',plan_sha256=digest(job/'plan.json'),steps={'prepare':{'status':'completed'}}))
            before=(job/'plan.json').read_bytes()
            with patch.object(jobs,'launch') as launch:
                with self.assertRaises(ValueError):jobs.append_tasks(job,['train'])
                launch.assert_not_called()
            self.assertEqual(before,(job/'plan.json').read_bytes())

if __name__=='__main__':unittest.main()
