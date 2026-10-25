"""Exercise orchestration with inert compute calls, not real GPU/CT workloads."""
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_worker as worker
from server_data import read_json,write_json,digest


class WorkerTests(unittest.TestCase):
    def fixture(self,root):
        job=root/'job'; code=job/'code';code.mkdir(parents=True)
        plan=dict(name='B1 task',directory=str(job),runtime=str(root/'runtime'),workspace=str(job/'workspace'),code=str(code),
                  data=str(root/'raw'),tasks=['prepare','train','predict'],methods=['B1'],folds=['ct_holdG'],
                  gpus=['GPU-0','GPU-1'],gpu_count=1,gpu_memory=5,source=None)
        write_json(job/'plan.json',plan)
        write_json(job/'state.json',dict(status='queued',plan_sha256=digest(job/'plan.json'),steps={t:{'status':'pending'} for t in plan['tasks']}))
        return job,plan

    def test_order_selection_and_complete_state(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            job,plan=self.fixture(Path(temp));calls=[]
            stack.enter_context(patch.dict(os.environ,{},clear=True))
            stack.enter_context(patch('zmic44_setup.environment',return_value={'WHOLE_HEART_ENV_PREFIX':str(Path(temp)/'env')}))
            stack.enter_context(patch('zmic44_setup.ensure_conda',return_value=Path(temp)/'conda'))
            stack.enter_context(patch('server_data.bind_workspace'))
            stack.enter_context(patch.object(worker,'record_environment'))
            stack.enter_context(patch('task_inputs.receipt'))
            stack.enter_context(patch.object(worker,'run_child',side_effect=lambda command,*a,**k:calls.append(command)))
            gpu=stack.enter_context(patch.object(worker,'wait_gpu'))
            worker.execute(job,9)
            modes=[c[c.index('--mode')+1] for c in calls if '--mode' in c]
            self.assertEqual(modes,['prepare','train','evaluate'])
            for command in calls:
                if '--models' in command:self.assertEqual(command[command.index('--models')+1],'B1')
            self.assertEqual(read_json(job/'state.json')['status'],'completed')
            gpu.assert_not_called() # train/predict use their own bounded queues, no early GPU gate

    def test_compute_failure_keeps_completed_prepare_and_stops_prediction(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            job,plan=self.fixture(Path(temp));calls=[]
            def fake(command,*args,**kwargs):
                calls.append(command)
                if '--mode' in command and command[command.index('--mode')+1]=='train':raise subprocess.CalledProcessError(2,command)
            stack.enter_context(patch.dict(os.environ,{},clear=True))
            stack.enter_context(patch('zmic44_setup.environment',return_value={'WHOLE_HEART_ENV_PREFIX':str(Path(temp)/'env')}))
            stack.enter_context(patch('zmic44_setup.ensure_conda',return_value=Path(temp)/'conda'))
            stack.enter_context(patch('server_data.bind_workspace'))
            stack.enter_context(patch.object(worker,'record_environment'))
            stack.enter_context(patch('task_inputs.receipt'))
            stack.enter_context(patch.object(worker,'run_child',side_effect=fake))
            with self.assertRaises(subprocess.CalledProcessError):worker.execute(job,9)
            state=read_json(job/'state.json')
            self.assertEqual(state['steps']['prepare']['status'],'completed')
            self.assertEqual(state['steps']['train']['status'],'failed')
            self.assertEqual(state['steps']['predict']['status'],'pending')
            self.assertFalse(any('evaluate' in c for c in calls))


if __name__=='__main__':unittest.main()
