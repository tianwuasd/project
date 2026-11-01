"""Failure injection for real-size gate, resource budget and resumable phase ordering."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from server_data import write_json,read_json,digest
import task_resources as resources
import task_inputs as inputs
import task_runtime as jobs
import preflight_gate as gate
import evaluation_parallel as queue
import task_queries as queries
from evaluation_contract import committed


class InnerContracts(unittest.TestCase):
    def test_resumed_model_cannot_estimate_with_old_epoch_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);out=work/'07_experiments/formal/imagechd_fold0_B0_seed0'
            write_json(out/'run.json',dict(status='running',planned_epochs=250,history=[{'started':'2026-10-25 12:00:00'},{'started':'2026-10-26 00:00:00'}]))
            (out/'training.log').write_text('2026-10-25 12:00:00: Epoch 9\n2026-10-25 12:01:00: Epoch time: 60 s\n')
            text='\n'.join(queries.model_lines(work))
            self.assertNotIn('240.0分钟',text);self.assertIn('无法估计',text)
    def test_effective_cpu_limits_loaded_libraries_and_environment(self):
        torch=Mock();limiter=Mock()
        with patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,{'threadpoolctl':SimpleNamespace(threadpool_limits=limiter)}):
            resources.apply_training_cpu({'threads':2},torch)
            limiter.assert_called_once_with(limits=2);torch.set_num_threads.assert_called_once_with(2)
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'],'2')

    def test_io_pressure_limits_reads_and_pending_models_are_visible(self):
        allocation=resources.allocate(dict(cpu_slots=32,available_memory_bytes=128*1024**3,io_pressure={'avg10':20}),3)
        self.assertEqual(allocation['cpu_tail_workers'],25)
        self.assertEqual(allocation['read_concurrency'],2)
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            write_json(work/'00_admin/model_selection.json',dict(folds=['imagechd_fold0','imagechd_fold1'],methods=['S1']))
            write_json(work/'00_admin/gpu_queue.json',dict(jobs=[dict(fold='imagechd_fold0',method='S1',status='running',gpu_uuid='GPU-actual')]))
            text='\n'.join(queries.model_lines(work))
            self.assertIn('imagechd_fold1/B0',text);self.assertIn('GPU-actual',text);self.assertIn('学习率',text)

    def test_shared_budget_and_binding_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'unbound';env={}
            snapshot=dict(cpu_slots=3,available_memory_bytes=9*1024**3)
            with patch.object(resources,'probe',return_value=snapshot):
                budget=resources.configure(root/'job',dict(workspace=str(work),gpu_count=5),env)
            self.assertFalse(work.exists())
            self.assertEqual(budget['parallel_workers'],1)
            self.assertLessEqual(budget['threads_per_worker']*budget['parallel_workers'],3)
            self.assertEqual(env['WHOLE_HEART_GPU_LIMIT'],'1')

    def test_real_size_gate_reuse_then_recheck_changed_gpu_or_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            write_json(work/'04_data/manifests/splits_v1.json',dict(folds={'imagechd_fold0':{'dataset_id':813}}))
            write_json(work/'04_data/derived/nnUNet_preprocessed/Dataset813_imagechd_fold0/WholeheartBaselinePlans.json',{'batch_size':2})
            for name in ['pip_freeze_training.txt','conda_explicit_training.txt']:
                path=work/'06_configs'/name;path.parent.mkdir(exist_ok=True);path.write_text('frozen')
            paths=[]
            def passed(path):
                paths.append(path);write_json(path,dict(status='passed',checkpoint_weights_equal=True,optimizer_state_restored=True,one_resumed_training_step_passed=True))
            gate.ensure(work,'imagechd_fold0','S1','GPU-0',['GPU-0','24GiB'],passed)
            gate.ensure(work,'imagechd_fold0','S1','GPU-0',['GPU-0','24GiB'],passed)
            self.assertEqual(len(paths),1)
            gate.ensure(work,'imagechd_fold0','S1','GPU-1',['GPU-1','24GiB'],passed)
            self.assertEqual(len(paths),2)
            (work/'06_configs/pip_freeze_training.txt').write_text('changed')
            gate.ensure(work,'imagechd_fold0','S1','GPU-0',['GPU-0','24GiB'],passed)
            self.assertEqual(len(paths),3)
            paths[-1].write_text('{}')
            with self.assertRaises(ValueError):gate.ensure(work,'imagechd_fold0','S1','GPU-0',['GPU-0','24GiB'],passed)

    def test_failed_preflight_cannot_publish_gate(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(gate,'signature',return_value={'test':1}):
            work=Path(tmp)
            with self.assertRaises(ValueError):
                gate.ensure(work,'imagechd_fold0','S1','GPU-0',[],lambda path:write_json(path,{'status':'failed'}))
            self.assertFalse((work/'00_admin/preflight/imagechd_fold0_B0_GPU-0.json').exists())

    def test_bounded_cpu_trials_are_measured_selected_and_frozen_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-0','WHOLE_HEART_CPU_THREADS':'4'},clear=True),patch.object(gate,'signature',return_value={'cpu_threads':'4'}),patch.object(gate.subprocess,'check_output',return_value='GPU-0,Test,600,24000\n'):
            work=Path(tmp);seen=[]
            def run(command,**kwargs):
                threads=int(kwargs['env']['WHOLE_HEART_CPU_THREADS']);seen.append(threads)
                write_json(Path(command[-1]),dict(status='passed',median_step_seconds={1:3.,2:1.,4:2.}[threads],checkpoint_weights_equal=True,optimizer_state_restored=True,one_resumed_training_step_passed=True))
            with patch.object(gate.subprocess,'run',side_effect=run):
                selected=gate.before_training(work,'imagechd_fold0','S1')
            self.assertEqual(seen,[1,2,4]);self.assertEqual(selected['threads'],2)
            # Different workspace simulates a new GPU compatibility check on resume.
            seen.clear();os.environ['WHOLE_HEART_CPU_THREADS']='4'
            with patch.object(gate.subprocess,'run',side_effect=run):
                selected=gate.before_training(work/'resume','imagechd_fold0','S1',selected)
            self.assertEqual(seen,[2]);self.assertEqual(selected['threads'],2)
            os.environ['WHOLE_HEART_CPU_THREADS']='1'
            with self.assertRaisesRegex(ValueError,'低于原训练'):gate.before_training(work,'imagechd_fold0','S1',selected)

    def test_source_frozen_evidence_is_checked_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();state=source/'state.json';state.write_text('original')
            record=dict(directory=str(source),workspace=str(source),files={},evidence={'state.json':digest(state)})
            state.write_text('changed')
            with self.assertRaisesRegex(ValueError,'来源任务证据'):
                inputs._copy_input_locked(dict(source=record,workspace=str(root/'new')))
            self.assertFalse((root/'new').exists())

    def test_simple_resume_checks_only_direct_dependencies_and_reports_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'code';code.mkdir()
            write_json(root/'plan.json',dict(code=str(code)));write_json(root/'code_manifest.json',{})
            write_json(root/'state.json',dict(steps={'prepare':{'status':'completed'},'smoke':{'status':'completed'},'train':{'status':'pending'}}))
            good=root/'cache';good.write_bytes(b'cache')
            stat=good.stat();info=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest(good))
            write_json(root/'receipts/prepare.json',dict(files={str(good):info}))
            write_json(root/'receipts/smoke.json',dict(files={str(root/'missing'):info}))
            inputs.verify_resume(root,'simple')
            self.assertEqual(read_json(root/'resume-check.json')['scope'],['prepare'])
            with self.assertRaises(FileNotFoundError):inputs.verify_resume(root,'full')
            self.assertEqual(read_json(root/'resume-check.json')['status'],'failed')

    def test_recalculated_upstream_cannot_use_old_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            request=dict(name='test',data=str(raw),runtime=str(root/'runtime'),results=str(root/'results'),
                         tasks=['train','summarize'],methods=['S1'],folds=['imagechd_fold0'],input_job='old',gpu_count=1,gpus=['GPU-0'])
            with self.assertRaisesRegex(ValueError,'必须同时选择预测'):jobs.validate_request(request)
            request['tasks']=['prepare','predict']
            with self.assertRaisesRegex(ValueError,'必须同时选择训练'):jobs.validate_request(request)

    def test_late_idle_gpu_joins_pending_cases_within_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            launched=[]
            class Process:
                def __init__(self,*a,**k):
                    self.pid=100+len(launched);self.polls=0;launched.append(k['env']['CUDA_VISIBLE_DEVICES'])
                def poll(self):self.polls+=1;return 0 if self.polls>2 else None
            def idle(*_):return ['GPU-0'] if len(launched)==0 else ['GPU-0','GPU-1']
            records=[dict(fold='imagechd_fold0',method='S1',index=0,case_id=f'Case{i}',status='pending') for i in range(5)]
            with patch.object(queue,'idle_gpus',side_effect=idle),patch.object(queue.subprocess,'Popen',Process),\
                 patch.object(queue,'prediction_budget',return_value=dict(workers=2,threads_per_worker=1,capacity_status='bounded')),\
                 patch.object(queue.time,'sleep'),patch.object(queue.time,'monotonic',side_effect=range(10000)):
                queue.run_phase(records,'sha',dict(max_gpus=2,retry_seconds=1),Path(tmp))
            self.assertIn('GPU-1',launched)

    def test_cpu_tail_resume_does_not_query_or_start_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp)/'04_data/manifests/case_inventory.json',dict(cases=[dict(case_id=c,image='synthetic') for c in ['Case1','Case2']]))
            masks=[];proc=Mock(pid=10);proc.poll.return_value=0
            def start(*a,**k):masks.append(k['env']['CUDA_VISIBLE_DEVICES']);return proc
            model=dict(fold='imagechd_fold0',method='S1',targets=['Case1','Case2'])
            with patch('task_resources.cpu_tail_budget',return_value=dict(workers=2,threads_per_worker=3,cpu_budget=6)) as fresh,patch('evaluation_contract.cached',side_effect=lambda *args:args[-1]=='gpu'),patch.object(queue,'idle_gpus') as idle,patch.object(queue.subprocess,'Popen',side_effect=start),patch.object(queue.time,'sleep'),patch.object(queue.time,'monotonic',side_effect=range(10000)),patch('evaluation_summary.summarize') as summary:
                queue.run_queue([model],'sha',dict(max_gpus=2,retry_seconds=1),Path(tmp))
            idle.assert_not_called();self.assertEqual(masks,['','']);summary.assert_called_once();self.assertGreaterEqual(fresh.call_count,3)

    def test_intermediate_commit_requires_all_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'logits.npy';data.write_bytes(b'array');path=root/'commit.json'
            write_json(path,dict(status='completed',signature={'case_id':'1'},outputs={'logits':dict(path=str(data),sha256=digest(data))}))
            self.assertTrue(committed(path,{'case_id':'1'},['logits']))
            data.write_bytes(b'changed')
            with self.assertRaises(ValueError):committed(path,{'case_id':'1'},['logits'])


if __name__=='__main__':unittest.main()
