"""模拟共享服务器，不占用实际GPU：空闲判定、并发上限、失败清理、版本迁移。"""
import json
import os
import signal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import evaluation_gpu as gpu
import evaluation_parallel as queue
from server_data import read_json, write_json


def rows(count=5):
    return [dict(index=str(i), uuid=f'GPU-{i}', free_mib=24000,
                 utilization=0, busy=False) for i in range(count)]


class GPUSelectionTests(unittest.TestCase):
    def setUp(self):
        self.settings = gpu.load_settings()

    def test_process_low_memory_and_transient_load_are_not_idle(self):
        samples = [rows(), rows(), rows()]
        for sample in samples:
            sample[0]['busy'] = True  # 利用率为0但已有任务，不能抢。
            sample[1]['free_mib'] = self.settings['min_free_mib'] - 1
        samples[1][2]['utilization'] = 70  # 一次采样繁忙也必须排除。
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu, 'snapshot', side_effect=samples), patch.object(gpu.time, 'sleep'):
            self.assertEqual(gpu.idle_gpus(self.settings), ['GPU-3', 'GPU-4'])

    def test_allocation_intersection_and_internal_first_gpu_override(self):
        env = dict(CUDA_VISIBLE_DEVICES='GPU-0', WHOLE_HEART_EVAL_VISIBLE_DEVICES='1,2,3', SLURM_JOB_GPUS='2,3,4')
        self.assertEqual(gpu.allowed_ids(rows(), self.settings, env), {'GPU-2', 'GPU-3'})
        self.assertEqual(gpu.allowed_ids(rows(), self.settings, {'CUDA_VISIBLE_DEVICES': ''}), set())
        with self.assertRaises(ValueError):
            gpu.allowed_ids(rows(), self.settings, {'CUDA_VISIBLE_DEVICES': '9'})
        self.assertEqual(gpu.allowed_ids(rows(), dict(self.settings, allowed_gpus=['2']), env), {'GPU-2'})

    def test_failed_query_does_not_report_idle(self):
        with patch.object(gpu, 'query', side_effect=RuntimeError('query failed')):
            with self.assertRaises(RuntimeError):
                gpu.snapshot()

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            write_json(path, dict(self.settings, max_gpus=0))
            with patch.object(gpu, 'CONFIG', path), self.assertRaises(ValueError):
                gpu.load_settings()

    def test_config_accepts_more_than_five_devices_without_fixed_server_assumption(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            write_json(path,dict(self.settings,max_gpus=8))
            with patch.object(gpu,'CONFIG',path):
                self.assertEqual(gpu.load_settings()['max_gpus'],8)


class ParallelQueueTests(unittest.TestCase):
    def setUp(self):
        budget=patch.object(queue,'prediction_budget',side_effect=lambda work,records,maximum:
            dict(workers=maximum,threads_per_worker=1,capacity_status='bounded'))
        budget.start();self.addCleanup(budget.stop)

    def test_sigterm_cleans_only_owned_worker_and_does_not_summarize(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            proc = Mock(pid=100)
            proc.poll.return_value = None
            previous = signal.getsignal(signal.SIGTERM)
            def stop_signal(*args, **kwargs):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                return proc
            with patch.object(queue, 'idle_gpus', return_value=['GPU-0']), \
                 patch.object(queue.subprocess, 'Popen', side_effect=stop_signal), \
                 patch.object(queue, 'stop_owned_process') as stop, patch('evaluation_summary.summarize') as summary:
                with self.assertRaises(InterruptedError):
                    queue.run_phase([dict(fold='imagechd_fold1', method='S1',index=0,case_id='Case1',status='pending')], 'sha', gpu.load_settings(), work)
                stop.assert_called_once_with(proc)
                summary.assert_not_called()
                self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def run_simulation(self, work, failure=False):
        live, peak, processes, masks = set(), [], [], []
        class Process:
            def __init__(self, *args, **kwargs):
                self.pid = 100 + len(processes)
                self.polls = 0
                self.rc = None
                processes.append(self)
                live.add(self.pid)
                peak.append(len(live))
                masks.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                live.discard(self.pid)
                self.rc = 1 if failure and self.pid == 100 else 0
                return self.rc
        def idle(settings, candidates=None):
            return [f'GPU-{i}' for i in range(5)] if candidates is None else list(candidates)
        models = [dict(fold=f'fold{i}', method='S1',index=i,case_id='Case1',status='pending') for i in range(10)]
        with patch.object(queue, 'idle_gpus', side_effect=idle), patch.object(queue.subprocess, 'Popen', Process), \
             patch.object(queue.time, 'sleep'), patch.object(queue.time, 'monotonic', side_effect=range(10000)), \
             patch.object(queue, 'stop_owned_process') as stop, patch('evaluation_summary.summarize') as summarize:
            if failure:
                with self.assertRaises(RuntimeError):
                    queue.run_phase(models, 'sha', gpu.load_settings(), work)
                summarize.assert_not_called()
                self.assertTrue(stop.called)
                self.assertEqual(len(processes), 3)  # 失败后不再启动下一批。
            else:
                queue.run_phase(models, 'sha', gpu.load_settings(), work)
                summarize.assert_not_called()
                self.assertEqual(len(processes), 10)
            self.assertLessEqual(max(peak), 3)
            self.assertEqual(set(masks), {'GPU-0', 'GPU-1', 'GPU-2'})

    def test_ten_jobs_maximum_three_workers_and_single_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            self.run_simulation(work)
            state = read_json(work / '08_results/evaluation/status.json')
            self.assertEqual(state['status'], 'completed')
            self.assertTrue(all(j['status'] == 'completed' for j in state['jobs']))

    def test_failure_stops_owned_workers_and_prevents_final_report(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            self.run_simulation(work, failure=True)
            self.assertEqual(read_json(work / '08_results/evaluation/status.json')['status'], 'failed')

    def test_no_gpu_waits_then_uses_one(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            proc = Mock(pid=123)
            proc.poll.return_value = 0
            with patch.object(queue, 'idle_gpus', side_effect=[[], ['GPU-1']]) as idle, \
                 patch.object(queue.subprocess, 'Popen', return_value=proc) as start, \
                 patch.object(queue.time, 'sleep'), patch.object(queue.time, 'monotonic', side_effect=range(10000)), \
                 patch('evaluation_summary.summarize'):
                queue.run_phase([dict(fold='imagechd_fold1', method='S1',index=0,case_id='Case1',status='pending')], 'sha', gpu.load_settings(), work)
                self.assertEqual(idle.call_count, 2)
                self.assertEqual(start.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-1')

    def test_managed_gpu_waits_for_verified_acquisition_then_admits_new_card(self):
        for initial in [{'GPU-held': {}}, {}]:
            with self.subTest(initially_held=list(initial)), tempfile.TemporaryDirectory() as temp:
                work=Path(temp);lease=work/'diagnostics/gpu-leases.json'
                write_json(lease,dict(error=None))
                restrictions=[];proc=Mock(pid=123);proc.poll.return_value=0
                def idle(settings,candidates=None):
                    self.assertIsNotNone(candidates,'managed idle probe must restrict its candidates')
                    restrictions.append(list(candidates))
                    return [gpu for gpu in ['GPU-new'] if gpu in candidates]
                with patch.dict(os.environ,{'VISTA_GPU_LEASES':str(lease)}),\
                     patch('gpu_fleet.verified_keepers',side_effect=[initial,{'GPU-held':{},'GPU-new':{}}]) as verified,\
                     patch.object(queue,'idle_gpus',side_effect=idle),\
                     patch.object(queue.subprocess,'Popen',return_value=proc) as start,\
                     patch.object(queue.time,'sleep'),patch.object(queue.time,'monotonic',side_effect=range(10000)):
                    queue.run_phase([dict(fold='imagechd_fold0',method='S1',index=0,case_id='Case1',status='pending')],
                        'sha',dict(max_gpus=2,retry_seconds=1),work)
                self.assertEqual(restrictions,[list(initial),['GPU-held','GPU-new']])
                self.assertEqual(verified.call_count,2)
                start.assert_called_once()
                self.assertEqual(start.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-new')

    def test_managed_gpu_reports_fleet_error_before_idle_probe_or_process_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);lease=work/'diagnostics/gpu-leases.json'
            write_json(lease,dict(error='simulated acquisition failure'))
            with patch.dict(os.environ,{'VISTA_GPU_LEASES':str(lease)}),\
                 patch('gpu_fleet.verified_keepers') as verified,\
                 patch.object(queue,'idle_gpus',side_effect=AssertionError('idle probe ran before fleet error')) as idle,\
                 patch.object(queue.subprocess,'Popen') as start,\
                 self.assertRaisesRegex(RuntimeError,'simulated acquisition failure'):
                queue.run_phase([dict(fold='imagechd_fold0',method='S1',index=0,case_id='Case1',status='pending')],
                    'sha',dict(max_gpus=2,retry_seconds=1),work)
            verified.assert_not_called();idle.assert_not_called();start.assert_not_called()
            self.assertEqual(read_json(work/'08_results/evaluation/status.json')['status'],'failed')

    def test_legacy_gpu_queue_keeps_unrestricted_idle_probe_without_fleet(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);proc=Mock(pid=123);proc.poll.return_value=0
            settings=dict(max_gpus=1,retry_seconds=1)
            with patch.dict(os.environ,{},clear=True),patch('cooperative_control.devices') as held,\
                 patch.object(queue,'idle_gpus',return_value=['GPU-free']) as idle,\
                 patch.object(queue.subprocess,'Popen',return_value=proc) as start:
                queue.run_phase([dict(fold='imagechd_fold0',method='S1',index=0,case_id='Case1',status='pending')],
                    'sha',settings,work)
            held.assert_not_called();idle.assert_called_once_with(settings)
            self.assertEqual(start.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-free')

    def test_old_project_compatibility_file_is_not_accepted(self):
        try:import evaluation as ev
        except ModuleNotFoundError:self.skipTest('requires imaging dependencies; no model execution')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = {'old.py': 'aaa'}, {'old.py': 'bbb', 'new.py': 'ccc'}
            write_json(root / '06_configs/evaluation_code_compatibility.json', dict(legacy=[old], release=[new]))
            with patch.object(ev, 'CODE_ROOT', root):
                self.assertFalse(ev.compatible_evaluation_code(old, new))
                self.assertFalse(ev.compatible_evaluation_code(old, dict(new, extra='unknown')))
                self.assertFalse(ev.compatible_evaluation_code({'old.py': 'changed'}, new))
                self.assertTrue(ev.compatible_evaluation_code(new, new))

    def test_environment_mismatch_does_not_rewrite_training_record(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            path = work / '06_configs/pip_freeze_training.txt'
            path.parent.mkdir()
            path.write_text('torch==2.8.0\n', encoding='utf-8')
            before = path.read_bytes()
            with patch.object(queue.subprocess, 'check_output', return_value='torch==2.9.0\n'), self.assertRaises(ValueError):
                queue.verify_environment(work)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
