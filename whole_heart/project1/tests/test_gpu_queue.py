"""模拟多卡及子进程：验证调度和隔离，不需要真实GPU，不训练。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import gpu_devices
import gpu_queue
from server_data import read_json, write_json

ROWS = '0, GPU-zero, RTX3090, 23000, 0\n2, GPU-two, RTX3090, 23000, 0\n3, GPU-three, RTX3090, 23000, 0\n'


class GPUSelectionTests(unittest.TestCase):
    def test_count_ids_and_uuid_alias_duplicates(self):
        with patch.object(gpu_devices.subprocess, 'check_output', side_effect=[ROWS, '']):
            self.assertEqual(gpu_devices.select_gpus(2, '0,3'), ['GPU-zero', 'GPU-three'])
        with patch.object(gpu_devices.subprocess, 'check_output', side_effect=[ROWS, '']), \
             patch('builtins.input', side_effect=['2', '0，2']):
            self.assertEqual(gpu_devices.select_gpus(), ['GPU-zero', 'GPU-two'])
        for count, ids in [(2, '0'), (2, '0,GPU-zero'), (0, ''), (4, '0,2,3,4')]:
            with self.subTest(count=count, ids=ids), patch.object(gpu_devices.subprocess, 'check_output', return_value=ROWS):
                with self.assertRaises(ValueError):
                    gpu_devices.select_gpus(count, ids)
        with patch.object(gpu_devices.subprocess, 'check_output', side_effect=[ROWS, 'GPU-three, 123\n']):
            with self.assertRaises(RuntimeError):
                gpu_devices.select_gpus(2, '0,3')


class QueueTests(unittest.TestCase):
    def test_handoff_only_retries_stale_utilization_without_external_process(self):
        cooling = gpu_devices.GPUCoolingDown('stale sample')
        with patch.object(gpu_queue, 'select_gpus', side_effect=[cooling, ['GPU-zero']]) as select, \
             patch.object(gpu_queue.time, 'sleep') as sleep:
            gpu_queue.check_handoff('GPU-zero', True)
            self.assertEqual(select.call_count, 2)
            sleep.assert_called_once_with(1)
        with patch.object(gpu_queue, 'select_gpus', side_effect=RuntimeError('external process')) as select:
            with self.assertRaises(RuntimeError):
                gpu_queue.check_handoff('GPU-zero', True)
            self.assertEqual(select.call_count, 1)
        with patch.object(gpu_queue, 'select_gpus', side_effect=cooling) as select, patch.object(gpu_queue.time, 'sleep'):
            with self.assertRaises(gpu_devices.GPUCoolingDown):
                gpu_queue.check_handoff('GPU-zero', True)
            self.assertEqual(select.call_count, 11)

    def test_sigterm_cleans_owned_job_and_restores_handler(self):
        from unittest.mock import Mock
        process = Mock(pid=123)
        process.poll.return_value = None
        installed = []
        original = object()
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(gpu_queue, 'select_gpus'), \
             patch.object(gpu_queue.subprocess, 'Popen', return_value=process), \
             patch.object(gpu_queue.signal, 'getsignal', return_value=original), \
             patch.object(gpu_queue.signal, 'signal', side_effect=lambda sig, handler: installed.append(handler)), \
             patch.object(gpu_queue.time, 'sleep', side_effect=lambda _: installed[0](gpu_queue.signal.SIGTERM, None)), \
             patch.object(gpu_queue, 'stop_owned_process') as stop:
            with self.assertRaises(InterruptedError):
                gpu_queue.run_training_queue([('ct_holdA', 'B0')], ['GPU-zero'], Path(temp))
            stop.assert_called_once_with(process)
            self.assertIs(installed[-1], original)
            state = read_json(Path(temp) / '00_admin/gpu_queue.json')
            self.assertEqual(state['jobs'][0]['status'], 'interrupted')

    def test_real_cpu_subprocesses_receive_only_assigned_gpu(self):
        # 使用真实子进程验证环境传递和日志，但运行的是微型CPU桩程序。
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / '05_src').mkdir()
            (root / '05_src/server_train.py').write_text(
                'import json, os, sys, time\n'
                'from pathlib import Path\n'
                'time.sleep(0.05)\n'
                'Path(sys.argv[1]+"_"+sys.argv[2]+".json").write_text(json.dumps({"gpu":os.environ["CUDA_VISIBLE_DEVICES"], "nested_queue":os.environ.get("WHOLE_HEART_GPU_UUIDS")}))\n'
                'print("CPU_STUB_COMPLETED")\n', encoding='utf-8')
            with patch.object(gpu_queue, 'CODE', root), patch.object(gpu_queue, 'select_gpus'):
                gpu_queue.run_training_queue([('foldA', 'B0'), ('foldA', 'B1'), ('foldB', 'B0')],
                                             ['GPU-zero', 'GPU-three'], root / 'work')
            state = read_json(root / 'work/00_admin/gpu_queue.json')
            self.assertEqual(state['status'], 'completed')
            for job in state['jobs']:
                observed = read_json(root / f"{job['fold']}_{job['method']}.json")
                self.assertEqual(observed['gpu'], job['gpu_uuid'])
                self.assertIsNone(observed['nested_queue'])
                self.assertIn('CPU_STUB_COMPLETED', Path(job['log']).read_text())

    def test_paired_initialization_does_not_fail_on_running_record(self):
        from server_train import publish_initial_weight
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.json', Path(temp) / 'b.json'
            write_json(b, {'status': 'running'})
            publish_initial_weight(a, {'status': 'running'}, b, 'same')
            publish_initial_weight(b, {'status': 'running'}, a, 'same')
            self.assertEqual(read_json(a)['initial_weight_sha256'], read_json(b)['initial_weight_sha256'])
            with self.assertRaises(ValueError):
                publish_initial_weight(b, {}, a, 'different')

    def simulate(self, root, failure=False):
        instances, calls, tick = [], [], [0]
        class Process:
            def __init__(self, command, **kwargs):
                self.pid = 100 + len(instances)
                self.gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
                # 第二张卡先完成，验证空闲卡能动态领取第三个任务。
                self.end = tick[0] + (3 if not instances else 1)
                self.code = 9 if failure and len(instances) == 1 else 0
                instances.append(self)
                calls.append((command, kwargs, tick[0]))
            def poll(self):
                return self.code if tick[0] >= self.end else None
        def advance(_):
            tick[0] += 1
        with patch.object(gpu_queue.subprocess, 'Popen', side_effect=Process), \
             patch.object(gpu_queue, 'select_gpus'), patch.object(gpu_queue.time, 'sleep', side_effect=advance):
            tasks = [('ct_holdA', 'B0'), ('ct_holdA', 'B1'), ('ct_holdB', 'B0'), ('ct_holdB', 'B1')]
            if failure:
                with self.assertRaises(RuntimeError):
                    gpu_queue.run_training_queue(tasks, ['GPU-zero', 'GPU-three'], root)
            else:
                gpu_queue.run_training_queue(tasks, ['GPU-zero', 'GPU-three'], root)
        return instances, calls, read_json(root / '00_admin/gpu_queue.json')

    def test_dynamic_queue_single_device_per_process(self):
        with tempfile.TemporaryDirectory() as temp:
            processes, calls, state = self.simulate(Path(temp))
            self.assertEqual(len(calls), 4)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual([p.gpu for p in processes[:3]], ['GPU-zero', 'GPU-three', 'GPU-three'])
            self.assertLess(calls[2][2], processes[0].end)
            for i, (command, kwargs, started) in enumerate(calls):
                self.assertIn('--resume', command)
                self.assertNotIn('WHOLE_HEART_GPU_UUIDS', kwargs['env'])
                self.assertTrue(kwargs['start_new_session'])
                for previous in range(i):
                    if processes[previous].gpu == processes[i].gpu:
                        self.assertGreaterEqual(started, processes[previous].end)

    def test_failure_stops_new_dispatch_and_drains_running_job(self):
        with tempfile.TemporaryDirectory() as temp:
            processes, calls, state = self.simulate(Path(temp), failure=True)
            self.assertEqual(len(calls), 2)
            self.assertEqual(state['status'], 'failed')
            self.assertEqual([j['status'] for j in state['jobs']], ['completed', 'failed', 'pending', 'pending'])
            self.assertEqual(processes[0].poll(), 0)

    def test_launch_error_stops_before_training(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(gpu_queue, 'select_gpus', side_effect=RuntimeError('GPU now busy')), \
             patch.object(gpu_queue.subprocess, 'Popen') as spawn:
            with self.assertRaises(RuntimeError):
                gpu_queue.run_training_queue([('ct_holdA', 'B0')], ['GPU-zero'], Path(temp))
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
