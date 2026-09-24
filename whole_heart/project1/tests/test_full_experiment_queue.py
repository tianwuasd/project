"""模拟任务调度：不训练，检查十次完成前绝不调用目标评价。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import server_launch as launcher
import server_data
import evaluation


class QueueTests(unittest.TestCase):
    def test_multigpu_prepares_once_then_waits_before_evaluation(self):
        import gpu_queue
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, work = root / 'data', root / 'work'
            data.mkdir()
            events = []
            args = ['server_launch.py', str(data), '--work-dir', str(work), '--mode', 'experiment', '--yes-train']
            with patch.object(sys, 'argv', args), \
                 patch.object(launcher, 'child', side_effect=lambda script, *args: events.append((script, args))), \
                 patch.object(launcher, 'check_selected_gpus'), patch.object(server_data, 'bind_workspace'), \
                 patch.object(launcher, 'preflight_signature', return_value={'same': True}), \
                 patch.object(evaluation, 'ROOT', work), \
                 patch.dict(launcher.os.environ, {'WHOLE_HEART_GPU_UUIDS': '["GPU-zero", "GPU-three"]'}), \
                 patch.object(gpu_queue, 'run_training_queue', side_effect=lambda tasks, gpus, work, resume: events.append(('queue', tasks))) as queue:
                launcher.main()
                self.assertEqual(queue.call_count, 1)
                self.assertEqual(len(queue.call_args.args[0]), 10)
                self.assertEqual(events[-1][0], 'evaluation.py')
                self.assertEqual(events[-2][0], 'queue')
                events.clear()
                queue.side_effect = RuntimeError('training failed')
                with self.assertRaises(RuntimeError):
                    launcher.main()
                self.assertNotIn('evaluation.py', [e[0] for e in events])

    def run_queue(self, root, fail=False):
        data, work = root / 'data', root / 'work'
        data.mkdir(exist_ok=True)
        calls = []
        def child(script, *args):
            calls.append((script, args))
            if fail and script == 'server_train.py' and args[:2] == ('mr_holdCD', 'B1'):
                raise RuntimeError('simulated interruption')
        args = ['server_launch.py', str(data), '--work-dir', str(work), '--mode', 'experiment', '--yes-train']
        with patch.object(sys, 'argv', args), patch.object(launcher, 'child', side_effect=child), \
             patch.object(launcher, 'runtime_check'), patch.object(server_data, 'bind_workspace'), \
             patch.object(launcher, 'preflight_signature', return_value={'test': 'same'}), \
             patch.object(evaluation, 'ROOT', work), patch.dict(launcher.os.environ):
            if fail:
                with self.assertRaises(RuntimeError):
                    launcher.main()
            else:
                launcher.main()
        return calls

    def test_full_order_and_resume_reuses_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = self.run_queue(root)
            names = [s for s, _ in calls]
            self.assertEqual(names[-1], 'evaluation.py')
            self.assertEqual(names.count('server_train.py'), 10)
            self.assertLess(names.index('run_smoke_suite.py'), names.index('server_train.py'))
            self.assertTrue(all('--resume' in args for script, args in calls if script == 'server_train.py'))
            resumed = self.run_queue(root)
            self.assertNotIn('run_smoke_suite.py', [s for s, _ in resumed])

    def test_last_training_failure_prevents_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = self.run_queue(Path(temp), fail=True)
            self.assertNotIn('evaluation.py', [s for s, _ in calls])


if __name__ == '__main__':
    unittest.main()
