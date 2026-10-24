"""Synthetic workbench/resource tests. Never reference research datasets."""
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from aaoca_server.panel import Panel, parse_setting
from aaoca_server.state import (
    Registry,
    dispatch_lock,
    job_status,
    read,
    validate_paths,
    write,
)

try:
    import nibabel as nib
    import numpy as np

    from aaoca_preprocessing import jobs
    from aaoca_preprocessing.autoresources import select_resources
    from aaoca_preprocessing.config import Config
    from aaoca_preprocessing.resources import allocation
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / 'panel'
        self.messages = []

    def panel(self, answers, **kwargs):
        answers = iter(answers)
        return Panel(self.runtime, ask=lambda _: next(answers), emit=self.messages.append, **kwargs)

    def test_exit_default_query_and_back_are_read_only(self):
        panel = self.panel(['', 'c', 'b', 'n', 'b', 'e', 'q'])
        self.assertEqual(panel.run(), 0)
        self.assertFalse(self.runtime.exists())
        self.assertTrue(any('暂无任务' in m for m in self.messages))

    def test_help_and_empty_status_work_without_site_packages(self):
        entry = Path(__file__).resolve().parents[1] / 'start_aaoca_server.py'
        for flag, expected in [('--help', 0), ('--status', 1)]:
            result = subprocess.run([sys.executable, '-S', str(entry), '--runtime', str(self.runtime), flag],
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, expected, result.stderr)
        self.assertFalse(self.runtime.exists())

    def test_parser_keeps_auto_and_rejects_invalid_numbers(self):
        for field in ['cpu_limit', 'memory_budget_gib', 'workers', 'max_case_gib']:
            self.assertIsNone(parse_setting(field, 'auto'))
            for text in ['-1', 'nan', 'inf']:
                with self.assertRaises(ValueError):
                    parse_setting(field, text)
        self.assertEqual(parse_setting('spacing', '.5 .5 1'), [.5, .5, 1])
        for value in ['.5 .5', '0 1 1', 'auto bad']:
            with self.assertRaises(ValueError): parse_setting('spacing', value)

    def test_config_save_does_not_launch(self):
        source = self.root / 'source'; source.mkdir()
        panel = self.panel(['c', '1', str(source), '2', str(self.root / 'results'), 'w', 'b', 'q'])
        with patch('aaoca_server.panel.prepare', side_effect=AssertionError('unexpected planning')):
            panel.run()
        self.assertEqual(read(self.runtime / 'settings.json')['spacing'], 'auto')
        self.assertEqual(Registry(self.runtime).entries(), [])

    def test_nested_paths_are_rejected_without_creating_any_files(self):
        source = self.root / 'source'; source.mkdir()
        for results, runtime in [(source / 'out', self.runtime), (self.root / 'out', source / 'panel'),
                                 (self.root / 'out', self.root / 'out/panel')]:
            with self.assertRaises(ValueError):
                validate_paths({'input': str(source), 'results': str(results)}, runtime)
        self.assertFalse(self.runtime.exists())

    def test_dispatch_lock_serializes_terminals(self):
        with dispatch_lock(self.runtime):
            with self.assertRaises(RuntimeError):
                with dispatch_lock(self.runtime): pass
        with dispatch_lock(self.runtime): pass

    def test_live_or_unknown_job_prevents_second_submission(self):
        registry = Registry(self.runtime)
        registry.register({'output': str(self.root / 'out'), 'name': '任务'})
        for alive in (True, None):
            with patch('aaoca_server.state.job_status', return_value={'alive': alive}):
                with self.assertRaises(RuntimeError): registry.ensure_idle()

    def test_abandoned_worker_is_reported_interrupted(self):
        output = self.root / 'out'
        write(self.root / 'out.runtime/worker.json', {'state': 'running'})
        self.assertEqual(job_status(output)['state'], 'interrupted')

    def test_tail_and_history_do_not_mutate_existing_state(self):
        output = self.root / 'out'
        write(self.runtime / 'history.json', [{'name': '甲任务', 'output': str(output), 'created_at': 'today'}])
        log = self.root / 'out.runtime/worker.log'; log.parent.mkdir()
        log.write_text('hello\n' * 30000, encoding='utf-8')
        panel = self.panel(['h', '1', 'l', 'o', 'q'])
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        panel.run()
        after = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertLess(max(len(m) for m in self.messages), 66000)

    def test_log_ctrl_c_returns_without_stopping_worker(self):
        output = self.root / 'out'
        write(self.runtime / 'history.json', [{'name': '任务', 'output': str(output)}])
        log = self.root / 'out.runtime/worker.log'; log.parent.mkdir()
        log.write_text('running\n')
        panel = self.panel([])
        with patch('aaoca_server.panel.job_status', return_value={'alive': True, 'runtime': str(log.parent)}), \
             patch('aaoca_server.panel.time.sleep', side_effect=KeyboardInterrupt):
            panel.logs(True)
        self.assertFalse((log.parent / 'stop.request').exists())


@unittest.skipUnless(HAS_IMAGING, 'Install requirements-preprocess.txt')
class AutomaticTests(unittest.TestCase):
    setUp = PanelTests.setUp
    panel = PanelTests.panel
    def image(self, name='a.nii.gz'):
        source = self.root / 'source'; source.mkdir(exist_ok=True)
        image = nib.Nifti1Image(np.linspace(-1500, 1500, 11*13*17, dtype=np.float32).reshape(11, 13, 17),
                               np.diag([.6, .6, .25, 1]))
        image.header.set_xyzt_units('mm')
        nib.save(image, source / name)
        return source

    def simulated(self, sizes, cpu=128, memory=240, request=None):
        report = {'cases': [{'status': 'ready', 'estimated_working_bytes': s * 2**30} for s in sizes]}
        limits = {'cpu_budget': cpu, 'available_memory_bytes': memory * 2**30,
                  'memory_budget_bytes': int(.9 * memory * 2**30)}
        with patch('aaoca_preprocessing.autoresources.allocation', return_value=limits), \
             patch('aaoca_preprocessing.autoresources.idle_fraction', return_value=1), \
             patch('aaoca_preprocessing.autoresources.plan', return_value=report):
            return select_resources('unused', (.5, .5, .5), request)

    def test_idle_machine_uses_high_parallelism_and_large_case_budget(self):
        config, evidence, _ = self.simulated([2] * 100)
        self.assertEqual(config.cpu_limit, 115)
        self.assertEqual(config.workers, 32)
        self.assertEqual(config.compute_threads, 3)
        self.assertEqual(config.memory_budget_gib, 216)
        self.assertGreater(config.max_case_gib, 8)
        self.assertEqual(evidence['oversized_cases'], 0)

    def test_large_and_small_cases_share_queue_without_global_max_penalty(self):
        config, evidence, _ = self.simulated([1] * 19 + [27], cpu=64, memory=128)
        self.assertGreater(config.workers, 4)
        self.assertEqual(evidence['oversized_cases'], 0)

    def test_unavailable_memory_never_admits_large_case(self):
        config, evidence, _ = self.simulated([27, 2], memory=16)
        self.assertEqual(evidence['oversized_cases'], 1)
        self.assertLessEqual(config.max_case_gib, 14.4)

    def test_auto_works_with_fractional_cpu_quota(self):
        config, _, _ = self.simulated([1, 1], cpu=.5, memory=16)
        self.assertEqual((config.workers, config.compute_threads), (1, 1))

    def test_manual_settings_are_validated_and_do_not_change_numerical_recipe(self):
        config, _, _ = self.simulated([2] * 10, cpu=12, memory=64,
                                     request={'cpu_limit': 12, 'workers': 4, 'compute_threads': 2, 'read_workers': 2})
        self.assertEqual((config.workers, config.compute_threads, config.read_workers), (4, 2, 2))
        self.assertEqual(config.signature(), Config(target_spacing=(.5, .5, .5)).signature())
        for request in [{'workers': 0}, {'cpu_limit': -1}, {'memory_budget_gib': float('inf')}, {'mystery': 1}]:
            with self.assertRaises(ValueError): self.simulated([1], request=request)

    def test_real_allocator_obeys_manual_caps_and_ninety_percent_memory(self):
        config = Config(workers=32, cpu_limit=2, memory_fraction=.9, memory_budget_gib=.1)
        result = allocation(config)
        self.assertLessEqual(result['cpu_budget'], 2)
        self.assertLessEqual(result['memory_budget_bytes'], .1 * 2**30)
        self.assertLessEqual(result['memory_budget_bytes'], .9 * result['available_memory_bytes'])

    def test_preview_and_cancel_produce_no_runtime_or_output(self):
        source = self.image()
        panel = self.panel(['n', '测试', '1', 'p', 's', 'no', 'b', 'q'],
                           overrides={'input': str(source), 'results': str(self.root / 'results')})
        with patch('aaoca_preprocessing.jobs.start', side_effect=AssertionError('must not launch')):
            panel.run()
        self.assertFalse(self.runtime.exists())
        self.assertFalse((self.root / 'results').exists())
        self.assertTrue(any('0.6 × 0.6 × 0.25' in m for m in self.messages))

    def wait(self, output):
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            status = jobs.status(output)
            if not status['process_alive']:
                return status
            time.sleep(.1)
        jobs.stop(output)
        self.fail('synthetic background job timed out')

    def test_full_menu_start_then_resume_preserves_spacing_and_pixels(self):
        source = self.image()
        self.image('b.nii.gz')
        panel = self.panel(['n', '合成任务', '', 's', 'y', 'q'],
                           overrides={'input': str(source), 'results': str(self.root / 'results')})
        panel.run()
        entries = Registry(self.runtime).entries()
        self.assertEqual(len(entries), 1, self.messages)
        output = Path(entries[0]['output'])
        self.addCleanup(lambda: jobs.stop(output) if jobs.status(output)['process_alive'] else None)
        status = self.wait(output)
        self.assertEqual(status['worker']['state'], 'completed', status)
        runtime = jobs.runtime_path(output)
        spec = read(runtime / 'job.json')
        first = read(runtime / 'resources.json')
        self.assertIn('resource_request', spec)
        np.testing.assert_allclose(spec['config']['target_spacing'], [.6, .6, .25])
        self.assertEqual(spec['config']['target_spacing'], first['selected']['target_spacing'])
        panel = self.panel(['r', '1', 'y', 'q'])
        panel.run()
        finished = self.wait(output)
        self.assertEqual(finished['worker']['reused'], 2, finished)
        second = read(runtime / 'resources.json')
        self.assertNotEqual(first['attempt'], second['attempt'])
        self.assertEqual(read(runtime / 'job.json'), spec)
        for path in (output / 'cases').glob('*/image.nii.gz'):
            array = nib.load(path).get_fdata()
            self.assertGreaterEqual(array.min(), -1)
            self.assertLessEqual(array.max(), 1)


if __name__ == '__main__':
    unittest.main()
