"""New tasks remain valid when the panel runs from an isolated update release."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import task_runtime as jobs
import task_panel as panel
import runtime_setup


class ManagedReleasePaths(unittest.TestCase):
    def test_release_handoff_preserves_runtime_resolved_from_legacy_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=root/'original-runtime';legacy=root/'checkout-settings.json'
            def optional(path):return {'base':str(runtime)} if Path(path)==legacy else None
            with patch.object(runtime_setup,'SETTINGS',legacy), \
                 patch.object(runtime_setup,'DEFAULT_PATHS',{'base':str(root/'unused-runtime')}), \
                 patch.object(panel,'read_optional',side_effect=optional), \
                 patch('task_update.selected_entry',return_value=root/'release/05_src/task_panel.py'), \
                 patch.object(panel.subprocess,'call',return_value=0) as launch:
                panel.main(['--profile','generic','--query','history'])
            command=launch.call_args.args[0]
            self.assertIn('--runtime',command)
            self.assertEqual(command[command.index('--runtime')+1],str(runtime.resolve()))
    def fixture(self, root, name='a' * 40, marker=True):
        raw = root / 'raw'
        raw.mkdir()
        runtime = root / 'runtime'
        code = runtime / 'releases' / name
        code.mkdir(parents=True)
        if marker:
            (code / 'release_manifest.json').write_text('{}', encoding='utf-8')
        for name in ['05_src/task_panel.py', '05_src/task_worker.py', '06_configs/baseline.json',
                     '04_data/manifests/README.md', 'bootstrap_server.sh']:
            path = code / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# synthetic frozen source\n')
        config = dict(name='task4', runtime=str(runtime), results=str(root/'results'), data=str(raw),
                      tasks=['environment','prepare'], methods=['B0'], folds=['ct_holdG'],
                      gpus=[], gpu_count=0, gpu_memory=20, input_job=None)
        return config, code

    def test_managed_release_can_submit_into_its_own_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            config, code = self.fixture(Path(temp))
            with patch.object(jobs, 'CODE', code):
                self.assertEqual(jobs.validate_request(config)['name'], 'task4')
                with patch.object(jobs, 'launch', return_value=0):
                    self.assertEqual(jobs.submit(config), 0)
                latest = jobs.latest(config['runtime'])
                self.assertIsNotNone(latest)
                self.assertEqual(jobs.load_plan(latest)['name'], 'task4')
                self.assertEqual(jobs.history(config['runtime'])[0]['name'], 'task4')
                self.assertEqual(json.loads((latest/'code_manifest.json').read_text())[str(Path('05_src')/'task_panel.py')],
                                 jobs.digest(code/'05_src/task_panel.py'))

    def test_release_layout_does_not_relax_results_or_data_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            config, code = self.fixture(Path(temp))
            with patch.object(jobs, 'CODE', code):
                for results in [code, code/'outputs', code.parent, code.parent.parent]:
                    with self.subTest(results=results), self.assertRaises(ValueError):
                        jobs.validate_request(dict(config, results=str(results)))
                with self.assertRaises(ValueError):
                    jobs.validate_request(dict(config, results=config['data']))

    def test_unmanaged_code_ancestors_remain_invalid_runtimes(self):
        for name, marker in [('not-a-commit', True), ('a'*40, False)]:
            with self.subTest(name=name, marker=marker), tempfile.TemporaryDirectory() as temp:
                config, code = self.fixture(Path(temp), name, marker)
                with patch.object(jobs, 'CODE', code), self.assertRaisesRegex(ValueError, '专用子目录'):
                    jobs.validate_request(config)
        with tempfile.TemporaryDirectory() as temp:
            config, code = self.fixture(Path(temp))
            checkout = Path(temp)/'checkout'/'project'
            checkout.mkdir(parents=True)
            with patch.object(jobs, 'CODE', checkout), self.assertRaisesRegex(ValueError, '专用子目录'):
                jobs.validate_request(dict(config, runtime=str(checkout.parent)))


if __name__ == '__main__':
    unittest.main()
