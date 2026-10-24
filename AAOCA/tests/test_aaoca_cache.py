"""Synthetic cache/template regression coverage; no real server or CT paths read."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from aaoca_preprocessing.config import Config
from aaoca_preprocessing.io import RunLock, discover
from aaoca_preprocessing.pipeline import run
from aaoca_server import defaults
from aaoca_server.cache import CacheMismatch, inspect_cache
from aaoca_server.panel import Back, Exit, Panel
from aaoca_server.state import read, write
from aaoca_server.state import Registry


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runtime = self.root / 'panel'

    def test_defaults_preserve_saved_settings_and_explicit_overrides(self):
        fresh = Panel(self.runtime)
        self.assertEqual(fresh.settings['input'], '/data_nas/zhangruichen/AAOCA')
        self.assertEqual(fresh.settings['results'], '/data_nas/zhangruichen/aaoca_preprocessed')
        self.assertEqual(fresh.settings['cache_mode'], 'reuse')
        self.assertFalse(self.runtime.exists())
        write(self.runtime / 'settings.json', {'input': 'saved-input', 'results': 'saved-results'})
        panel = Panel(self.runtime, {'input': 'explicit-input'})
        self.assertEqual(panel.settings['input'], 'explicit-input')
        self.assertEqual(panel.settings['results'], 'saved-results')

    def test_default_runtime_keeps_legacy_history_but_explicit_env_wins(self):
        legacy = self.root / 'legacy'
        preferred = self.root / 'preferred'
        with patch.object(defaults, 'LEGACY_RUNTIME', legacy), patch.object(defaults, 'SERVER_RUNTIME', preferred), \
             patch.dict('os.environ', {}, clear=True):
            self.assertEqual(defaults.runtime_default(), str(preferred))
            write(legacy / 'history.json', [{'name': 'old'}])
            self.assertEqual(defaults.runtime_default(), str(legacy))
            with patch.dict('os.environ', {'AAOCA_SERVER_RUNTIME': 'explicit'}):
                self.assertEqual(defaults.runtime_default(), 'explicit')

    def test_navigation_zero_is_a_value_and_cancel_discards_form(self):
        answers = iter(['0', '返回', '退出'])
        panel = Panel(self.runtime, ask=lambda _: next(answers), emit=lambda _: None)
        self.assertEqual(panel.prompt(''), '0')
        with self.assertRaises(Back): panel.prompt('')
        with self.assertRaises(Exit): panel.prompt('')
        answers = iter(['1', 'new-input', 'b'])
        panel.ask = lambda _: next(answers)
        with self.assertRaises(Back): panel.configure()
        self.assertEqual(panel.settings['input'], defaults.INPUT)
        self.assertFalse(self.runtime.exists())

    def test_legacy_names_and_nonrecent_results_are_read_only(self):
        write(self.runtime / 'history.json', [{'output': str(self.root / 'new')},
                                            {'output': str(self.root / 'old'), 'name': '旧任务'}])
        before = (self.runtime / 'history.json').read_bytes()
        answers = iter(['2'])
        panel = Panel(self.runtime, ask=lambda _: next(answers), emit=lambda _: None)
        self.assertEqual(panel.selected['name'], 'new')
        panel.results()
        self.assertEqual(panel.selected['name'], '旧任务')
        self.assertEqual((self.runtime / 'history.json').read_bytes(), before)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'input'; self.source.mkdir()
        self.output = self.root / 'output'
        self.runtime = self.root / 'output.runtime'; self.runtime.mkdir()
        for name in ['a', 'b']:
            image = nib.Nifti1Image(np.arange(216, dtype=np.float32).reshape(6, 6, 6) - 1100,
                                   np.diag([.7, .8, 1., 1.]))
            image.header.set_xyzt_units('mm')
            nib.save(image, self.source / (name + '.nii.gz'))
        self.config = Config(target_spacing=(.7, .8, 1.), max_case_gib=1, memory_budget_gib=.1)
        self.cases = discover(self.source)
        inputs = [{'input': c['input'], 'sha256': None, 'size': c['source'].stat().st_size,
                   'mtime_ns': c['source'].stat().st_mtime_ns} for c in self.cases]
        self.spec = dict(input=str(self.source), output=str(self.output), config=asdict(self.config),
                         config_sha256=self.config.signature(), inputs=inputs)
        write(self.runtime / 'job.json', self.spec)
        write(self.runtime / 'inputs.json', inputs)
        run(self.source, self.output, self.config)
        write(self.runtime / 'worker.json', {'state': 'completed'})
        self.settings = {'input': str(self.source), 'manifest': ''}

    def check(self):
        return inspect_cache(self.output, self.settings, self.config, emit=lambda _: None)

    def panel(self, answers=('y',)):
        runtime = self.root / 'panel'
        Registry(runtime).register({'name': '原任务', 'output': str(self.output)})
        iterator = iter(answers)
        return Panel(runtime, overrides={'input': str(self.source), 'results': str(self.root / 'new-results'),
                                        'spacing': list(self.config.target_spacing)},
                     ask=lambda _: next(iterator), emit=lambda _: None)

    def test_complete_reuse_does_not_submit_duplicate_job(self):
        panel = self.panel()
        with patch('aaoca_preprocessing.jobs.start', side_effect=AssertionError('duplicate')):
            self.assertTrue(panel.launch('新名称', panel.settings))
        self.assertEqual(panel.selected['name'], '原任务')
        self.assertEqual(len(panel.registry.entries()), 1)
        self.assertFalse((self.root / 'new-results').exists())
        self.assertEqual(read(panel.registry.root / 'last-cache-selection.json')['action'], 'reuse')

    def test_mismatch_falls_back_once_to_new_directory_and_preserves_original(self):
        panel = self.panel()
        panel.settings['spacing'] = [1, 1, 1]
        before = {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        def fake_start(source, output, *args, **kwargs):
            return {'runtime': str(output) + '.runtime'}
        with patch('aaoca_preprocessing.jobs.start', side_effect=fake_start) as start:
            panel.launch('新名称', panel.settings)
        self.assertEqual(start.call_count, 1)
        self.assertNotEqual(panel.selected['output'], str(self.output))
        self.assertTrue(panel.selected['cache_selection']['reason'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()})

    def test_system_error_never_creates_fallback_job(self):
        panel = self.panel()
        with patch('aaoca_server.cache.file_hash', side_effect=PermissionError('denied')), \
             patch('aaoca_preprocessing.jobs.start') as start:
            with self.assertRaises(PermissionError): panel.launch('新名称', panel.settings)
        start.assert_not_called()
        self.assertEqual(len(panel.registry.entries()), 1)

    def test_partial_cache_resumes_original_without_overwriting_request(self):
        write(self.output / 'progress.json', {'state': 'stopped'})
        panel = self.panel()
        before = (self.runtime / 'job.json').read_bytes()
        with patch('aaoca_server.updater.resume_saved') as resume:
            panel.launch('新名称', panel.settings)
        self.assertEqual(resume.call_args.args[0]['output'], str(self.output))
        self.assertEqual((self.runtime / 'job.json').read_bytes(), before)

    def test_rerun_always_uses_new_output_and_frozen_source(self):
        panel = self.panel(('1', '重复名称', 's', 'y'))
        def fake_start(source, output, *args, **kwargs):
            return {'runtime': str(output) + '.runtime'}
        with patch('aaoca_preprocessing.jobs.start', side_effect=fake_start) as start:
            panel.rerun()
        self.assertEqual(start.call_args.args[0], str(self.source))
        self.assertNotEqual(str(start.call_args.args[1]), str(self.output))
        self.assertEqual(start.call_args.args[3], str(self.runtime / 'inputs.json'))
        self.assertEqual(panel.selected['name'], '重复名称')

    def test_complete_cache_checks_both_images_and_does_not_rewrite_outputs(self):
        before = {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(self.check()['action'], 'reuse')
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob('*') if p.is_file()})
        case = self.output / 'cases' / self.cases[0]['case_id']
        with (case / 'image_hu.nii.gz').open('ab') as stream: stream.write(b'broken')
        with self.assertRaisesRegex(CacheMismatch, 'HU'): self.check()

    def test_partial_cache_can_resume_without_changing_frozen_inputs(self):
        case = self.output / 'cases' / self.cases[1]['case_id']
        for file in case.iterdir(): file.unlink()
        case.rmdir()
        write(self.output / 'progress.json', {'state': 'stopped'})
        frozen = (self.runtime / 'job.json').read_bytes()
        self.assertEqual(self.check()['action'], 'resume')
        self.assertEqual((self.runtime / 'job.json').read_bytes(), frozen)

    def test_source_recipe_or_manifest_changes_reject_reuse(self):
        changed = Config(target_spacing=(1, 1, 1))
        with self.assertRaisesRegex(CacheMismatch, '参数|实现'):
            inspect_cache(self.output, self.settings, changed)
        write(self.runtime / 'inputs.json', [])
        with self.assertRaisesRegex(CacheMismatch, '清单'): self.check()

    def test_source_sha_change_same_stat_is_detected(self):
        record_path = self.output / 'cases' / self.cases[0]['case_id'] / 'record.json'
        record = read(record_path)
        record['input_sha256'] = '0' * 64
        write(record_path, record)
        with self.assertRaisesRegex(CacheMismatch, '来源'): self.check()

    def test_invalid_raw_manifest_permission_or_live_lock_do_not_fallback(self):
        manifest = self.root / 'invalid.json'; manifest.write_text('not json')
        with self.assertRaises(json.JSONDecodeError):
            inspect_cache(self.output, dict(self.settings, manifest=str(manifest)), self.config)
        with patch('aaoca_server.cache.file_hash', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError): self.check()
        with RunLock(self.output):
            with self.assertRaises(RuntimeError): self.check()
        with patch('aaoca_server.cache.job_status', return_value={'alive': True}):
            with self.assertRaises(RuntimeError): self.check()

    def test_current_manifest_hash_failure_is_raw_input_error(self):
        manifest = self.root / 'wrong-hash.json'
        write(manifest, [{'input': c['input'], 'sha256': '0' * 64} for c in self.cases])
        with self.assertRaises(ValueError) as error:
            inspect_cache(self.output, dict(self.settings, manifest=str(manifest)), self.config)
        self.assertNotIsInstance(error.exception, CacheMismatch)

    def test_recipe_mismatch_cannot_hide_live_cache_locks(self):
        panel = self.panel(('y', 'y'))
        panel.settings['spacing'] = [1, 1, 1]
        for directory in [self.output, self.runtime]:
            with self.subTest(directory=directory):
                with RunLock(directory), patch('aaoca_preprocessing.jobs.start') as start:
                    with self.assertRaises(RuntimeError): panel.launch('新名称', panel.settings)
                start.assert_not_called()
                self.assertEqual(len(panel.registry.entries()), 1)

    def test_recipe_mismatch_cannot_hide_bad_current_manifest_hash(self):
        panel = self.panel()
        panel.settings['spacing'] = [1, 1, 1]
        manifest = self.root / 'wrong-hash.json'
        write(manifest, [{'input': c['input'], 'sha256': '0' * 64} for c in self.cases])
        panel.settings['manifest'] = str(manifest)
        with patch('aaoca_preprocessing.jobs.start') as start:
            with self.assertRaises(ValueError) as error: panel.launch('新名称', panel.settings)
        self.assertNotIsInstance(error.exception, CacheMismatch)
        start.assert_not_called()
        self.assertEqual(len(panel.registry.entries()), 1)

    def test_recipe_mismatch_cannot_hide_source_permission_failure(self):
        panel = self.panel()
        panel.settings['spacing'] = [1, 1, 1]
        with patch('aaoca_server.cache.file_hash', side_effect=PermissionError('denied')), \
             patch('aaoca_preprocessing.jobs.start') as start:
            with self.assertRaises(PermissionError): panel.launch('新名称', panel.settings)
        start.assert_not_called()
        self.assertEqual(len(panel.registry.entries()), 1)

    def test_current_manifest_frozen_stat_failure_never_reuses_or_falls_back(self):
        panel = self.panel()
        manifest = self.root / 'wrong-stat.json'
        write(manifest, [{'input': c['input'], 'size': 1, 'mtime_ns': 1} for c in self.cases])
        panel.settings['manifest'] = str(manifest)
        with patch('aaoca_preprocessing.jobs.start') as start:
            with self.assertRaises(ValueError) as error: panel.launch('新名称', panel.settings)
        self.assertNotIsInstance(error.exception, CacheMismatch)
        start.assert_not_called()
        self.assertEqual(len(panel.registry.entries()), 1)


if __name__ == '__main__':
    unittest.main()
