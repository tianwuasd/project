"""Synthetic user journeys and truthful progress; no ML or installation."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
from server_data import write_json
import task_queries as queries
import task_panel as panel


class ProgressTests(unittest.TestCase):
    def fixture(self, root):
        job = root / 'job'
        plan = dict(name='中文基线', workspace=str(job / 'workspace'), tasks=['prepare','train'])
        state = dict(status='running', current_step='installation', started='2026-10-25T01:00:00+00:00',
                     steps={'prepare': {'status':'pending'}, 'train': {'status':'pending'}})
        write_json(job / 'plan.json', plan)
        write_json(job / 'state.json', state)
        return job

    def test_installation_and_step_counts_visible_without_a_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            job = self.fixture(Path(temp))
            write_json(job / 'environment.json', dict(status='running', phase='pytorch', message='安装固定 PyTorch', updated='2026-10-25T01:01:00+00:00'))
            with patch.object(queries, 'active_processes', return_value=[{'pid':1}]):
                text = queries.query('status', Path(temp), job)
            self.assertIn('安装固定 PyTorch', text)
            self.assertIn('0/2', text)
            self.assertIn('数据预处理', text)
            self.assertIn('尚未', text)

    def test_preparation_counts_only_completed_selected_source_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            job = self.fixture(Path(temp)); work = job / 'workspace'
            write_json(work / '04_data/manifests/splits_v1.json', {'folds': {'imagechd_fold0': {'train':['a','b'], 'val':['c'], 'target':['target']}}})
            prep = work / '04_data/derived/nnUNet_preprocessed/Dataset901_imagechd_fold0'
            for case in ('a','target','unrelated'):
                write_json(prep / 'server_case_completion/cache' / (case + '.complete.json'), {'plan_sha256':'ok'})
            from task_progress import preparation_lines
            text = '\n'.join(preparation_lines(work, ['imagechd_fold0']))
            self.assertIn('1/3', text)
            self.assertNotIn('3/3', text)

    def test_failed_environment_still_identifies_the_real_substage(self):
        with tempfile.TemporaryDirectory() as temp:
            job=self.fixture(Path(temp))
            write_json(job/'environment.json',dict(status='failed',phase='verify',message='环境准备未完成',error='pip check failed'))
            text=queries.query('status',temp,job)
            self.assertIn('依赖版本核验',text)
            self.assertIn('pip check failed',text)

    def test_log_selection_and_new_files_and_truncation_are_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            job = self.fixture(Path(temp)); log = job / 'launcher.log'
            log.write_text('x' * 50000 + '\nend\n', encoding='utf-8')
            text = queries.query('logs', temp, job, log_source='0')
            self.assertIn('end', text); self.assertLess(len(text), 15000)
            cursor = queries.LogCursor()
            self.assertIn('end', cursor.read(log))
            self.assertEqual(cursor.read(log), '')
            log.write_text('replaced\n', encoding='utf-8')
            self.assertIn('replaced', cursor.read(log))
            model = job / 'workspace/07_experiments/formal/imagechd_fold0_B0/training.log'
            model.parent.mkdir(parents=True); model.write_text('Epoch 1\n', encoding='utf-8')
            self.assertTrue(any(p == model for _,_,p in queries.log_sources(job)))
            self.assertEqual(queries.query('logs', temp, job, log_source='0').count('Epoch'), 0)

    def test_log_menu_cancel_and_single_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            job = self.fixture(Path(temp)); (job/'launcher.log').write_text('installing\n', encoding='utf-8')
            with patch('builtins.input', side_effect=['0','1','b']), contextlib.redirect_stdout(io.StringIO()) as output:
                panel.log_menu(temp, job)
            self.assertIn('installing', output.getvalue())
            with patch('builtins.input', side_effect=['b']), contextlib.redirect_stdout(io.StringIO()):
                panel.log_menu(temp, job)

    def test_install_environment_form_has_no_data_or_gpu_questions(self):
        args = panel.make_parser().parse_args([])
        with tempfile.TemporaryDirectory() as temp:
            args.runtime = str(Path(temp)/'runtime')
            with patch.object(panel, 'new_config') as create, patch.object(panel.jobs,'submit') as submit, patch('builtins.input',side_effect=['b']), contextlib.redirect_stdout(io.StringIO()):
                panel.environment_menu(args, args.runtime, {})
            create.assert_not_called(); submit.assert_not_called()

    def test_environment_submission_keeps_compute_preferences(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'raw').mkdir()
            config=dict(name='环境准备',runtime=str(root/'runtime'),results=str(root/'results'),data=str(root/'raw'),
                        tasks=['environment'],methods=['S1'],folds=['imagechd_fold0'],gpus=[],gpu_count=0,gpu_memory=5)
            settings=root/'runtime/settings.json'
            write_json(settings, {'name':'训练默认值','tasks':['prepare','smoke']})
            before=settings.read_bytes()
            with patch.object(panel.jobs,'snapshot'), patch.object(panel.jobs,'launch',return_value=0):
                panel.jobs.submit(config)
            self.assertEqual(settings.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
