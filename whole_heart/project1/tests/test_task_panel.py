"""Synthetic control-plane tests. No model training, installation or network calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import task_panel as panel
import task_runtime as jobs
import task_queries as queries
from server_data import read_json, write_json, digest


class PanelTests(unittest.TestCase):
    def test_navigation_preserves_zero_and_cancel(self):
        with patch('builtins.input', return_value='0'):
            self.assertEqual(panel.ask('GPU'), '0')
        for text, error in [('b',panel.Back),('q',panel.Quit)]:
            with patch('builtins.input', return_value=text), self.assertRaises(error):
                panel.ask('名称')

    def test_no_environment_or_gpu_for_status_with_site_packages_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / 'absent'
            result = subprocess.run([sys.executable,'-B','-S',str(Path(panel.__file__)),'--runtime',str(runtime),'--query','status'],
                                    capture_output=True, text=True, encoding='utf-8', env=dict(os.environ,PYTHONUTF8='1'))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(runtime.exists())
            self.assertIn('尚无任务', result.stdout)

    def test_gpu_candidates_respect_allocation_even_when_idle(self):
        rows = [dict(index=str(i),uuid=f'GPU-{i}',busy=False,free_mib=24000,utilization=0) for i in range(3)]
        with patch('evaluation_gpu.snapshot', return_value=rows), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'1'}, clear=True):
            self.assertEqual(panel.resolve_gpus('1'), ['GPU-1'])
            with self.assertRaises(ValueError):
                panel.resolve_gpus('0')
            with self.assertRaises(ValueError):
                panel.resolve_gpus('1,GPU-1')

    def test_cpu_form_does_not_query_gpu_and_keeps_multistep_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'raw').mkdir()
            args=panel.make_parser().parse_args(['--runtime',str(root/'runtime'),'--results',str(root/'results'),
                                               '--data-dir',str(root/'raw'),'--tasks','environment,prepare','--name','中文 任务','--non-interactive'])
            with patch.object(panel,'resolve_gpus') as gpu, patch.object(panel,'query') as query:
                value=panel.new_config(args,{},False)
                gpu.assert_not_called(); query.assert_not_called()
            self.assertEqual(value['tasks'], ['environment','prepare'])
            self.assertEqual(value['gpus'], [])

    def test_multi_model_form_and_explicit_cli_override_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'raw').mkdir()
            args=panel.make_parser().parse_args(['--runtime',str(root/'runtime'),'--results',str(root/'results'),
                '--data-dir',str(root/'raw'),'--tasks','prepare,train,predict','--models','B1','--scope','ct-first',
                '--gpus','0,1','--gpu-count','1','--name','only B1','--non-interactive'])
            with patch.object(panel,'resolve_gpus',return_value=['GPU-0','GPU-1']),patch.object(panel,'query',return_value='GPU'):
                value=panel.new_config(args,{'methods':['B0','B1']},False)
            self.assertEqual(value['methods'],['B1'])
            self.assertEqual(value['gpu_count'],1)
            self.assertEqual(value['tasks'],['prepare','train','predict'])

    def test_result_navigation_and_query_do_not_modify_old_results(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)/'old'; runtime=Path(temp)/'runtime'
            report=work/'09_reports/final_evaluation/summary.json'
            write_json(report,dict(results={'CT_raw':dict(methods={'B1':dict(center_equal_dice=.7,worst_center_dice=.7)},paired=None)}))
            before=(report.read_bytes(),report.stat().st_mtime_ns)
            with contextlib.redirect_stdout(io.StringIO()),patch('builtins.input',side_effect=['1','b','0']):
                panel.result_menu(work)
            text=queries.query('results',runtime,work)
            self.assertIn('B1',text); self.assertIn('不适用',text)
            self.assertEqual(before,(report.read_bytes(),report.stat().st_mtime_ns))
            self.assertFalse(runtime.exists())

    def test_epoch_estimate_uses_completed_timed_epochs_only(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);folder=work/'07_experiments/formal/ct_holdG_B1_seed0'
            write_json(folder/'run.json',dict(status='running',planned_epochs=10))
            (folder/'training.log').write_text('Epoch 0\ntrain_loss 1.0\nEpoch time: 60 s\nEpoch 1\n',encoding='utf-8')
            text='\n'.join(queries.model_lines(work))
            self.assertIn('1/10',text)
            self.assertIn('9.0分钟',text)
            (folder/'training.log').write_text('queue heartbeat\n',encoding='utf-8')
            self.assertIn('无法估计','\n'.join(queries.model_lines(work)))

    def test_prediction_progress_counts_only_recorded_completed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            write_json(work/'08_results/evaluation/frozen_models.json',{'models':[dict(fold='ct_holdG',method='B1',targets=['Case1','Case2'])]})
            write_json(work/'08_results/evaluation/ct_holdG_B1/Case1.json',dict(status='completed',seconds=60))
            write_json(work/'08_results/evaluation/ct_holdG_B1/Case2.json',dict(status='failed',seconds=1))
            text='\n'.join(queries.prediction_lines(work))
            self.assertIn('1/2',text);self.assertIn('1.0分钟',text)


class JobTests(unittest.TestCase):
    def config(self, root):
        (root/'raw').mkdir(exist_ok=True)
        return dict(name='同名 任务',runtime=str(root/'runtime'),results=str(root/'results'),data=str(root/'raw'),
                    tasks=['environment'],methods=['B1'],folds=['ct_holdG'],gpus=[],gpu_count=0,gpu_memory=5,input_job=None)

    def test_same_names_unique_directories_and_immutable_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); config=self.config(root)
            with patch.object(jobs,'snapshot'),patch.object(jobs,'launch',return_value=0):
                jobs.submit(config); jobs.submit(config)
            entries=jobs.history(config['runtime'])
            self.assertEqual(len(entries),2)
            self.assertNotEqual(entries[0]['directory'],entries[1]['directory'])
            for row in entries:
                self.assertEqual(jobs.load_plan(row['directory'])['methods'],['B1'])
            plan=Path(entries[0]['directory'])/'plan.json'
            value=read_json(plan); value['methods']=['B0']; write_json(plan,value)
            with self.assertRaises(ValueError):jobs.load_plan(plan.parent)

    def test_history_tolerates_broken_neighbour_without_changing_last(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); config=self.config(root)
            with patch.object(jobs,'snapshot'),patch.object(jobs,'launch',return_value=0):jobs.submit(config)
            pointer=Path(config['runtime'])/'last-job.json'; before=pointer.read_bytes()
            index=Path(config['runtime'])/'history.json'
            values=read_json(index); values['jobs'].append(str(root/'bad')); write_json(index,values)
            self.assertEqual(len(jobs.history(config['runtime'])),2)
            self.assertEqual(pointer.read_bytes(),before)

    def test_stop_only_matches_identity_and_deduplicates(self):
        records=[dict(pid=41,identity={'start':'one'}),dict(pid=41,identity={'start':'one'}),dict(pid=42,identity={'start':'old'})]
        with patch.object(jobs,'active_processes',return_value=records),patch.object(jobs,'alive',side_effect=lambda r:r['pid']==41),patch.object(jobs.os,'kill') as kill:
            jobs.stop('unused')
            self.assertEqual(kill.call_count,1)
            self.assertEqual(kill.call_args.args[0],41)

    def test_lock_contention_does_not_create_task_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); config=self.config(root)
            with jobs.runtime_lock(config['runtime']):
                with self.assertRaises((OSError,ValueError)):
                    jobs.submit(config)
            self.assertFalse((root/'results').exists())


if __name__ == '__main__':
    unittest.main()
