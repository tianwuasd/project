import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_panel as panel
import server_profile


class SimplePanelTests(unittest.TestCase):
    def setup_form(self,root):
        raw=root/'raw';raw.mkdir()
        profile=dict(name='generic',runtime=str(root/'runtime'),results=str(root/'results'),
                     data=str(raw),conda='',gpu_memory=20)
        args=panel.make_parser().parse_args(['--runtime',profile['runtime'],'--results',profile['results']])
        return profile,args

    def test_assessment_asks_only_data_and_model_with_single_preprocessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profile,args=self.setup_form(root)
            with patch.object(server_profile,'load',return_value=profile),patch.object(panel.jobs,'history',return_value=[]),\
                 patch('builtins.input',side_effect=['','S1']) as user,\
                 patch('evaluation_gpu.snapshot',side_effect=AssertionError('GPU form query forbidden')),\
                 contextlib.redirect_stdout(io.StringIO()):
                config=panel.simple_config(args,{},kind='assessment')
            self.assertEqual(user.call_count,2)
            self.assertEqual(config['tasks'],['assessment'])
            self.assertEqual(config['gpu_policy'],'automatic')
            self.assertEqual(Path(config['results']),root/'results/assessments')
            self.assertFalse((root/'runtime').exists())

    def test_formal_has_three_blocks_and_no_resource_or_output_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profile,args=self.setup_form(root)
            with patch.object(server_profile,'load',return_value=profile),patch.object(panel.jobs,'history',return_value=[]),\
                 patch('builtins.input',side_effect=['','S1','1,2,3']) as user,contextlib.redirect_stdout(io.StringIO()):
                config=panel.simple_config(args,{})
            self.assertEqual(config['tasks'],['prepare','train','predict'])
            self.assertEqual(user.call_count,3)
            labels='\n'.join(call.args[0] for call in user.call_args_list)
            for forbidden in ['任务名称','结果根目录','GPU','线程','batch']:
                self.assertNotIn(forbidden,labels)

    def test_missing_prepare_is_an_explained_dependency_not_a_fourth_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profile,args=self.setup_form(root)
            with patch.object(server_profile,'load',return_value=profile),patch.object(panel.jobs,'history',return_value=[]),\
                 patch('builtins.input',side_effect=['','S1','2']),contextlib.redirect_stdout(io.StringIO()):
                config=panel.simple_config(args,{})
            self.assertEqual(config['selected_blocks'],['train'])
            self.assertEqual(config['tasks'],['prepare','train'])
            self.assertEqual(config['automatic_prerequisites'],['prepare'])

    def test_prediction_does_not_silently_start_long_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profile,args=self.setup_form(root)
            with patch.object(server_profile,'load',return_value=profile),patch.object(panel.jobs,'history',return_value=[]),\
                 patch('builtins.input',side_effect=['','S1','3']),contextlib.redirect_stdout(io.StringIO()),\
                 self.assertRaisesRegex(ValueError,'已训练模型'):
                panel.simple_config(args,{})


if __name__=='__main__':unittest.main()
