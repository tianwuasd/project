"""Primary choices and explicit new-catalog dependencies; no compute execution."""
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
import pipeline_catalog
from server_data import write_json


class StageBoardTests(unittest.TestCase):
    def setup_board(self,tmp):
        root=Path(tmp);raw=root/'raw';raw.mkdir()
        profile=dict(name='generic',runtime=str(root/'runtime'),results=str(root/'results'),
                     data=str(raw),conda='',gpu_memory=20)
        args=panel.make_parser().parse_args(['--runtime',profile['runtime']])
        return root,profile,args

    def test_preparation_asks_only_source_and_auto_selects_single_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,profile,args=self.setup_board(tmp)
            with patch.object(server_profile,'load',return_value=profile),\
                 patch('builtins.input',side_effect=['']) as user,contextlib.redirect_stdout(io.StringIO()):
                config=panel.stage_config(args,{},'prepare')
            self.assertEqual(user.call_count,1)
            self.assertEqual(config['tasks'],['prepare']);self.assertEqual(config['catalog_schema'],2)
            self.assertEqual(Path(config['results']),root/'results/pipeline_v2/prepare')
            self.assertFalse(config['input_job']);self.assertFalse((root/'runtime').exists())

    def source(self,root,profile,stage):
        path=(root/'results/pipeline_v2'/stage/'source').resolve()
        plan=dict(runtime=profile['runtime'],data=profile['data'],preprocessor='nnunet_default',
                  gpu_memory=20,methods=['B0'],folds=['ct_holdG'],stage=stage,catalog_schema=2)
        rows=[dict(directory=str(path),name='来源任务',stage=stage,status='completed')]
        return path,plan,rows

    def test_training_selects_preparation_then_model_without_raw_or_gpu_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,profile,args=self.setup_board(tmp);source,old,rows=self.source(root,profile,'prepare')
            with patch.object(server_profile,'load',return_value=profile),\
                 patch.object(pipeline_catalog,'history',return_value=rows),patch.object(panel.jobs,'load_plan',return_value=old),\
                 patch('builtins.input',side_effect=['1','B0']) as user,contextlib.redirect_stdout(io.StringIO()):
                config=panel.stage_config(args,{},'train')
            self.assertEqual(user.call_count,2)
            self.assertEqual(config['tasks'],['train'])
            self.assertEqual(config['input_job'],str(source));self.assertEqual(config['upstream_jobs'],[str(source)])
            self.assertEqual(config['data'],profile['data'])
            labels=' '.join(c.args[0] for c in user.call_args_list)
            self.assertNotIn('GPU',labels);self.assertNotIn('数据目录',labels)

    def test_prediction_selects_completed_training_and_only_its_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,profile,args=self.setup_board(tmp);source,old,rows=self.source(root,profile,'train')
            with patch.object(server_profile,'load',return_value=profile),\
                 patch.object(pipeline_catalog,'history',return_value=rows),patch.object(panel.jobs,'load_plan',return_value=old),\
                 patch('builtins.input',side_effect=['1']) as user,contextlib.redirect_stdout(io.StringIO()):
                config=panel.stage_config(args,{},'predict')
            self.assertEqual(user.call_count,1);self.assertEqual(config['tasks'],['predict'])
            self.assertEqual(config['methods'],['B0']);self.assertEqual(config['upstream_jobs'],[str(source)])

    def test_no_source_does_not_silently_add_upstream_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,profile,args=self.setup_board(tmp)
            with patch.object(server_profile,'load',return_value=profile),patch.object(pipeline_catalog,'history',return_value=[]),\
                 patch('builtins.input') as user,self.assertRaisesRegex(ValueError,'已完成的新版预处理'):
                panel.stage_config(args,{},'train')
            user.assert_not_called()

    def test_main_menu_has_separate_boards_and_keeps_all_utilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,profile,args=self.setup_board(tmp);output=io.StringIO()
            with patch.object(server_profile,'load',return_value=profile),patch.object(server_profile,'show'),\
                 patch.object(panel.jobs,'history',return_value=[]),patch.object(panel,'query',return_value='新版尚无任务'),\
                 patch('builtins.input',side_effect=['q']),contextlib.redirect_stdout(output):
                self.assertEqual(panel.panel(args,profile['runtime'],{}),0)
            text=output.getvalue()
            for label in ['1 预处理','2 训练','3 预测','完整测评','r 重启任务','d 删除任务',
                          '更新程序','GitHub网络设置','上传已提交代码','上传日志','查看结果','选择详细日志','显卡','进程','任务概览']:
                self.assertIn(label,text)
            self.assertNotIn('续跑/追加',text);self.assertNotIn('k 独立重跑',text)

    def test_completed_restart_keeps_original_source_and_creates_one_same_stage_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,profile,args=self.setup_board(tmp);job=root/'finished';job.mkdir()
            write_json(job/'state.json',dict(status='completed'))
            plan=dict(stage='train',name='模型B0',tasks=['train'],input_job='original-prepare',upstream_jobs=['original-prepare'])
            with patch.object(panel.jobs,'load_plan',return_value=plan),patch.object(panel,'query',return_value='完成'),\
                 patch.object(panel.jobs,'active_processes',return_value=[]),patch.object(panel,'ask',return_value='RESTART'),\
                 patch.object(panel.jobs,'submit') as submit,patch.object(panel.jobs,'resume') as resume,\
                 contextlib.redirect_stdout(io.StringIO()):
                panel.restart_job(args,profile['runtime'],job)
            config=submit.call_args.args[0]
            self.assertEqual(config['tasks'],['train']);self.assertEqual(config['input_job'],'original-prepare')
            self.assertEqual(config['upstream_jobs'],['original-prepare']);resume.assert_not_called()

    def test_incomplete_restart_resumes_existing_job_with_default_integrity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,profile,args=self.setup_board(tmp);job=root/'interrupted';job.mkdir()
            write_json(job/'state.json',dict(status='failed'))
            with patch.object(panel.jobs,'load_plan',return_value=dict(stage='train')),\
                 patch.object(panel,'query',return_value='失败'),patch.object(panel.jobs,'active_processes',return_value=[]),\
                 patch.object(panel,'ask',return_value='RESTART'),patch.object(panel.jobs,'submit') as submit,\
                 patch.object(panel.jobs,'resume') as resume,contextlib.redirect_stdout(io.StringIO()):
                panel.restart_job(args,profile['runtime'],job)
            resume.assert_called_once_with(job,args.foreground,check='simple');submit.assert_not_called()


if __name__=='__main__':unittest.main()
