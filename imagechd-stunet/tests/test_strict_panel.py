import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_panel as panel
import task_runtime as jobs
import server_profile


class StrictPanelTests(unittest.TestCase):
    def rows(self):
        return [dict(index=str(i),uuid=f'GPU-{i}',free_mib=24000,utilization=0,busy=False,name='RTX3090') for i in range(3)]

    def test_auto_pool_uses_only_inherited_allocation(self):
        with patch('evaluation_gpu.snapshot',return_value=self.rows()), patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,2'},clear=True):
            self.assertEqual(panel.resolve_gpus('auto'),['GPU-1','GPU-2'])

    def test_pool_can_be_smaller_than_concurrency_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'raw').mkdir()
            config=dict(name='少卡先运行',runtime=str(root/'runtime'),results=str(root/'results'),data=str(root/'raw'),tasks=['prepare','train'],methods=['S1'],folds=['imagechd_fold0'],gpus=['GPU-1'],gpu_count=3,gpu_memory=5)
            self.assertEqual(jobs.validate_request(config)['gpu_count'],3)

    def test_task_form_orders_name_paths_count_then_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'raw').mkdir()
            args=panel.make_parser().parse_args(['--runtime',str(root/'runtime'),'--profile','generic'])
            prompts=[]
            def answer(label,default=None):
                prompts.append(label)
                if label.startswith('任务显示名称'):return 'B0 测试'
                if '步骤' in label:return 'prepare,train'
                if '结果' in label:return str(root/'results')
                if '原始' in label:return str(root/'raw')
                if '最多' in label:return '2'
                if '候选' in label:return 'auto'
                return default
            with patch.object(panel,'ask',side_effect=answer),patch.object(panel,'choose_job',return_value=None),patch.object(panel,'show_gpu_choices'),patch.object(panel,'resolve_gpus',return_value=['GPU-1']):
                config=panel.new_config(args,{})
            name=next(i for i,x in enumerate(prompts) if '名称' in x)
            data=next(i for i,x in enumerate(prompts) if '原始' in x)
            count=next(i for i,x in enumerate(prompts) if '最多' in x)
            pool=next(i for i,x in enumerate(prompts) if '候选' in x)
            self.assertLess(name,data);self.assertLess(data,count);self.assertLess(count,pool)
            self.assertEqual(config['gpu_count'],2)

    def test_known_paths_are_visible_and_cli_wins(self):
        profile=server_profile.load('zmic44')
        self.assertEqual(profile['data'],'/data_nas/zhangruichen/baidu_import/Wholeheart_Train_Dataset')
        args=panel.make_parser().parse_args(['--runtime','/chosen/runtime','--data-dir','/chosen/data'])
        values=server_profile.paths(profile,{'data':'/old'},args)
        self.assertEqual(values['data'],'/chosen/data')
        self.assertEqual(values['runtime'],'/chosen/runtime')
        with contextlib.redirect_stdout(io.StringIO()) as output:
            server_profile.show(profile,values)
        self.assertIn('/chosen/runtime',output.getvalue())
        self.assertIn('预处理缓存',output.getvalue())

if __name__=='__main__':unittest.main()
