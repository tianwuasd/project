"""20 GiB defaults and total-aware idle thresholds, no real GPUs."""
import os
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_panel as panel
import gpu_devices
import evaluation_gpu
from gpu_budget import MEMORY_POLICY, required_free_mib


class GPUMemoryTests(unittest.TestCase):
    def config(self, root, tasks='prepare', source=None, extra=(), saved=None):
        raw=root/'raw';raw.mkdir(exist_ok=True)
        argv=['--profile','zmic44','--runtime',str(root/'runtime'),'--results',str(root/'results'),
              '--data',str(raw),'--tasks',tasks,'--name','budget test',*extra]
        if source:
            old=root/'old';old.mkdir(exist_ok=True)
            (old/'plan.json').write_text(json.dumps(dict(data=str(raw),gpu_memory=5)),encoding='utf-8')
            argv+=['--input-job',str(old)]
        with patch.object(panel,'resolve_gpus',return_value=['GPU-zero']):
            return panel.new_config(panel.make_parser().parse_args(argv),saved or {},interactive=False)

    def test_zmic_new_task_uses_20_instead_of_legacy_5_preference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            args=panel.make_parser().parse_args(['--profile','zmic44','--runtime',str(root/'runtime'),'--results',str(root/'results'),'--data',str(raw),'--tasks','prepare','--name','20 GiB'])
            config=panel.new_config(args,{'gpu_memory':5},interactive=False)
            self.assertEqual(config['gpu_memory'],20)

    def test_explicit_budget_and_current_policy_preferences_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            saved=dict(gpu_memory=16,gpu_memory_policy=MEMORY_POLICY)
            self.assertEqual(self.config(root,saved=saved)['gpu_memory'],16)
            self.assertEqual(self.config(root,saved=saved,extra=['--gpu-memory','12'])['gpu_memory'],12)

    def test_changed_source_budget_creates_new_cache_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=self.config(root,source=True)
            self.assertEqual(result['gpu_memory'],20)
            self.assertEqual(result['preprocess_cache'],'new')
            self.assertEqual(json.loads((root/'old/plan.json').read_text(encoding='utf-8'))['gpu_memory'],5)

    def test_changed_source_budget_requires_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'加入1预处理'):
                self.config(Path(tmp),tasks='smoke,train',source=True)

    def test_prediction_only_keeps_source_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.config(Path(tmp),tasks='predict',source=True)
            self.assertEqual(result['gpu_memory'],5)

    def test_20gb_admission_threshold_and_busy_card_rejection(self):
        settings=dict(min_free_mib=6000,max_utilization_percent=10,planner_gpu_memory_gib=20)
        row=dict(index='0',uuid='GPU-zero',free_mib=20992,total_mib=24576,utilization=0,busy=False)
        self.assertEqual(required_free_mib(20,24576),20992)
        self.assertIsNone(evaluation_gpu.idle_reason(row,settings))
        self.assertIn('不足',evaluation_gpu.idle_reason(dict(row,free_mib=20991),settings))
        self.assertIn('计算进程',evaluation_gpu.idle_reason(dict(row,busy=True),settings))
        self.assertIn('利用率',evaluation_gpu.idle_reason(dict(row,utilization=11),settings))
        with self.assertRaisesRegex(ValueError,'超过'):
            evaluation_gpu.idle_reason(dict(row,total_mib=16384),settings)

    def test_20gb_training_selection_rejects_insufficient_free_memory(self):
        for free,passes in [(20992,True),(20991,False)]:
            rows=f'0, GPU-zero, RTX3090, {free}, 0, 24576\n'
            with self.subTest(free=free),patch.dict(os.environ,{'WHOLE_HEART_MANAGED':'1','WHOLE_HEART_GPU_MEMORY':'20'},clear=True),patch.object(gpu_devices.subprocess,'check_output',side_effect=[rows,'']):
                if passes:self.assertEqual(gpu_devices.select_gpus(1,'0'),['GPU-zero'])
                else:
                    with self.assertRaises(gpu_devices.GPUBusy):gpu_devices.select_gpus(1,'0')

    def test_gpu_form_shows_total_and_free_separately(self):
        def query(fields,kind='gpu'):
            return [] if kind=='compute-apps' else [['0','GPU-zero','RTX3090','23000','0','24576']]
        with patch.object(evaluation_gpu,'query',side_effect=query),patch.dict(os.environ,{},clear=True),contextlib.redirect_stdout(io.StringIO()) as output:
            panel.show_gpu_choices()
        text=output.getvalue()
        self.assertIn('总显存MiB / 空闲显存MiB',text)
        self.assertIn('24576 / 23000',text)

    def test_24gb_idle_card_is_not_required_to_have_24_point_5gb_free(self):
        rows='0, GPU-zero, RTX3090, 24000, 0, 24576\n'
        with patch.dict(os.environ,{'WHOLE_HEART_MANAGED':'1','WHOLE_HEART_GPU_MEMORY':'24'},clear=True),patch.object(gpu_devices.subprocess,'check_output',side_effect=[rows,'']):
            self.assertEqual(gpu_devices.select_gpus(1,'0'),['GPU-zero'])

    def test_evaluation_24gb_budget_uses_actual_total(self):
        row=dict(index='0',uuid='GPU-zero',free_mib=24000,total_mib=24576,utilization=0,busy=False)
        with patch.dict(os.environ,{'WHOLE_HEART_MANAGED':'1','WHOLE_HEART_GPU_MEMORY':'24','WHOLE_HEART_GPU_UUIDS':'["GPU-zero"]','WHOLE_HEART_GPU_LIMIT':'1'},clear=True):
            settings=evaluation_gpu.load_settings()
            self.assertIsNone(evaluation_gpu.idle_reason(row,settings))
