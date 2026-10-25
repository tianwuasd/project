"""Subset training dispatch and metric aggregation; never execute a neural network."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from server_data import read_json,write_json,digest
from model_catalog import save_selection
import server_launch as launcher
import evaluation as ev
from evaluation_summary import summarize


class SelectionPipelineTests(unittest.TestCase):
    def test_only_b1_is_dispatched_in_selected_fold(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp); raw=root/'raw';raw.mkdir();work=root/'work'
            stack.enter_context(patch.object(sys,'argv',['server_launch.py',str(raw),'--work-dir',str(work),
                '--mode','train','--models','B1','--folds','ct_holdG','--yes-train','--managed']))
            stack.enter_context(patch('server_data.bind_workspace'))
            stack.enter_context(patch.dict(launcher.os.environ, {'WHOLE_HEART_GPU_UUIDS':'["GPU-1"]'},clear=True))
            child=stack.enter_context(patch.object(launcher,'child'))
            queue=stack.enter_context(patch('gpu_queue.run_training_queue'))
            launcher.main()
            self.assertEqual(queue.call_args.args[0],[('ct_holdG','B1')])
            self.assertEqual(read_json(work/'00_admin/last_queue.json')['methods'],['B1'])
            self.assertEqual(child.call_count,1)

    def test_cpu_prepare_does_not_check_or_allocate_gpu(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);raw=root/'raw';raw.mkdir();work=root/'work'
            stack.enter_context(patch.object(sys,'argv',['server_launch.py',str(raw),'--work-dir',str(work),
                '--mode','prepare','--models','B0,B1','--folds','ct_holdG','--managed']))
            stack.enter_context(patch('server_data.bind_workspace'))
            stack.enter_context(patch.dict(launcher.os.environ, {'WHOLE_HEART_GPU_UUIDS':'[]'},clear=True))
            gpu=stack.enter_context(patch.object(launcher,'runtime_check'))
            selected=stack.enter_context(patch.object(launcher,'check_selected_gpus'))
            child=stack.enter_context(patch.object(launcher,'child'))
            launcher.main()
            gpu.assert_not_called();selected.assert_not_called()
            child.assert_called_once_with('server_prepare.py','ct_holdG','--gpu-memory',5)

    def test_single_b1_freezes_and_summarizes_without_b0_or_mri(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'05_src').mkdir()
            save_selection(root,['ct_holdG'],['B1'])
            split=dict(dataset_id=703,target=['Case0001'])
            write_json(root/'04_data/manifests/splits_v1.json',{'folds':{'ct_holdG':split}})
            for name in ['pip_freeze_training.txt','conda_explicit_training.txt']:
                write_json(root/'06_configs'/name,{})
            prep=root/'04_data/derived/nnUNet_preprocessed/Dataset703_ct_holdG'
            write_json(prep/'Project1Plans.json',{})
            out=root/'07_experiments/formal/ct_holdG_B1_seed0'
            sig=dict(fold='ct_holdG',method='B1',seed=0,plan_sha256=digest(prep/'Project1Plans.json'),
                splits_sha256=digest(root/'04_data/manifests/splits_v1.json'),
                environment_sha256=digest(root/'06_configs/pip_freeze_training.txt'),
                conda_environment_sha256=digest(root/'06_configs/conda_explicit_training.txt'),code_sha256={})
            write_json(out/'run.json',dict(status='completed',formal_training=True,signature=sig,split=split,initial_weight_sha256='same'))
            for name in ['checkpoint_best.pth','checkpoint_final.pth']:write_json(out/'fold_0'/name,{})
            with patch.multiple(ev,ROOT=root,CODE_ROOT=root):models,sha=ev.freeze_models()
            self.assertEqual(len(models),1)
            output=root/'prediction.bin';output.write_bytes(b'synthetic')
            score=dict(macro_dice=.75,both_empty_classes=0,one_empty_classes=0,defined_classes=1,
                       classes={'Myo':dict(dice=.75,hd_mm=1.,hd95_mm=.5,status='defined')})
            write_json(root/'08_results/evaluation/ct_holdG_B1/Case0001.json',dict(status='completed',
                signature={'frozen_sha256':sha},modality='CT',outputs={'raw':{'path':str(output),'sha256':digest(output)}},results={'raw':score,'lcc':score}))
            summarize(root)
            result=read_json(root/'09_reports/final_evaluation/summary.json')['results']
            self.assertEqual(set(result),{'CT_raw','CT_lcc'})
            self.assertIsNone(result['CT_raw']['paired'])
            self.assertEqual(set(result['CT_raw']['methods']),{'B1'})


if __name__ == '__main__':unittest.main()
