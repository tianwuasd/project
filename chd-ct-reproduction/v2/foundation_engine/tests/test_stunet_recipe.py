"""Scientific recipe and completed-model identity, independent of real data."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


class OfficialRecipeTests(unittest.TestCase):
    def test_saved_oof_summary_displays_case_and_fold_scores(self):
        from task_queries import result_lines
        from server_data import write_json
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            write_json(work/'09_reports/final_evaluation/summary.json',dict(results={'CT_raw':dict(
                methods={'S1':dict(case_mean_dice=.75,fold_equal_dice=.74,worst_fold_dice=.70,per_fold={'fold0':.70})})}))
            with patch('task_queries.context',return_value=({},work,{'status':'completed'})):
                text='\n'.join(result_lines(work,detailed=True))
            self.assertIn('0.75',text)
            self.assertIn('0.74',text)
            self.assertNotIn('最差中心',text)

    def test_smoke_summary_allows_pretrained_and_random_weights_to_differ(self):
        import summarize_smoke as summary
        from server_data import write_json,read_json
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            write_json(work/'00_admin/model_selection.json',dict(folds=['imagechd_fold0'],methods=['S1','S0']))
            entries=[]
            for method in ['S1','S0']:
                report=work/(method+'.json');log=work/(method+'.log')
                write_json(report,dict(status='passed',smoke_only=True,initial_weight_sha256=method,
                    plan_sha256='same-plan',train_seconds_excluding_first_two=[1.,1.],validation_step_seconds=.5,peak_allocated_gib=1.))
                log.write_text('REPORT: '+str(report)+'\n',encoding='utf-8')
                entries.append(dict(fold='imagechd_fold0',method=method,status='passed',log=str(log)))
            write_json(work/'09_reports/smoke_suite_synthetic/suite.json',entries)
            with patch.object(summary,'ROOT',work):summary.selected_summary()
            result=read_json(work/'09_reports/smoke_summary.json')
            self.assertAlmostEqual(result['rows'][0]['one_training_hours_extrapolated'],1000*(250+50*.5)/3600)

    def test_small_scratch_and_finetuning_follow_distinct_official_learning_rates(self):
        from stunet_recipe import recipe, budget, require_trainer_budget
        self.assertEqual(budget(),dict(epochs=1000,train_iterations=250,validation_iterations=50))
        self.assertEqual(recipe('S1')['initial_lr'],.001)
        self.assertEqual(recipe('S0')['initial_lr'],.01)
        self.assertEqual(recipe('S1')['weight_decay'],3e-5)
        with self.assertRaises(ValueError):recipe('B0')
        from types import SimpleNamespace
        with self.assertRaises(ValueError):
            require_trainer_budget(SimpleNamespace(num_epochs=250,num_iterations_per_epoch=250,num_val_iterations_per_epoch=50))

    def test_new_request_and_resource_metadata_agree_on_1000_without_changing_batch(self):
        from assessment_policy import automatic_request
        from training_policy import distribution
        from live_resources import training_policy
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'data').mkdir()
            request=automatic_request(dict(name='test',runtime=str(p/'runtime'),results=str(p/'results')),p/'data')
            self.assertEqual(request['global_batch_size'],4)
            self.assertEqual(request['formal_budget']['epochs'],1000)
            self.assertIn('official',request['training_recipe_id'])
        self.assertEqual(distribution(4,['GPU-0'])['formal_epochs'],1000)
        self.assertEqual(training_policy(dict(cpu_slots=8,available_memory_bytes=16*2**30),1,1024)['formal_epochs'],1000)

    def test_trainers_apply_official_hyperparameters_and_budget_before_initialize(self):
        import stunet_trainers as st
        labels={'background':0,'LV':1,'RV':2,'LA':3,'RA':4,'MYO':5,'AO':6,'PA':7,'ignore':8}
        with patch.object(st.BaselineNNUNet,'__init__',return_value=None):
            for cls,lr in [(st.STUNetPretrained,.001),(st.STUNetRandom,.01)]:
                trainer=cls({},'3d_fullres',0,dict(labels=labels))
                self.assertEqual((trainer.num_epochs,trainer.num_iterations_per_epoch,trainer.num_val_iterations_per_epoch),(1000,250,50))
                self.assertEqual((trainer.initial_lr,trainer.weight_decay),(lr,3e-5))

    def test_legacy_completed_250_is_not_an_official_1000_model(self):
        from stunet_recipe import require_completed_recipe,recipe
        old=dict(status='completed',formal_training=True,planned_epochs=250,completed_epochs=250,signature={})
        with self.assertRaisesRegex(ValueError,'配方|1000'):require_completed_recipe(old,'S1')
        new=dict(old,planned_epochs=1000,completed_epochs=1000,signature=dict(training_recipe=recipe('S1')))
        require_completed_recipe(new,'S1')

    def test_evaluation_accepts_formal_batch_signature_and_freezes_final(self):
        import evaluation as ev
        import server_train as train
        from server_data import write_json,read_json
        from stunet_recipe import recipe
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            fold='imagechd_fold0';method='S1'
            split=dict(dataset_id=921,train=['a'],val=['b'],target=['c'])
            write_json(work/'04_data/manifests/splits_v1.json',dict(folds={fold:split}))
            write_json(work/'04_data/derived/nnUNet_preprocessed/Dataset921_imagechd_fold0/WholeheartBaselinePlans.json',{})
            for name in ['pip_freeze_training.txt','conda_explicit_training.txt']:
                path=work/'06_configs'/name;path.parent.mkdir(exist_ok=True);path.write_text('synthetic environment')
            write_json(work/'00_admin/model_selection.json',dict(folds=[fold],methods=[method]))
            with patch.dict(os.environ,{'WHOLE_HEART_GLOBAL_BATCH':'4'}):
                sig=train.training_signature(work,fold,method)
            out=work/f'07_experiments/formal/{fold}_{method}_seed0'
            write_json(out/'run.json',dict(status='completed',formal_training=True,planned_epochs=1000,
                completed_epochs=1000,signature=sig,split=split,initial_weight_sha256='synthetic-pretrained'))
            for name in ['checkpoint_final.pth','checkpoint_best.pth']:
                path=out/'fold_0'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(name.encode())
            with patch.object(ev,'ROOT',work),patch('server_data.freeze_source_labels'):
                models,_=ev.freeze_models()
            self.assertTrue(models[0]['checkpoint'].endswith('checkpoint_final.pth'))
            self.assertEqual(models[0]['signature']['selected_global_batch_size'],4)
            frozen=read_json(work/'08_results/evaluation/frozen_models.json')
            self.assertEqual(frozen['protocol']['version'],2)
            self.assertEqual(frozen['protocol']['checkpoint'],'checkpoint_final.pth')


if __name__=='__main__':unittest.main()
