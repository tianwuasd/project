"""Post-trial sampling lag must not reject a safe owned GPU group."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import cooperative_control as control
import evaluation_gpu
from server_data import write_json


class Clock:
    def __init__(self):self.now=0
    def monotonic(self):return self.now
    def sleep(self,seconds):self.now+=seconds


class AdmissionTests(unittest.TestCase):
    def rows(self,util=0,busy=False,free=24000):
        return [dict(uuid='GPU-'+str(i),index=str(i),free_mib=free,total_mib=24576,
                     busy=busy,utilization=(32 if i==0 else util),process_pids=[999] if busy else [])
                for i in range(4)] if util else [dict(uuid='GPU-'+str(i),index=str(i),free_mib=free,
                    total_mib=24576,busy=busy,utilization=0,process_pids=[999] if busy else []) for i in range(4)]

    def settle(self,rows,wait=4):
        clock=Clock()
        with patch.object(evaluation_gpu,'snapshot',side_effect=rows) as snapshot, \
             patch.object(evaluation_gpu,'allowed_ids',return_value={'GPU-'+str(i) for i in range(4)}), \
             patch.object(control.time,'monotonic',side_effect=clock.monotonic), \
             patch.object(control.time,'sleep',side_effect=clock.sleep), \
             patch('runtime_observations.event') as event, \
             patch.dict(os.environ,{'WHOLE_HEART_GPU_MEMORY':'20'}),contextlib.redirect_stdout(io.StringIO()):
            result=control.settled_admit(['GPU-'+str(i) for i in range(4)],Path('.'),'after_capacity',max_wait_seconds=wait)
        return result,clock.now,snapshot.call_count,event.call_args_list

    def test_four_high_readings_after_trial_require_two_clean_samples(self):
        result,elapsed,samples,events=self.settle([self.rows(100),self.rows(),self.rows()])
        self.assertEqual(len(result[0]),4)
        self.assertFalse(result[2]);self.assertEqual(samples,3);self.assertEqual(elapsed,2)
        self.assertTrue(any(c.kwargs.get('context')=='after_capacity' for c in events))

    def test_external_pid_appearing_during_settle_is_never_ignored(self):
        result,_,samples,_=self.settle([self.rows(100),self.rows(busy=True),self.rows()])
        self.assertFalse(result[0]);self.assertEqual(samples,2)
        self.assertTrue(all(r['reason_code']=='external_process' for r in result[2]))

    def test_persistent_high_utilization_times_out_without_relaxing_threshold(self):
        result,elapsed,samples,_=self.settle([self.rows(100)]*5,wait=3)
        self.assertFalse(result[0]);self.assertEqual(elapsed,3);self.assertEqual(samples,4)
        self.assertTrue(all(r['reason_code']=='utilization' for r in result[2]))

    def test_delayed_memory_release_is_also_checked_without_forcing_capacity(self):
        result,_,samples,_=self.settle([self.rows(free=9000),self.rows(),self.rows()])
        self.assertEqual(len(result[0]),4);self.assertEqual(samples,3)

    def test_one_clean_sample_at_deadline_is_not_enough(self):
        result,elapsed,samples,_=self.settle([self.rows(100),self.rows(100),self.rows()],wait=2)
        self.assertFalse(result[0]);self.assertEqual(elapsed,2);self.assertEqual(samples,3)
        self.assertTrue(all(r['reason_code']=='stability_pending' for r in result[2]))

    def test_slow_query_after_deadline_does_not_accept_a_late_second_clean_sample(self):
        with patch.object(evaluation_gpu,'snapshot',side_effect=[self.rows(100),self.rows(),self.rows()]), \
             patch.object(evaluation_gpu,'allowed_ids',return_value={'GPU-'+str(i) for i in range(4)}), \
             patch.object(control.time,'monotonic',side_effect=[0,0,9,11]), \
             patch.object(control.time,'sleep'),patch('runtime_observations.event'), \
             contextlib.redirect_stdout(io.StringIO()):
            ready,_,excluded=control.settled_admit(['GPU-'+str(i) for i in range(4)],Path('.'),'after_capacity')
        self.assertFalse(ready);self.assertTrue(excluded)

    def test_two_clean_samples_completed_exactly_at_deadline_pass(self):
        with patch.object(evaluation_gpu,'snapshot',side_effect=[self.rows(100),self.rows(),self.rows()]), \
             patch.object(evaluation_gpu,'allowed_ids',return_value={'GPU-'+str(i) for i in range(4)}), \
             patch.object(control.time,'monotonic',side_effect=[0,0,9,10]), \
             patch.object(control.time,'sleep'),patch('runtime_observations.event'), \
             contextlib.redirect_stdout(io.StringIO()):
            ready,_,excluded=control.settled_admit(['GPU-'+str(i) for i in range(4)],Path('.'),'after_capacity')
        self.assertEqual(len(ready),4);self.assertFalse(excluded)

    def test_interruption_during_settle_propagates(self):
        with patch.object(control,'admit',return_value=([],{},[dict(gpu_uuid='a',reason='busy sample',reason_code='utilization')])), \
             patch.object(control.time,'sleep',side_effect=KeyboardInterrupt),patch('runtime_observations.event'), \
             contextlib.redirect_stdout(io.StringIO()),self.assertRaises(KeyboardInterrupt):
            control.settled_admit(['a'],Path('.'),'after_capacity')

    def test_monitor_failure_is_not_treated_as_idle_or_retried(self):
        with patch.object(control,'admit',side_effect=RuntimeError('nvidia-smi failed')) as probe, \
             self.assertRaisesRegex(RuntimeError,'nvidia-smi'):
            control.settled_admit(['a'],Path('.'),'after_capacity')
        self.assertEqual(probe.call_count,1)

    def test_all_real_busy_devices_are_released_then_wait_for_new_safe_group(self):
        bad=dict(gpu_uuid='a',reason='external PID',reason_code='external_process')
        with patch.object(control,'settled_admit',side_effect=[([],{},[bad]),(['b'],{'b':20*2**30},[])]), \
             patch.object(control,'release_devices') as release,patch.object(control,'devices',return_value=['b']) as wait, \
             patch('runtime_observations.event'),contextlib.redirect_stdout(io.StringIO()):
            group,_,excluded=control.ready_group(Path('.'),['a'])
        self.assertEqual(group,['b']);self.assertEqual(excluded,[bad]);wait.assert_called_once_with()
        release.assert_called_once_with(['a'],wait=True)

    def test_post_trial_external_race_rebuilds_group_instead_of_aborting(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'04_data/manifests/splits_v1.json',dict(folds={'f':dict(dataset_id=813)}))
            write_json(root/'04_data/derived/nnUNet_preprocessed/Dataset813_f/WholeheartBaselinePlans.json',
                       dict(configurations={'3d_fullres':dict(batch_size=2,patch_size=[8,8,8])}))
            decisions=iter([(['a','b'],dict(a=20*2**30,b=20*2**30),[]),
                            (['b'],dict(b=20*2**30),[dict(gpu_uuid='a',reason='external PID',reason_code='external_process')]),
                            (['b'],dict(b=20*2**30),[]),(['b'],dict(b=20*2**30),[])])
            calls=[]
            def trial(w,f,m,g,b,r,k,**kwargs):
                calls.append(list(g));return dict(local_batch_size=b,rank_results=[])
            with patch.object(control,'settled_admit',side_effect=lambda *a,**kw:next(decisions)), \
                 patch.object(control,'release_devices'),patch.object(control,'trial',side_effect=trial), \
                 patch('task_resources.probe',return_value=dict(cpu_budget=16,memory_budget_bytes=100*2**30)), \
                 contextlib.redirect_stdout(io.StringIO()):
                chosen=control.tune(root,'f','B0','formal',['a','b'])
            self.assertEqual(calls,[['a','b'],['b']]);self.assertEqual(chosen['gpu_uuids'],['b'])
            self.assertEqual(chosen['admission_restarts'],1)

    def test_real_external_card_does_not_release_other_cards_with_only_stale_utilization(self):
        external=dict(gpu_uuid='a',reason='external PID',reason_code='external_process')
        transient=dict(gpu_uuid='b',reason='sample lag',reason_code='utilization')
        with patch.object(control,'settled_admit',side_effect=[([],{},[external,transient]),(['b'],{'b':20*2**30},[])]), \
             patch.object(control,'release_devices') as release,patch('runtime_observations.event'):
            group,_,excluded=control.ready_group(Path('.'),['a','b'])
        self.assertEqual(group,['b']);self.assertEqual(excluded,[external])
        release.assert_called_once_with(['a'],wait=True)

    def test_capacity_admission_churn_restarts_are_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'04_data/manifests/splits_v1.json',dict(folds={'f':dict(dataset_id=813)}))
            write_json(root/'04_data/derived/nnUNet_preprocessed/Dataset813_f/WholeheartBaselinePlans.json',
                       dict(configurations={'3d_fullres':dict(batch_size=2,patch_size=[8,8,8])}))
            error=control.ResourceAdmissionError([dict(gpu_uuid='a',reason='external',reason_code='external_process')])
            with patch.object(control,'ready_group',return_value=(['a'],{'a':20*2**30},[])), \
                 patch.object(control,'trial',side_effect=error) as trial, \
                 patch('task_resources.probe',return_value=dict(cpu_budget=16,memory_budget_bytes=100*2**30)), \
                 self.assertRaises(control.ResourceAdmissionError):
                control.tune(root,'f','B0','formal',['a'])
            self.assertEqual(trial.call_count,4)

    def train_fixture(self,root):
        write_json(root/'04_data/manifests/splits_v1.json',dict(folds={'f':dict(dataset_id=813)}))
        write_json(root/'04_data/derived/nnUNet_preprocessed/Dataset813_f/WholeheartBaselinePlans.json',
                   dict(configurations={'3d_fullres':dict(batch_size=2,patch_size=[8,8,8])}))
        for name in ('pip_freeze_training.txt','conda_explicit_training.txt'):
            path=root/'06_configs'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'fixture')
        return dict(gpu_uuids=['a','b'],local_batch_size=2,global_batch_size=4,effective_cpu={},
                    train_patches_per_epoch=1000,reference_train_patches_per_epoch=500,execution_request={})

    def test_formal_launch_reselects_without_blanket_release_of_mixed_sample(self):
        hard=dict(gpu_uuid='a',reason='external PID',reason_code='external_process')
        soft=dict(gpu_uuid='b',reason='sample lag',reason_code='utilization')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);chosen=self.train_fixture(root)
            with patch.dict(sys.modules,{'server_train':Mock(select_checkpoint=Mock(side_effect=FileNotFoundError))}), \
                 patch.object(control,'tune',return_value=chosen) as tune, \
                 patch.object(control,'settled_admit',side_effect=[([],{},[hard,soft]),(['a','b'],{},[])]), \
                 patch.object(control,'release_devices') as release, \
                 patch('ddp_runtime.launch_ranks',return_value=[dict(result=dict(status='completed',
                    checkpoint='final',next_epoch=250,initial_weight_sha256='seed0'))]) as launch, \
                 contextlib.redirect_stdout(io.StringIO()):control.train(root,'f','B0')
            self.assertEqual(tune.call_count,2);launch.assert_called_once();release.assert_not_called()

    def test_formal_launch_admission_churn_is_bounded(self):
        bad=dict(gpu_uuid='a',reason='external PID',reason_code='external_process')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);chosen=self.train_fixture(root)
            with patch.dict(sys.modules,{'server_train':Mock(select_checkpoint=Mock(side_effect=FileNotFoundError))}), \
                 patch.object(control,'tune',return_value=chosen) as tune, \
                 patch.object(control,'settled_admit',return_value=([],{},[bad])), \
                 patch('ddp_runtime.launch_ranks') as launch,contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaises(control.ResourceAdmissionError):control.train(root,'f','B0')
            self.assertEqual(tune.call_count,4);launch.assert_not_called()


if __name__=='__main__':unittest.main()
