"""Automatic decisions, ownership and measurement failure boundaries; CPU only."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from server_data import write_json,read_json
import cooperative_control as control
import gpu_fleet
import evaluation_gpu
import rank_affinity


class AutomaticExecutionTests(unittest.TestCase):
    def setUp(self):
        helper=patch.object(control,'admit',side_effect=lambda g:(list(g),{u:20*2**30 for u in g},[]))
        helper.start();self.addCleanup(helper.stop)

    def workspace(self,tmp):
        root=Path(tmp)
        write_json(root/'04_data/manifests/splits_v1.json',dict(folds={'imagechd_fold0':dict(dataset_id=813)}))
        write_json(root/'04_data/derived/nnUNet_preprocessed/Dataset813_imagechd_fold0/WholeheartBaselinePlans.json',
            dict(configurations={'3d_fullres':dict(batch_size=2,patch_size=[160,192,192])}))
        return root

    def measurement(self,gpus,local,resource,seconds=4):
        return dict(status='passed',world_size=len(gpus),gpu_uuids=gpus,global_batch_size=resource['global_batch_size'],
            local_batch_size=local,measured_training_steps=10,median_step_seconds=seconds,
            validation_step_seconds=.3,peak_allocated_gib=12,peak_reserved_gib=13,
            effective_cpu=resource,total_cpu_slots=resource['total_cpu_slots'],rank_results=[dict(gpu_uuid=g,
                local_batch_size=resource['local_batch_sizes'][i],
                supply_transfer=dict(train=dict(source=dict(peak_payload_bytes=37074272*resource['local_batch_sizes'][i],workers=[]))),
            memory_facts=dict(fixed_bytes=4*2**30,peak_allocated_bytes=(4+4*resource['local_batch_sizes'][i])*2**30,peak_reserved_bytes=19*2**30,optimizer_initialized=True)) for i,g in enumerate(gpus)])

    def run_tuning(self,root,fail=False,world=8):
        called=[]
        def measure(work,fold,method,gpus,local,resource,kind,**kwargs):
            called.append((gpus,local,resource,kind))
            if len(called)==2 and fail:
                from ddp_runtime import RankGroupFailure
                raise RankGroupFailure('simulated group OOM',[dict(classification='gpu_oom')])
            return self.measurement(gpus,local,resource,4 if len(called)==1 else 3)
        snapshot=dict(cpu_slots=128,available_memory_bytes=200*2**30,shared_memory_available_bytes=100*2**30)
        with patch('task_resources.probe',return_value=snapshot),patch.object(control,'trial',side_effect=measure),\
             patch.object(control,'topology',side_effect=lambda g:[['same GPU','24576','driver'] for _ in g]),\
             patch.object(control,'release_devices'),contextlib.redirect_stdout(io.StringIO()):
            chosen=control.tune(root,'imagechd_fold0','S1','assessment',[f'GPU-{i}' for i in range(world)])
        return chosen,called

    def test_bounded_real_measurement_policy_allows_eight_and_records_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.workspace(tmp);chosen,called=self.run_tuning(root)
            self.assertEqual(len(called),2)
            self.assertEqual(len(called[1][0]),8)
            self.assertEqual(chosen['world_size'],8)
            self.assertGreater(chosen['sampling_ratio'],1)
            self.assertEqual(chosen['train_patches_per_epoch'],chosen['global_batch_size']*250)
            self.assertEqual(chosen['formal_budget'],dict(epochs=1000,train_iterations=250,validation_iterations=50))
            self.assertEqual(chosen['gpu_memory_reservation']['minimum_gib'],4)
            self.assertEqual(chosen['gpu_memory_reservation']['fraction_of_total'],.10)
            self.assertEqual(set(chosen['gpu_memory_reservation']['usable_bytes']),set(chosen['gpu_uuids']))
            self.assertLessEqual(chosen['effective_cpu']['total_cpu_slots'],128)

    def test_gpu_oom_reduces_batch_and_retains_all_compatible_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.workspace(tmp);chosen,called=self.run_tuning(root,True,world=4)
            self.assertEqual(chosen['world_size'],4)
            self.assertEqual(chosen['local_batch_size'],1)
            self.assertEqual(len(called),3)
            self.assertEqual(chosen['capacity_attempts'][1]['classification'],'gpu_oom')
            self.assertEqual(chosen['capacity_reductions'],1)

    def test_interruption_is_not_a_successful_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.workspace(tmp)
            with patch('task_resources.probe',return_value=dict(cpu_slots=8)),\
                 patch.object(control,'trial',side_effect=KeyboardInterrupt),\
                 patch.object(control,'release_devices'),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(KeyboardInterrupt):
                control.tune(root,'imagechd_fold0','S1','assessment',['GPU-0'])

    def test_only_registered_live_keeper_with_live_owner_is_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'diagnostics/gpu-leases.json'
            good=dict(gpu_uuid='GPU-0',status='ready',pid=321,identity='signed',owner=dict(pid=100,identity='owner'))
            rogue=dict(good,gpu_uuid='GPU-1',pid=322)
            write_json(path,dict(keepers=[good,rogue]))
            write_json(root/'processes/321.json',dict(pid=321,identity='signed',role='gpu-keeper'))
            with patch.object(gpu_fleet,'alive',return_value=True):
                self.assertEqual(list(gpu_fleet.verified_keepers(path)),['GPU-0'])
            with patch.object(gpu_fleet,'alive',return_value=False):
                self.assertEqual(gpu_fleet.verified_keepers(path),{})

    def test_external_gpu_process_is_never_hidden_by_keeper(self):
        env=dict(VISTA_GPU_LEASES='unit-test')
        def query(fields,kind='gpu'):
            if kind=='compute-apps':return [['GPU-0','321'],['GPU-0','999']]
            return [['0','GPU-0','24000','0','24576']]
        with patch.dict(os.environ,env),patch.object(evaluation_gpu,'query',side_effect=query),\
             patch.object(gpu_fleet,'verified_keepers',return_value={'GPU-0':dict(pid=321)}):
            row=evaluation_gpu.snapshot()[0]
        self.assertTrue(row['busy']);self.assertEqual(row['process_pids'],[321,999])

    def test_numa_missing_evidence_keeps_inherited_allocation(self):
        self.assertEqual(rank_affinity.parse_cpulist('0-2,5,7-8'),{0,1,2,5,7,8})
        with patch.object(os,'sched_getaffinity',return_value={2,3},create=True),\
             patch.object(os,'sched_setaffinity',create=True) as setter,\
             patch.object(evaluation_gpu,'query',side_effect=FileNotFoundError('no live NUMA')):
            result=rank_affinity.bind(0,['GPU-0'],dict(threads=1))
        self.assertEqual(result['status'],'unavailable');setter.assert_not_called()

    def test_cpu_tail_waits_for_verified_process_exit_not_a_missing_metadata_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gpu-leases.json';holder=dict(pid=321,identity='signed')
            with patch.dict(os.environ,{'VISTA_GPU_LEASES':str(path)}),\
                 patch.object(gpu_fleet,'verified_keepers',return_value={'GPU-0':holder}) as probe,\
                 patch('task_runtime.alive',side_effect=[True,False]) as living,\
                 patch.object(control.time,'sleep'):
                control.release_devices(['GPU-0'],wait=True)
            probe.assert_called_once();self.assertEqual(living.call_count,2)
            self.assertEqual(read_json(path.with_name('gpu-release-request.json'))['gpu_uuids'],['GPU-0'])

    def test_unconfirmed_keeper_exit_times_out_before_cpu_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gpu-leases.json'
            with patch.dict(os.environ,{'VISTA_GPU_LEASES':str(path)}),\
                 patch.object(gpu_fleet,'verified_keepers',return_value={'GPU-0':dict(pid=321,identity='signed')}),\
                 patch('task_runtime.alive',return_value=True),patch.object(control.time,'monotonic',side_effect=[0,31]),\
                 self.assertRaisesRegex(TimeoutError,'未开始CPU收尾'):
                control.release_devices(['GPU-0'],wait=True)


if __name__=='__main__':unittest.main()
