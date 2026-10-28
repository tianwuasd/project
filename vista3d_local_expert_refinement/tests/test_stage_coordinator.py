import os
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import cooperative_control as control
import batch_pool
import rank_affinity
from live_resources import training_policy

class StagePolicyTests(unittest.TestCase):
    def test_capacity_uses_all_rank_initialized_evidence(self):
        facts=lambda peak:dict(fixed_bytes=4,peak_allocated_bytes=peak,peak_reserved_bytes=19,optimizer_initialized=True)
        result=dict(local_batch_size=2,rank_results=[dict(gpu_uuid='a',memory_facts=facts(8)),dict(gpu_uuid='b',memory_facts=facts(12))])
        inferred=control.infer_batch(result,{'a':20,'b':20})
        self.assertEqual(inferred['local_batch_size'],4)
        self.assertEqual(len(inferred['ranks']),2)

    def test_changed_batch_recalculates_ram_and_all_ranks(self):
        snapshot=dict(cpu_budget=191,memory_budget_bytes=1000*2**30,cpu_ids=list(range(240)))
        one=control.stage_config(snapshot,[16,16,16],2,8)
        larger=control.stage_config(snapshot,[16,16,16],8,8)
        self.assertEqual(larger['batch_bytes'],one['batch_bytes']*4)
        self.assertLessEqual(larger['total_cpu_slots'],191)
        self.assertEqual(len(larger['rank_settings']),8)
        self.assertTrue(all(r['batch_bytes']==larger['batch_bytes'] for r in larger['rank_settings']))
        self.assertIn('coarse',larger['memory_components']['estimate_scope'])

    def test_unknown_evidence_keeps_proven_batch(self):
        self.assertEqual(control.infer_batch(dict(local_batch_size=2,rank_results=[]),{})['local_batch_size'],2)

    def test_cpu_failure_never_classified_as_gpu_oom(self):
        from ddp_runtime import classify_failure
        self.assertEqual(classify_failure(MemoryError('OOM'))['classification'],'cpu_memory')
        self.assertEqual(classify_failure(RuntimeError('CUDA out of memory'))['classification'],'noncapacity')

    def test_shm_live_capacity_checked_before_mapping(self):
        with patch.object(batch_pool,'shared_free_bytes',return_value=3),patch('multiprocessing.shared_memory.SharedMemory') as factory:
            with self.assertRaises(OSError):batch_pool.allocate_shared(4)
            factory.assert_not_called()

    def test_shm_enospc_closes_only_new_owned_segment(self):
        segment=Mock();segment._fd=42
        with patch.object(batch_pool,'shared_free_bytes',return_value=100),patch('multiprocessing.shared_memory.SharedMemory',return_value=segment),patch.object(batch_pool.sys,'platform','linux'),patch.object(os,'posix_fallocate',side_effect=OSError('ENOSPC'),create=True):
            with self.assertRaises(OSError):batch_pool.allocate_shared(4)
        segment.close.assert_called_once();segment.unlink.assert_called_once()

    def test_shm_unknown_linux_reservation_does_not_risk_sigbus(self):
        with patch.object(batch_pool,'shared_free_bytes',return_value=None),patch.object(batch_pool.sys,'platform','linux'),patch('multiprocessing.shared_memory.SharedMemory') as factory:
            with self.assertRaises(OSError):batch_pool.allocate_shared(4)
            factory.assert_not_called()

    def test_affinity_counts_threads_and_alternating_pools(self):
        row=dict(main_threads=4,train_workers=16,worker_threads=2,val_workers=4,val_worker_threads=2)
        self.assertEqual(rank_affinity.required_slots(row),36)


class CoordinatorBehaviorTests(unittest.TestCase):
    def workspace(self,root):
        from server_data import write_json
        write_json(root/'04_data/manifests/splits_v1.json',dict(folds={'f':dict(dataset_id=813)}))
        write_json(root/'04_data/derived/nnUNet_preprocessed/Dataset813_f/WholeheartBaselinePlans.json',dict(configurations={'3d_fullres':dict(batch_size=16,patch_size=[8,8,8])}))
        return root

    def test_capacity_failure_bounded_three_reductions_not_ranking(self):
        from ddp_runtime import RankGroupFailure
        errors=RankGroupFailure('actual gpu OOM',[dict(classification='gpu_oom')])
        with tempfile.TemporaryDirectory() as tmp:
            root=self.workspace(Path(tmp));calls=[]
            def attempt(w,f,m,g,b,r,k,**kw):calls.append(b);raise errors
            with patch.object(control,'admit',return_value=(['GPU-0'],{'GPU-0':20*2**30},[])),patch('task_resources.probe',return_value=dict(cpu_budget=20,memory_budget_bytes=100*2**30)),patch.object(control,'trial',side_effect=attempt),patch('training_policy.select_trial') as search:
                with self.assertRaises(RankGroupFailure):control.tune(root,'f','B0','assessment',['GPU-0'])
            self.assertEqual(calls,[16,8,4,2]);search.assert_not_called()

    def test_eager_recovery_preserves_proven_batch_above_plan_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.workspace(Path(tmp));batches=[]
            def attempt(w,f,m,g,b,r,k,**kw):
                batches.append(b)
                return dict(status='passed',local_batch_size=b)
            with patch.object(control,'admit',return_value=(['GPU-0'],{'GPU-0':20*2**30},[])),patch('task_resources.probe',return_value=dict(cpu_budget=20,memory_budget_bytes=100*2**30)),patch.object(control,'trial',side_effect=attempt):
                result=control.tune(root,'f','B0','formal',['GPU-0'],max_local_batch=32,execution=dict(compile=False))
            self.assertEqual(batches,[32]);self.assertEqual(result['local_batch_size'],32)

    def test_noncapacity_error_surfaces_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.workspace(Path(tmp))
            with patch.object(control,'admit',return_value=(['GPU-0'],{'GPU-0':20*2**30},[])),patch('task_resources.probe',return_value=dict(cpu_budget=20,memory_budget_bytes=100*2**30)),patch.object(control,'trial',side_effect=ValueError('bad data')) as attempt:
                with self.assertRaisesRegex(ValueError,'bad data'):control.tune(root,'f','B0','assessment',['GPU-0'])
            self.assertEqual(attempt.call_count,1)

    def test_admission_reserves_once_and_excludes_external_pid_and_visibility(self):
        import evaluation_gpu
        rows=[dict(uuid='a',index='0',free_mib=23000,total_mib=24576,busy=False,utilization=0),dict(uuid='b',index='1',free_mib=24000,total_mib=24576,busy=True,utilization=0),dict(uuid='c',index='2',free_mib=24000,total_mib=24576,busy=False,utilization=0)]
        with patch.object(evaluation_gpu,'snapshot',return_value=rows),patch.object(evaluation_gpu,'allowed_ids',return_value={'a','b'}),patch.dict(os.environ,{'WHOLE_HEART_GPU_MEMORY':'20'}):
            group,usable,excluded=control.admit(['a','b','c'])
        self.assertEqual(group,['a']);self.assertEqual(usable['a'],int((23000-4096)*2**20))
        self.assertEqual({r['gpu_uuid'] for r in excluded},{'b','c'})

    def test_all_rank_cpu_masks_bound_and_physical_core_first(self):
        resource=control.stage_config(dict(cpu_budget=191,memory_budget_bytes=1000*2**30,cpu_ids=list(range(240)),physical_cores=[dict(cpu=i,socket=0,core=i%120) for i in range(240)]),[8,8,8],2,8)
        pools={str(i):(i//4,set(range(i//4*120,(i//4+1)*120))) for i in range(8)}
        masks=rank_affinity.placement(list(pools),resource,set(range(240)),pools)
        self.assertEqual(len(set(c for row in masks for c in row)),sum(map(len,masks)))
        self.assertLessEqual(sum(map(len,masks)),191)
        self.assertTrue(all(set(row)<=set(range(240)) for row in masks))
        self.assertEqual(rank_affinity.physical_first({0,1,2,3},[dict(cpu=i,socket=0,core=i%2) for i in range(4)]),[0,1,2,3])

    def test_numa_binding_applies_all_owned_native_tids(self):
        resource=dict(rank_settings=[dict(main_threads=1,train_workers=1,worker_threads=2,val_workers=0)],original_cpu_affinity=[2,3,4],cpu_budget=3)
        with patch.object(os,'sched_getaffinity',return_value={2,3,4},create=True),patch.object(os,'sched_setaffinity',create=True) as setter,patch('evaluation_gpu.query',side_effect=OSError('no NUMA evidence')),patch.object(Path,'iterdir',return_value=[Path('10'),Path('11')]):
            result=rank_affinity.bind(0,['a'],resource)
        self.assertEqual(result['status'],'bound');self.assertEqual(result['native_tids_bound'],[10,11])
        self.assertEqual([c.args[0] for c in setter.call_args_list],[10,11])

    def test_slowest_rank_windows_require_both_duration_and_steps(self):
        from training_feedback import slowest_window
        rows=[dict(rank=0,steps=20,seconds=40,wait_fraction=.8),dict(rank=1,steps=20,seconds=50,wait_fraction=.4)]
        self.assertEqual(slowest_window(rows)['rank'],1)
        self.assertIsNone(slowest_window([dict(rows[0],seconds=29),rows[1]]))
        self.assertIsNone(slowest_window([dict(rows[0],steps=19),rows[1]]))

    def test_asymmetric_supply_wait_survives_nearly_equal_ddp_wall_times(self):
        from training_feedback import slowest_window
        from live_resources import feedback_policy
        resource=training_policy(dict(cpu_budget=160,memory_budget_bytes=100*2**30),2,2**20)
        self.assertEqual([r['train_workers'] for r in resource['rank_settings']],[16,16])
        self.assertEqual(resource['limits']['train_workers'],[38,38])
        rows=[dict(rank=0,steps=20,seconds=31,wait_fraction=.1),
              dict(rank=1,steps=20,seconds=30.9,wait_fraction=25/30.9)]
        window=slowest_window(rows)
        decision=feedback_policy(resource,[window,window],resource)
        self.assertEqual(decision['action'],'expand')
        self.assertEqual([r['train_workers'] for r in decision['settings']['rank_settings']],[24,24])
        self.assertEqual(window['rank'],0)
        self.assertEqual(window['slowest_wall_rank'],0)
        self.assertEqual(window['supply_wait_rank'],1)
        self.assertEqual(window['wait_fraction'],25/30.9)
        self.assertEqual(window['rank_windows'],rows)
        low=slowest_window([dict(row,wait_fraction=.1) for row in rows])
        self.assertEqual(feedback_policy(resource,[low,low],resource)['action'],'keep')
        for rank in range(2):
            for invalid in (dict(seconds=29.9),dict(steps=19)):
                with self.subTest(rank=rank,invalid=invalid):
                    incomplete=[dict(row,**invalid) if i==rank else row for i,row in enumerate(rows)]
                    self.assertIsNone(slowest_window(incomplete))

    def test_two_windows_apply_cpu_expansion_without_own_busy_probe(self):
        from training_feedback import RollingFeedback
        resource=control.stage_config(dict(cpu_budget=80,memory_budget_bytes=100*2**30),[8,8,8],2,1)
        class Dist:
            def all_gather_object(self,rows,local):rows[0]=local
            def broadcast_object_list(self,p,src):pass
        clock=[0.]
        trainer=Mock(current_epoch=0)
        with tempfile.TemporaryDirectory() as tmp,patch('training_feedback.apply_pools',return_value=dict(status='bound')) as apply,patch('task_resources.probe',side_effect=AssertionError('must not measure own load')):
            feedback=RollingFeedback(resource,0,1,tmp,clock=lambda:clock[0])
            for _ in range(40):
                clock[0]+=2
                feedback.observe(dict(batch_wait_seconds=1.5),trainer,Dist())
            apply.assert_called_once()
            self.assertGreater(feedback.resource['rank_settings'][0]['train_workers'],16)
            self.assertLessEqual(feedback.resource['total_cpu_slots'],80)
            self.assertEqual(feedback.events[-1]['decision']['applied_boundary'],'replay_safe_batch')

    def test_unsupported_source_defers_without_claiming_safe_pool_rebuild(self):
        from training_feedback import apply_pools
        class Trainer:pass
        trainer=Trainer();trainer.dataloader_train=object();trainer.dataloader_val=object()
        self.assertFalse(apply_pools(trainer,dict(rank_settings=[{}]),0))

    def test_assessment_writes_observation_without_runtime_receipt(self):
        import assessment_run
        from server_data import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime=root/'runtime'
            plan=dict(workspace=str(root),runtime=str(runtime),data='synthetic',formal_budget={},assessment_budget={},folds=['f'],methods=['B0'],gpu_memory=20)
            fleet=Mock(path=root/'leases');fleet.start.return_value=fleet
            chosen=dict(gpu_uuids=['GPU-0'],output='synthetic',effective_cpu={})
            def predict(*args,**kwargs):write_json(root/'09_reports/assessment_prediction_f_B0.json',dict(status='passed'))
            budget=dict(workers=1,threads_per_worker=1)
            with patch('assessment_prepare.prepare',return_value=dict(prediction_ids=['case'])),patch('runtime_observations.collect'),patch('gpu_fleet.GPUFleet',return_value=fleet),patch.object(control,'tune',return_value=chosen),patch.object(control,'identity',return_value={}),patch('task_worker.run_child',side_effect=predict),patch('task_resources.prediction_budget',return_value=budget) as gpu_budget,patch('task_resources.cpu_tail_budget',return_value=budget) as cpu_budget:
                report=assessment_run.run(plan)
            gpu_budget.assert_called_once_with(root,[dict(fold='f',case_id='case')],1)
            cpu_budget.assert_called_once_with(root,[dict(fold='f',method='B0',case_id='case')],assessment=True)
            self.assertEqual(report['function_status'],'passed');self.assertIsNone(report['resource_recommendation'])
            self.assertEqual(report['effect_status'],'insufficient_long_window_evidence')
            self.assertFalse((runtime/'optimization').exists());fleet.close.assert_called_once()

class ReplayLoader:
    replayable=True
    def __iter__(self):return self
    def __next__(self):
        import random
        import torch
        return {'data':torch.tensor([random.random()]),'target':[torch.zeros(1)]}

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional actual CPU Torch integration')
class ReplayReconfigurationTests(unittest.TestCase):
    def test_real_workers_reconfigure_preserves_model_consumed_cursor(self):
        import torch
        from batch_transfer import BatchTransfer
        from batch_pool import EpochBatchIterator
        reference=BatchTransfer(EpochBatchIterator(ReplayLoader(),workers=0,seed=11,phase_steps=6),torch.device('cpu'))
        source=EpochBatchIterator(ReplayLoader(),workers=1,seed=11,phase_steps=6,prefetch=3,max_shared_bytes=4096)
        actual=BatchTransfer(source,torch.device('cpu'))
        try:
            expected=next(reference)['data'].clone();reference.mark_consumed()
            self.assertTrue(torch.equal(next(actual)['data'],expected));actual.mark_consumed()
            abandoned=next(actual)['data'].clone()
            old=source.observations()['worker_pids']
            actual.reconfigure(workers=2,worker_threads=1,prefetch=4,max_shared_bytes=4096)
            self.assertEqual(actual.consumed,1);self.assertEqual(source.state_dict()['next_index'],1)
            self.assertTrue(torch.equal(next(actual)['data'],abandoned));actual.mark_consumed()
            new=source.observations()['worker_pids'];self.assertEqual(len(new),2);self.assertFalse(set(old)&set(new))
            for _ in range(4):
                reference.mark_consumed() if reference.active is not None else None
                expected=next(reference)['data'].clone();reference.mark_consumed()
                # reference first remaining value is the replayed abandoned batch
                if reference.consumed==2:continue
                self.assertTrue(torch.equal(next(actual)['data'],expected));actual.mark_consumed()
        finally:actual.close();reference.close()
        self.assertEqual(source.observations()['worker_pids'],[])


class GroupRecoveryTests(unittest.TestCase):
    def test_formal_whole_group_oom_and_compile_resume_only_complete_checkpoint(self):
        import sys
        from ddp_runtime import RankGroupFailure
        from test_integration_fixes import TopologyAndRegroupTests
        from server_data import read_json
        for category in ('gpu_oom','compile_retry'):
            with self.subTest(category=category),tempfile.TemporaryDirectory() as tmp:
                root=TopologyAndRegroupTests().workspace(tmp);launches=[];decisions=[]
                def tune(*args,**kwargs):
                    decisions.append(kwargs)
                    local=kwargs.get('max_local_batch',4)
                    return dict(gpu_uuids=['GPU-0','GPU-1'],world_size=2,local_batch_size=local,
                        global_batch_size=local*2,effective_cpu={},train_patches_per_epoch=local*500,
                        reference_train_patches_per_epoch=500,execution_request=kwargs.get('execution',dict(compile=True)))
                def launch(target,args,*other,**kwargs):
                    launches.append(args)
                    if len(launches)==1:raise RankGroupFailure('injected '+category,[dict(classification=category)])
                    return [dict(result=dict(status='completed',checkpoint='final',next_epoch=250,initial_weight_sha256='seed0'))]
                with patch.dict(sys.modules,{'server_train':Mock(select_checkpoint=Mock(return_value=Path('last-complete')))}),patch.object(control,'tune',side_effect=tune),patch.object(control,'admit',side_effect=lambda g:(g,{},[])),patch('ddp_runtime.launch_ranks',side_effect=launch):
                    control.train(root,'ct_holdG','B0')
                self.assertEqual(len(launches),2);self.assertIsNone(launches[0]['checkpoint'])
                self.assertEqual(launches[1]['checkpoint'],'last-complete')
                self.assertEqual(decisions[1]['max_local_batch'],2 if category=='gpu_oom' else 4)
                self.assertEqual(launches[1]['execution']['compile'],category!='compile_retry')
                self.assertEqual(read_json(root/'07_experiments/formal/ct_holdG_B0_seed0/run.json')['status'],'completed')

    def test_typed_oom_compile_and_unknown_cause_classification(self):
        import sys
        from ddp_runtime import classify_failure
        from training_execution import EagerRetryRequired
        class CudaOOM(RuntimeError):pass
        compiler=type('InductorError',(RuntimeError,),{'__module__':'torch._inductor.exc'})
        with patch.dict(sys.modules,{'torch':Mock(cuda=Mock(OutOfMemoryError=CudaOOM))}):
            for cause,expected in [(CudaOOM('capacity'),'gpu_oom'),(MemoryError('host'),'cpu_memory'),(compiler('compiler'),'compile_retry'),(ValueError('data'),'noncapacity'),(RuntimeError('NCCL'),'noncapacity')]:
                wrapper=EagerRetryRequired('retry requested');wrapper.__cause__=cause
                self.assertEqual(classify_failure(wrapper)['classification'],expected)

if __name__=='__main__':unittest.main()
