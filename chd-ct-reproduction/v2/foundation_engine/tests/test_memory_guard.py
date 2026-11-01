"""Live headroom decisions, independent of CT/CUDA and host resource sizes."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
GIB=2**30


class LiveMemoryTests(unittest.TestCase):
    def test_previously_finite_cgroup_limit_disappearance_blocks_growth(self):
        from memory_guard import growth_admitted
        self.assertFalse(growth_admitted(1,dict(reserve_bytes=4*GIB,known_limits={'/cg':64*GIB}),
            dict(available_bytes=100*GIB,cgroups=[dict(path='/cg',limit_bytes=None)])))

    def test_private_group_metadata_and_recycled_pid_are_checked(self):
        import ddp_runtime
        from types import SimpleNamespace
        from unittest.mock import Mock
        from server_data import write_json
        owner=dict(start='old',boot='boot')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'invocation';record=root/'rank-0/process-group.json'
            write_json(record,dict(pid=101,pgid=101,sid=999,identity=owner,invocation_id='invocation'))
            kill=Mock()
            with patch.object(ddp_runtime,'os',SimpleNamespace(name='posix',killpg=kill)):
                self.assertEqual(ddp_runtime._owned_groups(root,{101:owner}),[])
                write_json(record,dict(pid=101,pgid=101,sid=101,identity=owner,invocation_id='invocation'))
                self.assertEqual(ddp_runtime._owned_groups(root,{101:owner}),[101])
                with patch('task_runtime.identity',return_value=dict(start='reused',boot='boot')),self.assertRaises(RuntimeError):
                    ddp_runtime._signal_group(101,15,{101:owner})
                kill.assert_not_called()

    def test_owner_birth_identity_can_be_captured_from_unreaped_zombie(self):
        import ddp_runtime
        from types import SimpleNamespace
        stat='123 (python) '+' '.join(['Z']+['0']*18+['321'])
        with patch.object(ddp_runtime,'os',SimpleNamespace(name='posix')),patch('task_runtime.identity',return_value=None),\
             patch.object(Path,'read_text',side_effect=[stat,'boot-id']):
            self.assertEqual(ddp_runtime._ownership_identity(123),dict(boot='boot-id',start='321'))
    def test_owned_shared_memory_is_reclaimed_after_group_exit(self):
        from memory_guard import cleanup_owned_segments
        from multiprocessing.shared_memory import SharedMemory
        from server_data import write_json
        from uuid import uuid4
        with tempfile.TemporaryDirectory() as tmp:
            invocation=Path(tmp)/uuid4().hex;name='vista_'+invocation.name+'_'+uuid4().hex
            owner=dict(pid=101,identity=dict(start='owner'))
            segment=SharedMemory(name=name,create=True,size=64)
            try:
                write_json(invocation/'shared-memory'/f'{name}.json',dict(name=name,size=64,rank_owner=owner))
                self.assertEqual(cleanup_owned_segments(invocation,{101:owner['identity']}),[name])
            finally:
                segment.close()
                try:segment.unlink()
                except FileNotFoundError:pass

    def test_gpu_discovery_stays_healthy_when_selected_cap_is_full(self):
        from gpu_fleet import GPUFleet
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            fleet=GPUFleet(tmp,Path(tmp)/'runtime',max_training_gpus=4)
            fleet.stop=Mock();fleet.stop.wait.side_effect=[False,False,False,True]
            rows=[dict(uuid=f'g{i}',total_mib=24576) for i in range(8)]
            def acquire(uuid):fleet.processes[uuid]=(Mock(poll=Mock(return_value=None)),Mock(),Mock())
            with patch('evaluation_gpu.snapshot',return_value=rows),patch('evaluation_gpu.allowed_ids',return_value={r['uuid'] for r in rows}),\
                 patch('evaluation_gpu.idle_reason',return_value=None),patch.object(fleet,'_publish'),patch.object(fleet,'_acquire',side_effect=acquire),\
                 patch('runtime_observations.event'),patch('task_resources.probe',return_value={}):
                fleet._watch()
            self.assertIsNone(fleet.error)
            self.assertEqual(len(fleet.processes),4)
    def test_supervisor_stops_before_start_on_low_memory(self):
        from ddp_runtime import launch_ranks
        from memory_guard import HostMemoryPressureError
        from types import SimpleNamespace
        def low():raise HostMemoryPressureError(dict(reason='injected low headroom'))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(HostMemoryPressureError):
            launch_ranks('test_memory_guard:unused',{},['CPU-0'],Path(tmp)/'ranks',backend='gloo',memory_guard=SimpleNamespace(check=low))

    def test_cleanup_only_matches_current_invocation_rank_and_live_identity(self):
        from ddp_runtime import _registered_workers
        from server_data import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);owner=dict(pid=101,identity=dict(start='original'))
            base=dict(pid=201,identity=dict(start='worker'),role='batch:train',ddp_invocation='current',rank_owner=owner)
            write_json(root/'processes/201.json',base)
            write_json(root/'processes/202.json',dict(base,pid=202,ddp_invocation='other'))
            write_json(root/'processes/203.json',dict(base,pid=203,rank_owner=dict(pid=101,identity=dict(start='reused'))))
            with patch.dict(os.environ,{'WHOLE_HEART_JOB':tmp}),patch('task_runtime.alive',return_value=True):
                self.assertEqual([r['pid'] for r in _registered_workers('current',{101:owner['identity']})],[201])
    def test_cgroup_remaining_and_events_bound_host_available(self):
        from memory_guard import memory_snapshot
        files={'/proc/meminfo':'MemTotal: 262144000 kB\nMemAvailable: 104857600 kB\n',
               '/cg/memory.max':str(64*GIB),'/cg/memory.current':str(60*GIB),
               '/cg/memory.events':'oom 2\noom_kill 1\n'}
        result=memory_snapshot(read_text=lambda p:files.get(str(p)),groups=[('/cg',2)])
        self.assertEqual(result['available_bytes'],4*GIB)
        self.assertEqual(result['host_available_bytes'],100*GIB)
        self.assertEqual(result['cgroups'][0]['events']['oom_kill'],1)

    def test_unknown_finite_cgroup_usage_does_not_authorize_growth(self):
        from memory_guard import memory_snapshot,growth_admitted
        files={'/proc/meminfo':'MemTotal: 262144000 kB\nMemAvailable: 104857600 kB\n','/cg/memory.max':str(64*GIB)}
        result=memory_snapshot(read_text=lambda p:files.get(str(p)),groups=[('/cg',2)])
        self.assertIsNone(result['available_bytes'])
        self.assertFalse(growth_admitted(1,dict(reserve_bytes=4*GIB),result))

    def test_live_pressure_raises_structured_error_before_another_allocation(self):
        from memory_guard import LiveMemoryGuard,HostMemoryPressureError
        samples=iter([dict(available_bytes=100*GIB,cgroups=[]),dict(available_bytes=10*GIB,cgroups=[])])
        with tempfile.TemporaryDirectory() as tmp:
            guard=LiveMemoryGuard(tmp,dict(allocation_snapshot=dict(available_memory_bytes=100*GIB)),reader=lambda:next(samples))
            with self.assertRaises(HostMemoryPressureError) as caught:guard.check(force=True)
            self.assertEqual(caught.exception.classification,'cpu_memory_pressure')
            self.assertEqual(caught.exception.evidence['reserve_bytes'],20*GIB)
            self.assertTrue((Path(tmp)/'diagnostics/memory-guard-latest.json').exists())

    def test_expansion_needs_fresh_remaining_space_beyond_reserve(self):
        from memory_guard import growth_admitted
        policy=dict(reserve_bytes=20*GIB)
        self.assertFalse(growth_admitted(3*GIB,policy,dict(available_bytes=22*GIB)))
        self.assertTrue(growth_admitted(2*GIB,policy,dict(available_bytes=22*GIB)))
        self.assertFalse(growth_admitted(0,policy,dict(available_bytes=19*GIB)))

    def test_feedback_does_not_expand_into_low_live_headroom(self):
        from live_resources import training_policy,feedback_policy
        base=training_policy(dict(cpu_slots=128,available_memory_bytes=100*GIB),1,16*2**20)
        base['host_memory_guard']=dict(reserve_bytes=20*GIB)
        cap=dict(base,live_memory=dict(available_bytes=19*GIB))
        windows=[dict(steps=20,seconds=30,wait_fraction=.9,io_pressure=dict(avg10=0))]*2
        result=feedback_policy(base,windows,cap)
        self.assertEqual(result['settings']['rank_settings'][0]['train_workers'],base['rank_settings'][0]['train_workers'])
        self.assertNotEqual(result['action'],'expand')


class SelectedBatchTests(unittest.TestCase):
    def test_formal_host_pressure_recovers_complete_checkpoint_with_same_batch(self):
        import cooperative_control as control
        import sys
        from unittest.mock import Mock
        from test_integration_fixes import TopologyAndRegroupTests
        from memory_guard import HostMemoryPressureError
        decisions=[];calls=[]
        def tune(*args,**kwargs):
            decisions.append(kwargs)
            return dict(gpu_uuids=['g0','g1'],global_batch_size=4,selected_global_batch_size=4,max_global_batch_size=4,
                local_batch_size=2,world_size=2,local_batch_sizes=[2,2],train_patches_per_epoch=1000,
                reference_train_patches_per_epoch=500,effective_cpu=dict(memory_budget_bytes=30*GIB,estimated_ram_bytes=20*GIB),
                host_budget_bytes=kwargs.get('host_budget_bytes'),execution_request=dict(compile=False))
        def launch(target,args,*other,**kwargs):
            calls.append(args)
            if len(calls)==1:raise HostMemoryPressureError(dict(reason='injected low RAM'))
            return [dict(result=dict(status='completed',next_epoch=250,checkpoint='final',initial_weight_sha256='seed0'))]
        parent=Mock();parent.select_checkpoint.return_value=Path('last_complete.pth')
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'WHOLE_HEART_GLOBAL_BATCH':'4'}),\
             patch.dict(sys.modules,{'server_train':parent}),patch.object(control,'tune',side_effect=tune),\
             patch('ddp_runtime.launch_ranks',side_effect=launch),patch.object(control,'admit',side_effect=lambda g:(g,{},[])):
            control.train(TopologyAndRegroupTests().workspace(tmp),'imagechd_fold0','S1')
        self.assertEqual(calls[1]['checkpoint'],'last_complete.pth')
        self.assertEqual([r['global_batch_size'] for r in calls],[4,4])
        self.assertEqual(decisions[1]['host_budget_bytes'],12*GIB)
        self.assertEqual(decisions[1]['host_reduction_budget'],2)
    def test_request_and_cli_default_and_explicit_batch(self):
        from assessment_policy import automatic_request
        from task_panel import make_parser,stage_config
        from task_runtime import validate_request
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            profile=dict(name='zmic44',data=str(raw),runtime=str(root/'runtime'),results=str(root/'results'),gpu_memory=20)
            config=automatic_request(profile,raw,kind='assessment')
            self.assertEqual(config['global_batch_size'],4)
            self.assertFalse(config['allow_batch_growth'])
            with self.assertRaises(ValueError):validate_request(dict(config,global_batch_size=0))
            args=make_parser().parse_args(['--data',str(raw),'--global-batch','3'])
            with patch('server_profile.load',return_value=profile):chosen=stage_config(args,{},'assessment',interactive=False)
            self.assertEqual(chosen['global_batch_size'],3)
            self.assertEqual(chosen['max_global_batch_size'],3)

    def test_fixed_tuning_keeps_four_and_caps_nonempty_ranks(self):
        from test_automatic_execution import AutomaticExecutionTests
        import cooperative_control as control
        helper=AutomaticExecutionTests()
        for world in (1,2,3,4,6):
            with self.subTest(world=world),tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'WHOLE_HEART_GLOBAL_BATCH':'4'}),patch.object(control,'admit',side_effect=lambda g:(list(g),{u:20*GIB for u in g},[])):
                chosen,calls=helper.run_tuning(helper.workspace(tmp),world=world)
                self.assertEqual(chosen['global_batch_size'],4)
                self.assertEqual(chosen['world_size'],min(4,world))
                self.assertTrue(all(c[2]['global_batch_size']==4 for c in calls))
                self.assertEqual(sum(chosen['local_batch_sizes']),4)

    def test_fixed_gpu_oom_requests_more_cards_instead_of_lowering_batch(self):
        from test_automatic_execution import AutomaticExecutionTests
        import cooperative_control as control
        from ddp_runtime import RankGroupFailure
        helper=AutomaticExecutionTests();calls=[]
        def trial(work,fold,method,gpus,local,resource,kind,**kw):
            calls.append(resource['global_batch_size'])
            if len(calls)==1:raise RankGroupFailure('injected OOM',[dict(classification='gpu_oom')])
            return helper.measurement(gpus,local,resource)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'WHOLE_HEART_GLOBAL_BATCH':'4'}),\
             patch.object(control,'admit',side_effect=lambda g:(list(g),{u:20*GIB for u in g},[])),\
             patch.object(control,'trial',side_effect=trial),patch('task_resources.probe',return_value=dict(cpu_slots=128,available_memory_bytes=200*GIB)),\
             patch.object(control,'_larger_fixed_group',return_value=(['a','b'],{'a':20*GIB,'b':20*GIB},[])),patch.object(control,'topology',return_value=[]):
            chosen=control.tune(helper.workspace(tmp),'imagechd_fold0','S1','formal',['a'])
        self.assertEqual(chosen['global_batch_size'],4)
        self.assertEqual(chosen['local_batch_sizes'],[2,2])
        self.assertEqual(calls,[4,4,4])
    def test_requested_default_and_explicit_partitions(self):
        from batch_distribution import requested_batch,layout,validate_global_batch
        with patch.dict(os.environ,{},clear=True):self.assertEqual(requested_batch(),4)
        with patch.dict(os.environ,{'WHOLE_HEART_GLOBAL_BATCH':'3'}):self.assertEqual(requested_batch(),3)
        for world,expected in [(1,[4]),(2,[2,2]),(3,[2,1,1]),(4,[1,1,1,1])]:
            self.assertEqual(layout(4,world,global_batch=4),expected)
        with self.assertRaises(ValueError):layout(4,5,global_batch=4)
        for bad in (0,-1,True,2.5,'2.5'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_global_batch(bad)


if __name__=='__main__':unittest.main()
