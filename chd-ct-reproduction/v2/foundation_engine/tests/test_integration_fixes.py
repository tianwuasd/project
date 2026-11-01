"""Focused integration review regressions; files and simulated resources only."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from server_data import write_json,read_json,digest
import cooperative_control as control
import evaluation_parallel as queue
import gpu_fleet
import pipeline_catalog as catalog
import task_panel as panel
import task_queries as queries
import task_resources as resources


class CatalogBoundaryTests(unittest.TestCase):
    def fixture(self,temp):
        root=Path(temp).resolve();runtime=root/'runtime';job=root/'results/pipeline_v2/prepare/job'
        job.mkdir(parents=True)
        plan=dict(catalog_schema=2,tasks=['prepare'],runtime=str(runtime),directory=str(job),
            results=str(job.parent),workspace=str(job/'workspace'),data=str(root/'raw'),name='synthetic',
            created='2026-10-27',upstream_jobs=[])
        write_json(job/'plan.json',plan)
        write_json(job/'state.json',dict(status='completed',plan_sha256=digest(job/'plan.json')))
        catalog.register(runtime,job)
        return root,runtime,job,plan

    def damage_workspace(self,root,job,plan):
        old=root/'unindexed-old-workspace';old.mkdir()
        sentinel=old/'00_admin/gpu_queue.json'
        write_json(sentinel,dict(private_sentinel='never-read'))
        write_json(job/'plan.json',dict(plan,workspace=str(old)))
        write_json(job/'state.json',dict(status='completed',plan_sha256=digest(job/'plan.json')))
        return sentinel

    def test_changed_workspace_rejected_at_actual_cli_for_queries_stop_restart_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            root,runtime,job,plan=self.fixture(temp);sentinel=self.damage_workspace(root,job,plan)
            self.assertEqual(catalog.history(runtime)[0]['status'],'记录缺失或损坏')
            original=Path.read_text;reads=[]
            def read(path,*args,**kwargs):
                reads.append(path);return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',read),patch.object(panel.jobs,'stop') as stop,\
                 patch.object(panel.jobs,'resume') as resume,patch.object(panel,'query') as query,\
                 patch('panel_publication.sharing.run_upload') as upload:
                cases=[['--job',str(job),'--query',kind] for kind in ['status','logs','results','processes']]
                cases += [['--job',str(job),'--stop'],['--resume-job',str(job)],['--upload-job',str(job)]]
                for args in cases:
                    with self.subTest(args=args),self.assertRaisesRegex(ValueError,'损坏'):
                        panel.main(['--runtime',str(runtime),'--loaded-release',*args])
                for action in [stop,resume,query,upload]:action.assert_not_called()
            self.assertNotIn(sentinel,reads)

    def test_direct_queries_restart_upload_and_selection_reject_damaged_handle(self):
        from panel_publication.sharing import run_upload
        with tempfile.TemporaryDirectory() as temp:
            root,runtime,job,plan=self.fixture(temp);self.damage_workspace(root,job,plan)
            args=panel.make_parser().parse_args(['--runtime',str(runtime),'--upload-job',str(job)])
            with patch.dict(os.environ,{'VISTA_CATALOG_VERSION':'2'}),patch.object(panel,'ask',return_value='1'):
                for action in [lambda:queries.query('status',runtime,job),lambda:panel.choose_job(runtime),
                    lambda:panel.restart_job(args,runtime,job),lambda:run_upload(args,dict(runtime=str(runtime)),False)]:
                    with self.assertRaisesRegex(ValueError,'损坏'):action()

    def test_panel_never_reprobes_rejected_process_metadata(self):
        import server_profile
        with tempfile.TemporaryDirectory() as temp:
            root,runtime,job,plan=self.fixture(temp)
            (job/'processes').write_text('not a process directory',encoding='utf-8')
            self.assertEqual(catalog.history(runtime)[0]['status'],'记录缺失或损坏')
            args=panel.make_parser().parse_args(['--runtime',str(runtime)])
            profile=dict(runtime=str(runtime),results=str(root/'results'),data=str(root/'raw'))
            with patch.dict(os.environ,{'VISTA_CATALOG_VERSION':'2'}),\
                 patch.object(panel.jobs,'active_processes',side_effect=AssertionError('raw probe rejected row')),\
                 patch.object(panel,'ask',return_value='0'),patch.object(server_profile,'load',return_value=profile),\
                 patch.object(server_profile,'show'),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(panel.panel(args,str(runtime),{}),0)


class PredictionGateTests(unittest.TestCase):
    def test_eight_idle_cards_obey_live_one_worker_host_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);live=set();peaks=[];processes=[]
            class Process:
                def __init__(self,*args,**kwargs):
                    self.pid=100+len(processes);self.polls=0
                    processes.append(self);live.add(self.pid);peaks.append(len(live))
                    self.env=kwargs['env']
                def poll(self):
                    self.polls+=1
                    if self.polls<2:return None
                    live.discard(self.pid);return 0
            records=[dict(fold='imagechd_fold0',method='S1',index=0,case_id=f'case{i}',status='pending') for i in range(4)]
            with patch.object(queue,'idle_gpus',return_value=[f'GPU-{i}' for i in range(8)]),\
                 patch.object(queue,'prediction_budget',return_value=dict(workers=1,threads_per_worker=1,capacity_status='bounded')) as gate,\
                 patch.object(queue.subprocess,'Popen',Process),patch.object(queue.time,'sleep'),\
                 patch.object(queue.time,'monotonic',side_effect=range(10000)),contextlib.redirect_stdout(io.StringIO()):
                queue.run_phase(records,'sha',dict(max_gpus=8,retry_seconds=1),root)
            self.assertEqual(max(peaks),1);self.assertGreaterEqual(gate.call_count,4)
            self.assertEqual({p.env['WHOLE_HEART_CPU_THREADS'] for p in processes},{'1'})

    def test_known_insufficient_one_case_stops_before_infer(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(queue,'idle_gpus',return_value=['GPU-0']),\
                 patch.object(queue,'prediction_budget',return_value=dict(workers=0,threads_per_worker=1,capacity_status='known_insufficient_memory')),\
                 patch.object(queue.subprocess,'Popen') as launch,self.assertRaisesRegex(MemoryError,'未启动GPU'):
                queue.run_phase([dict(fold='f',method='S1',index=0,case_id='c',status='pending')],
                    'sha',dict(max_gpus=8,retry_seconds=1),Path(temp))
            launch.assert_not_called()

    def test_header_budget_accounts_for_resampling_cpu_ram_io_and_unknowns(self):
        class Image:
            shape=(10,20,30)
            header=Mock(get_zooms=lambda:(2.,3.,4.))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'04_data/manifests/case_inventory.json',dict(cases=[dict(case_id='c',image='synthetic')]))
            write_json(root/'04_data/manifests/splits_v1.json',dict(folds={'f':dict(dataset_id=813)}))
            write_json(root/'04_data/derived/nnUNet_preprocessed/Dataset813_f/WholeheartBaselinePlans.json',
                dict(configurations={'3d_fullres':dict(spacing=[2.,1.,1.])}))
            snapshots=[dict(cpu_slots=8,available_memory_bytes=100*2**30,io_pressure={'avg10':20}),
                       dict(cpu_slots=1,available_memory_bytes=2**30),dict(cpu_slots=8,available_memory_bytes=None)]
            with patch.dict(sys.modules,{'nibabel':Mock(load=Mock(return_value=Image()))}),\
                 patch.object(resources,'probe',side_effect=snapshots),patch.dict(os.environ,{'WHOLE_HEART_CPU_THREADS':'2'}):
                reports=[resources.prediction_budget(root,[dict(fold='f',case_id='c')],8) for _ in snapshots]
            self.assertEqual(reports[0]['cases'][0]['resampled_shape_plans'],[60,60,20])
            self.assertEqual(reports[0]['workers'],1)
            self.assertEqual(reports[1]['workers'],0);self.assertEqual(reports[1]['capacity_status'],'known_insufficient_cpu')
            self.assertEqual(reports[2]['workers'],1);self.assertIn('unknown',reports[2]['capacity_status'])


class TopologyAndRegroupTests(unittest.TestCase):
    def setUp(self):
        helper=patch.dict(sys.modules,{'server_train':Mock(select_checkpoint=Mock(),archive_uncheckpointed=Mock())})
        helper.start();self.addCleanup(helper.stop)

        admission=patch.object(control,'admit',side_effect=lambda g:(list(g),{u:20*2**30 for u in g},[]))
        admission.start();self.addCleanup(admission.stop)

    def workspace(self,temp):
        root=Path(temp)
        write_json(root/'04_data/manifests/splits_v1.json',dict(folds={'imagechd_fold0':dict(dataset_id=813)}))
        write_json(root/'04_data/manifests/imagechd_fold0_source_labels.json',{})
        write_json(root/'04_data/derived/nnUNet_preprocessed/Dataset813_imagechd_fold0/WholeheartBaselinePlans.json',
            dict(configurations={'3d_fullres':dict(batch_size=2,patch_size=[16,16,16])}))
        for file in ['pip_freeze_training.txt','conda_explicit_training.txt']:
            path=root/'06_configs'/file;path.parent.mkdir(exist_ok=True);path.write_text('synthetic',encoding='utf-8')
        return root

    def measurement(self,gpus,resource,seconds=1):
        return dict(status='passed',world_size=len(gpus),gpu_uuids=list(gpus),local_batch_size=2,
            global_batch_size=2*len(gpus),median_step_seconds=seconds,validation_step_seconds=.1,
            measured_training_steps=10,total_cpu_slots=resource['total_cpu_slots'],effective_cpu=resource,
            train_patches_per_epoch=500*len(gpus),reference_train_patches_per_epoch=500,peak_reserved_gib=20)

    def test_stage_group_is_frozen_until_full_epoch_membership_boundary(self):
        snapshot=dict(cpu_slots=16,available_memory_bytes=100*2**30,shared_memory_available_bytes=100*2**30)
        for explicit in [False,True]:
            with self.subTest(explicit=explicit),tempfile.TemporaryDirectory() as temp:
                root=self.workspace(temp);calls=[]
                def measure(w,f,m,g,l,r,k,**kwargs):
                    calls.append(list(g));return self.measurement(g,r,2 if len(g)==1 else 1)
                with patch('task_resources.probe',return_value=snapshot),patch.object(control,'trial',side_effect=measure),\
                     patch.object(control,'devices',side_effect=[['GPU-0'],['GPU-0','GPU-1','GPU-2']]) as available,\
                     patch.object(control,'topology',side_effect=lambda g:[['same','24576','driver'] for _ in g]),\
                     patch.object(control,'release_devices'),contextlib.redirect_stdout(io.StringIO()):
                    chosen=control.tune(root,'imagechd_fold0','S1','assessment',['GPU-0'] if explicit else None)
                self.assertEqual(calls,[['GPU-0']])
                self.assertEqual(available.call_count,0 if explicit else 1)
                self.assertFalse(chosen['resource_receipt_reusable'])

    def test_receipt_one_card_requires_bounded_test_before_first_three_card_formal_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.workspace(temp);runtime=root/'runtime';order=[]
            receipt_report=root/'assessment.json';write_json(receipt_report,dict(status='passed'))
            config=self.measurement(['GPU-0'],dict(total_cpu_slots=1),2)
            write_json(runtime/'optimization/imagechd_fold0_B0.json',dict(chosen=config,assessment_report=str(receipt_report),
                assessment_sha256=digest(receipt_report),measurement_coverage=[dict(status='passed',world_size=1,
                    hardware=[['same','24576','driver']],effective_cpu={'total_cpu_slots':1},local_batch_size=2)]))
            def tune(*args,**kwargs):
                order.append('tune');return self.measurement(['GPU-0','GPU-1','GPU-2'],dict(total_cpu_slots=3))
            def launch(target,args,gpus,*other,**kwargs):
                order.append(('formal',list(gpus)));return [dict(result=dict(status='completed',checkpoint='cp',next_epoch=250,initial_weight_sha256='weight'))]
            with patch.dict(os.environ,{'VISTA_RUNTIME':str(runtime)}),patch.object(control,'identity',return_value={}),\
                 patch('training_policy.receipt_compatibility',return_value=[]),patch('task_resources.probe',return_value={}),\
                 patch.object(control,'devices',return_value=['GPU-0','GPU-1','GPU-2']),\
                 patch.object(control,'topology',side_effect=lambda g:[['same','24576','driver'] for _ in g]),\
                 patch.object(control,'tune',side_effect=tune),patch.object(control,'release_devices'),\
                 patch('ddp_runtime.launch_ranks',side_effect=launch),contextlib.redirect_stdout(io.StringIO()):
                control.train(root,'imagechd_fold0','S1')
            self.assertEqual(order,['tune',('formal',['GPU-0','GPU-1','GPU-2'])])

    def test_equivalent_replacement_has_explicit_count_coverage(self):
        coverage=[dict(status='passed',world_size=3,hardware=[['same','24576','driver']]*3,
            local_batch_size=2,effective_cpu={'total_cpu_slots':3})]
        with patch.object(control,'topology',side_effect=lambda g:[['same','24576','driver'] for _ in g]):
            self.assertTrue(control.coverage_matches(coverage,['replacement-A','replacement-B','replacement-C']))
            self.assertFalse(control.coverage_matches(coverage,['replacement-A']))

    def test_covered_old_receipt_never_bypasses_fresh_formal_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.workspace(temp);runtime=root/'runtime';groups=[]
            report=root/'assessment.json';write_json(report,dict(status='passed'))
            config=self.measurement(['old-UUID'],dict(total_cpu_slots=1),2)
            config['selected_hardware']=[['same','24576','driver']]
            write_json(runtime/'optimization/imagechd_fold0_B0.json',dict(chosen=config,assessment_report=str(report),
                assessment_sha256=digest(report),measurement_coverage=[dict(status='passed',world_size=3,
                    hardware=[['same','24576','driver']]*3,effective_cpu={'total_cpu_slots':3},local_batch_size=2)]))
            class Watch:
                def __init__(self,target,**kwargs):self.target=target
                def start(self):self.target()
                def join(self,**kwargs):pass
            class Event:
                def __init__(self):self.waits=0
                def wait(self,seconds):self.waits+=1;return self.waits>1
                def set(self):pass
            def launch(target,args,gpus,*other,**kwargs):
                self.assertFalse(Path(args['regroup_request']).exists())
                groups.append(list(gpus));return [dict(result=dict(status='completed',checkpoint='cp',next_epoch=250,initial_weight_sha256='weight'))]
            with patch.dict(os.environ,{'VISTA_RUNTIME':str(runtime)}),patch.object(control,'identity',return_value={}),\
                 patch('training_policy.receipt_compatibility',return_value=[]),patch('task_resources.probe',return_value={}),\
                 patch.object(control,'devices',return_value=['new-A','new-B','new-C']),\
                 patch.object(control,'topology',side_effect=lambda g:[['same','24576','driver'] for _ in g]),\
                 patch('evaluation_gpu.query',return_value=[[g,'same','24576','driver'] for g in ['new-A','new-B','new-C']]),\
                 patch.object(control,'tune',return_value=self.measurement(['new-A','new-B','new-C'],dict(total_cpu_slots=3))) as tune,patch.object(control,'release_devices') as release,\
                 patch.object(control.threading,'Thread',Watch),patch.object(control.threading,'Event',Event),\
                 patch('ddp_runtime.launch_ranks',side_effect=launch),contextlib.redirect_stdout(io.StringIO()):
                control.train(root,'imagechd_fold0','S1')
            tune.assert_called_once();self.assertEqual(groups,[['new-A','new-B','new-C']])
            release.assert_not_called()

    def test_regroup_keeps_all_ready_cards_without_a_speed_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.workspace(temp);requests=[];groups=[];tunes=[]
            initial=self.measurement(['GPU-0','GPU-1'],dict(total_cpu_slots=2),5)
            smaller=self.measurement(['GPU-0','GPU-1','GPU-2'],dict(total_cpu_slots=3),9)
            class Watch:
                def __init__(self,target,**kwargs):self.target=target
                def start(self):self.target()
                def join(self,**kwargs):pass
            class Event:
                def __init__(self):self.waits=0
                def wait(self,seconds):self.waits+=1;return self.waits>1
                def set(self):pass
            def tune(*args,**kwargs):tunes.append(args);return initial if len(tunes)==1 else smaller
            def launch(target,args,gpus,*other,**kwargs):
                groups.append(list(gpus));request=Path(args['regroup_request'])
                requests.append(read_json(request)['gpu_uuids'] if request.exists() else None)
                return [dict(result=dict(status='regroup' if len(groups)==1 else 'completed',checkpoint='cp',next_epoch=250,
                    initial_weight_sha256='weight'))]
            with patch.dict(os.environ,{'VISTA_RUNTIME':str(root/'runtime')}),patch.object(control,'identity',return_value={}),\
                 patch.object(control,'devices',return_value=['GPU-0','GPU-1','GPU-2']),patch.object(control,'tune',side_effect=tune),\
                 patch.object(control,'release_devices') as release,patch('ddp_runtime.launch_ranks',side_effect=launch),\
                 patch.object(control.threading,'Thread',Watch),patch.object(control.threading,'Event',Event),\
                 contextlib.redirect_stdout(io.StringIO()):
                control.train(root,'imagechd_fold0','S1')
            self.assertEqual(groups,[['GPU-0','GPU-1'],['GPU-0','GPU-1','GPU-2']])
            self.assertEqual(requests,[['GPU-0','GPU-1','GPU-2'],None])
            release.assert_not_called()


class DiscoveryLifecycleTests(unittest.TestCase):
    def test_final_release_waits_for_discovery_ack_even_with_no_verified_ready_keeper(self):
        with tempfile.TemporaryDirectory() as temp:
            lease=Path(temp)/'gpu-leases.json';write_json(lease,dict(generation='current'))
            def acknowledge(seconds):
                command=read_json(lease.with_name('gpu-release-request.json'))
                write_json(lease,dict(generation='current',discovery_enabled=False,release_ack=command['id']))
            with patch.dict(os.environ,{'VISTA_GPU_LEASES':str(lease)}),\
                 patch.object(gpu_fleet,'verified_keepers',return_value={}),patch.object(control.time,'sleep',side_effect=acknowledge) as wait:
                control.release_devices([],wait=True,final=True)
            self.assertEqual(wait.call_count,1)
            self.assertTrue(read_json(lease.with_name('gpu-release-request.json'))['final'])

    def watch(self,fleet,steps=3):
        class Stop:
            calls=0
            def wait(self,seconds):self.calls+=1;return self.calls>steps
        fleet.stop=Stop()
        row=dict(index='0',uuid='GPU-0',free_mib=24000,total_mib=24576,utilization=0,busy=False)
        with patch('evaluation_gpu.snapshot',return_value=[row]),patch('evaluation_gpu.allowed_ids',return_value={'GPU-0'}),\
             patch('evaluation_gpu.idle_reason',return_value=None),patch.object(fleet,'_acquire') as acquire,\
             patch('runtime_observations.event'),patch('task_resources.probe',return_value={}),\
             patch.object(gpu_fleet.time,'monotonic',side_effect=range(1000,10000,1000)):
            fleet._watch()
        return acquire

    def test_final_release_permanently_disables_reacquisition_beyond_ten_minutes(self):
        with tempfile.TemporaryDirectory() as temp:
            fleet=gpu_fleet.GPUFleet(Path(temp)/'job',Path(temp)/'runtime')
            process=Mock();stream=Mock();fleet.processes['GPU-0']=(process,stream,Path(temp)/'unready.json')
            write_json(fleet.path.with_name('gpu-release-request.json'),dict(id='end',generation=fleet.generation,
                gpu_uuids=[],final=True))
            acquire=self.watch(fleet)
            acquire.assert_not_called();self.assertFalse(fleet.discovery_enabled)
            self.assertEqual(read_json(fleet.path)['release_ack'],'end')
            process.terminate.assert_called_once();stream.close.assert_called_once()

    def test_restart_generation_ignores_previous_final_request(self):
        with tempfile.TemporaryDirectory() as temp:
            fleet=gpu_fleet.GPUFleet(Path(temp)/'job',Path(temp)/'runtime')
            write_json(fleet.path.with_name('gpu-release-request.json'),dict(id='old-end',generation='old-invocation',
                gpu_uuids=[],final=True))
            acquire=self.watch(fleet)
            self.assertTrue(acquire.called);self.assertTrue(fleet.discovery_enabled)
            self.assertIsNone(read_json(fleet.path)['release_ack'])


if __name__=='__main__':unittest.main()
