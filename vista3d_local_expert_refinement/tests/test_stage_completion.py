"""CPU-only stage, diagnostic and GPU lifetime regression contracts."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


class CaseStages(unittest.TestCase):
    def test_case_producer_hash_participates_in_strict_shared_identity(self):
        from preparation_identity import current,FILES
        from test_task_lifecycle import InputTests
        from task_inputs import inspect_input,CacheMismatch
        from server_data import read_json,write_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,source,work,prep=InputTests().fixture(root)
            signature=read_json(prep/'server_preparation_signature.json')
            self.assertIn('case_execution.py',signature['implementation_sha256'])
            signature['implementation_sha256']['case_execution.py']='changed-helper'
            write_json(prep/'server_preparation_signature.json',signature)
            with self.assertRaises(CacheMismatch):inspect_input(config)
            for name in FILES:(root/name).write_text('same')
            before=current(root);(root/'case_execution.py').write_text('changed only helper')
            after=current(root)
            self.assertNotEqual(before,after)
            self.assertEqual(before['server_prepare.py'],after['server_prepare.py'])

    def test_stage_entry_refresh_rejects_known_zero_cpu_or_ram(self):
        import task_resources as resource
        for facts in [dict(cpu_budget=0,memory_budget_bytes=10),dict(cpu_budget=4,memory_budget_bytes=0)]:
            with tempfile.TemporaryDirectory() as temp,patch.object(resource,'probe',return_value=facts) as observe:
                with self.assertRaises(MemoryError):resource.begin_cpu_stage(temp,'prepare')
                observe.assert_called_once()

    def test_zero_queue_stops_instead_of_deadlocking(self):
        from cpu_case_queue import run
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex((ValueError,MemoryError),'capacity|容量'):
            run([('case',['unused'])],0,temp,lambda _:None)

    def test_blosc_adapter_overrides_explicit_threads_and_restores(self):
        import types
        from case_execution import leaf_threads
        calls=[]
        fake=types.SimpleNamespace(asarray=lambda *a,**kw:calls.append(kw),get_nthreads=lambda:1,set_nthreads=lambda n:1)
        original=fake.asarray
        with patch.dict(sys.modules,{'blosc2':fake}):
            with leaf_threads(15) as evidence:
                fake.asarray([1],chunks=(1,),cparams={'nthreads':4,'clevel':5,'codec':'LZ4HC','filters':['shuffle']})
            self.assertIs(fake.asarray,original)
        self.assertEqual(calls[0]['cparams']['nthreads'],15)
        self.assertEqual(calls[0]['cparams']['clevel'],5)
        self.assertEqual(evidence['blosc_calls'][0]['nthreads'],15)

    def test_header_budget_twelve_cases_threads_above_four(self):
        import types
        import task_resources as resource
        image=types.SimpleNamespace(shape=(8,8,8),header=types.SimpleNamespace(get_zooms=lambda:(1,1,1)))
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'nibabel':types.SimpleNamespace(load=lambda _:image)}),patch.object(resource,'probe',return_value=dict(cpu_budget=191,memory_budget_bytes=200*2**30)):
            value=resource.case_budget(temp,['x']*12)
        self.assertEqual(value['workers'],12)
        self.assertEqual(value['threads_per_worker'],15)


class DiagnosticExport(unittest.TestCase):
    def test_archive_excludes_raw_weights_credentials_and_redacts_log_text(self):
        import zipfile
        from diagnostic_export import export
        from server_data import write_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'job';work=job/'workspace'
            write_json(job/'plan.json',dict(workspace=str(work),api_key='private-value'))
            for path in ['diagnostics/events/1.jsonl','07_experiments/tuning/a/ranks/rank0/stderr.log','00_admin/stage-decisions/one.json','09_reports/assessment.json']:
                p=work/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('password=private-value',encoding='utf-8')
            for path in ['diagnostics/secret.json','diagnostics/raw/image.nii.gz','07_experiments/tuning/a/checkpoint.pth','env/secret.txt']:
                p=work/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('private-value')
            archive=export(job,root/'exports')
            with zipfile.ZipFile(archive) as z:
                names=z.namelist();contents=b''.join(z.read(n) for n in names)
            self.assertTrue(any(n.endswith('stderr.log') for n in names))
            self.assertFalse(any(n.endswith('.pth') or 'secret.json' in n or 'image.nii' in n for n in names))
            self.assertNotIn(b'private-value',contents)

    def test_reparse_attribute_is_rejected_even_without_symlink_privilege(self):
        from diagnostic_export import export
        from server_data import write_json
        import types
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'job';work=job/'workspace'
            write_json(job/'plan.json',dict(workspace=str(work)))
            path=work/'diagnostics/reparse.log';path.parent.mkdir(parents=True);path.write_text('fixture')
            original=Path.lstat
            def attribute(candidate,*args,**kwargs):
                result=original(candidate,*args,**kwargs)
                if candidate==path:return types.SimpleNamespace(st_mode=result.st_mode,st_file_attributes=0x400)
                return result
            with patch.object(Path,'lstat',attribute),self.assertRaisesRegex(ValueError,'符号链接|联接'):export(job,root/'exports')

    def test_secret_key_block_and_symlink_are_rejected_or_removed(self):
        from diagnostic_export import scrub,export
        from server_data import write_json
        self.assertNotIn('secret-body',scrub('before\n-----BEGIN PRIVATE KEY-----\nsecret-body\n-----END PRIVATE KEY-----\nafter'))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'job';work=job/'workspace';(work/'diagnostics').mkdir(parents=True)
            write_json(job/'plan.json',dict(workspace=str(work)))
            secret=root/'outside.log';secret.write_text('private')
            try:(work/'diagnostics/link.log').symlink_to(secret)
            except OSError:self.skipTest('host does not permit creating symlink fixture')
            with self.assertRaisesRegex(ValueError,'符号链接|联接'):export(job,root/'exports')

    def test_archive_rejects_escaped_workspace(self):
        from diagnostic_export import export
        from server_data import write_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write_json(root/'job/plan.json',dict(workspace=str(root/'outside')))
            with self.assertRaises(ValueError):export(root/'job',root/'exports')


class PredictionLifecycle(unittest.TestCase):
    def test_twelve_cases_use_all_eight_admitted_workers_and_preserve_partial_failure(self):
        import persistent_prediction as resident
        from server_data import write_json,read_json
        for failure in [False,True]:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                work=Path(temp);completed=set();processes=[];assignments=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        self.directory=Path(command[-2]);self.pid=10+len(processes);self.returncode=None;self.sequence=0
                        processes.append(self)
                    def poll(self):
                        request=self.directory/'request.json'
                        if request.exists():
                            record=read_json(request)
                            if record['id']!=self.sequence:
                                self.sequence=record['id'];rows=record['records'];assignments.append((self.pid,[r['case_id'] for r in rows]))
                                if failure:
                                    completed.add(rows[0]['case_id'])
                                    write_json(self.directory/'failure.json',{'error':'synthetic case read failure'})
                                    return None  # structured failure must wake parent before exit/child cleanup
                                completed.update(r['case_id'] for r in rows)
                                write_json(self.directory/'response.json',{'id':self.sequence})
                        return self.returncode
                    def wait(self,timeout=None):self.returncode=0;return 0
                records=[dict(fold='f',method='B0',index=0,case_id=str(i),status='pending') for i in range(12)]
                with patch.dict(os.environ,{},clear=True),patch('evaluation_gpu.idle_gpus',return_value=[f'gpu{i}' for i in range(8)]),                     patch('task_resources.prediction_budget',return_value=dict(workers=8,threads_per_worker=2,cpu_budget=16)),                     patch('evaluation_contract.cached',side_effect=lambda w,m,c,s,p:c in completed),                     patch.object(resident.subprocess,'Popen',Process),patch.object(resident,'stop_owned_process'),patch.object(resident.time,'sleep'):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'synthetic case read failure'):
                            resident.run(records,[dict(fold='f',method='B0')],'sha',dict(max_gpus=8),work)
                    else:resident.run(records,[dict(fold='f',method='B0')],'sha',dict(max_gpus=8),work)
                state=read_json(work/'08_results/evaluation/status.json')
                if failure:
                    self.assertEqual(state['status'],'failed')
                    self.assertTrue(any(r['status']=='completed' for r in state['jobs']))
                else:
                    self.assertEqual(len({pid for pid,_ in assignments}),8)
                    self.assertEqual(len(completed),12)
                    self.assertTrue(all(len(cases)<=2 for _,cases in assignments))

    def test_resident_worker_reuses_model_and_prep_child_hides_cuda(self):
        import types
        from server_data import write_json,read_json,digest
        import persistent_prediction as resident
        calls=[];loaded=[]
        class Future:
            def result(self):return {'prepared':True}
        class Pool:
            def __init__(self,**kw):self.kw=kw
            def submit(self,fn,*args):calls.append(('prepare',args[2],os.environ['CUDA_VISIBLE_DEVICES']));return Future()
            def shutdown(self,**kw):calls.append(('closed',))
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);directory=work/'session';directory.mkdir()
            checkpoint=work/'checkpoint';checkpoint.write_text('fixture')
            model=dict(fold='f',method='B0',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),targets=['a','b','c'])
            frozen=work/'08_results/evaluation/frozen_models.json';write_json(frozen,dict(models=[model]))
            write_json(directory/'request.json',dict(id=1,threads=2,records=[dict(index=0,case_id='a'),dict(index=0,case_id='b')]))
            predictor=object()
            def make(_):loaded.append(1);return predictor,None
            def infer(work,model,cid,sha,**kw):
                self.assertIs(kw['predictor'],predictor);calls.append(('infer',cid))
            def publish(path,value):
                write_json(path,value)
                if Path(path).name=='response.json':
                    if value['id']==1:write_json(directory/'request.json',dict(id=2,threads=2,records=[dict(index=0,case_id='c')]))
                    else:write_json(directory/'stop.json',{})
            torch=types.SimpleNamespace(set_num_threads=lambda n:None,cuda=types.SimpleNamespace(empty_cache=lambda:None))
            with patch.dict(os.environ,{'WHOLE_HEART_WORKSPACE':str(work),'CUDA_VISIBLE_DEVICES':'GPU-one'}),                 patch.dict(sys.modules,{'torch':torch,'evaluation':types.SimpleNamespace(make_predictor=make),
                    'evaluation_phases':types.SimpleNamespace(preprocess_case=lambda *a:None,infer=infer),
                    'prepare_nnunet':types.SimpleNamespace(configure_paths=lambda:None)}),                 patch('concurrent.futures.ProcessPoolExecutor',Pool),patch.object(resident,'write_json',side_effect=publish):
                resident.worker(directory,digest(frozen))
            self.assertEqual(len(loaded),1)
            self.assertEqual([r[1] for r in calls if r[0]=='infer'],['a','b','c'])
            self.assertEqual(calls[0],('prepare','a',''))
            self.assertLess(calls.index(('prepare','b','GPU-one')),calls.index(('infer','a')))
            self.assertEqual(calls[-1],('closed',))

    def test_assessment_releases_unused_then_exits_gpu_before_tail_and_identity_never_waits_after(self):
        from server_data import write_json
        import assessment_run
        events=[]
        class Fleet:
            def __init__(self,job,*args):self.path=Path(job)/'diagnostics/gpu-leases.json';self.closed=False
            def start(self):events.append('start');return self
            def devices(self,wait=False):return ['gpu0','gpu1']
            def suspend_discovery(self,used):events.append(('suspend',used))
            def close(self):self.closed=True;events.append('exit_all_keepers')
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)/'workspace';write_json(work/'04_data/manifests/case_inventory.json',dict(cases=[dict(case_id='cal',image='synthetic')]))
            plan=dict(workspace=str(work),runtime=str(Path(temp)/'runtime'),data='synthetic',gpu_memory=20,
                folds=['f'],methods=['B0'],formal_budget={},assessment_budget={})
            def identity(*args):events.append('identity');return {'provenance':True}
            def child(command,env,**kw):
                phase=command[-1];events.append(phase)
                if phase=='cpu':
                    self.assertEqual(events[-2],'budget_tail');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
                    write_json(work/'09_reports/assessment_prediction_f_B0.json',dict(status='passed'))
            def tail(*a,**kw):
                self.assertEqual(events[-1],'exit_all_keepers');events.append('budget_tail')
                return dict(workers=1,threads_per_worker=2)
            with patch('assessment_prepare.prepare',return_value={'prediction_ids':['cal']}),                 patch('cooperative_control.tune',return_value=dict(gpu_uuids=['gpu0','gpu1'],output='trial')),                 patch('cooperative_control.identity',side_effect=identity),patch('gpu_fleet.GPUFleet',Fleet),                 patch('task_worker.run_child',side_effect=child),patch('runtime_observations.collect'),                 patch('task_resources.prediction_budget',return_value=dict(workers=1,threads_per_worker=2)),                 patch('task_resources.case_budget',side_effect=tail):
                report=assessment_run.run(plan)
            self.assertLess(events.index('identity'),events.index(('suspend',['gpu0'])))
            self.assertLess(events.index('gpu'),events.index('exit_all_keepers'))
            self.assertLess(events.index('exit_all_keepers'),events.index('cpu'))
            self.assertEqual(events.count('identity'),1)
            self.assertEqual(report['models'][0]['identity'],{'provenance':True})

    def test_formal_failure_exits_workers_and_releases_leases_without_tail(self):
        import evaluation_parallel
        events=[]
        model=dict(fold='f',method='B0',targets=['a'])
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'VISTA_GPU_LEASES':'fixture'}),             patch('evaluation_contract.cached',return_value=False),             patch('persistent_prediction.run',side_effect=RuntimeError('actual worker failed')),             patch('gpu_fleet.verified_keepers',return_value={'gpu0':{}}),             patch('cooperative_control.release_devices',side_effect=lambda *a,**k:events.append(k)),             patch.object(evaluation_parallel,'run_phase') as tail:
            with self.assertRaisesRegex(RuntimeError,'actual worker failed'):
                evaluation_parallel.run_queue([model],'sha',{},Path(temp))
            tail.assert_not_called();self.assertEqual(events,[{'wait':True,'final':True}])


class ReportingContracts(unittest.TestCase):
    def test_large_budget_caps_numexpr_only_and_preserves_explicit_max(self):
        from task_resources import thread_environment
        value=thread_environment(191,{})
        self.assertEqual(value['NUMEXPR_NUM_THREADS'],'64')
        self.assertEqual(value['OMP_NUM_THREADS'],'191')
        self.assertEqual(thread_environment(191,{'NUMEXPR_MAX_THREADS':'32'})['NUMEXPR_NUM_THREADS'],'32')

    def test_isolated_release_prepare_train_predict_graph_keeps_runtime_and_strict_cache(self):
        import shutil
        import task_runtime as jobs
        import task_inputs
        from server_data import write_json,read_json,digest
        from test_task_lifecycle import InputTests
        from test_task_release_paths import ManagedReleasePaths
        from pipeline_catalog import latest,history
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,release=ManagedReleasePaths().fixture(root)
            config.update(catalog_schema=2,stage='prepare',tasks=['prepare'],gpu_memory=5,
                preprocessor='nnunet_default',results=str(root/'results/pipeline_v2/prepare'))
            seed=root/'seed';seed.mkdir();_,source,_,_=InputTests().fixture(seed)
            def complete(job):
                state=read_json(job/'state.json');state['status']='completed'
                for step in state['steps'].values():step['status']='completed'
                write_json(job/'state.json',state)
            with patch.object(jobs,'CODE',release),patch.object(jobs,'launch',return_value=0):
                jobs.submit(config);prepare=latest(config['runtime']);plan=jobs.load_plan(prepare)
                work=Path(plan['workspace']);shutil.copytree(source,work)
                metadata=read_json(work/'00_admin/server_config.json');metadata['data_root']=config['data'];metadata['work_root']=str(work)
                write_json(work/'00_admin/server_config.json',metadata);complete(prepare)
                self.assertIsNone(task_inputs.source_compatibility(prepare))
                training=dict(config,name='train',stage='train',tasks=['train'],gpus=['GPU-test'],gpu_count=1,
                    input_job=str(prepare),upstream_jobs=[str(prepare)],results=str(root/'results/pipeline_v2/train'))
                jobs.submit(training);train=latest(config['runtime']);train_plan=jobs.load_plan(train)
                task_inputs.copy_input(train_plan);train_work=Path(train_plan['workspace'])
                out=train_work/'07_experiments/formal/ct_holdG_B0_seed0'
                write_json(out/'run.json',dict(status='completed',formal_training=True))
                for name in ['checkpoint_best.pth','checkpoint_final.pth']:
                    file=out/'fold_0'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'synthetic checkpoint marker')
                complete(train)
                prediction=dict(training,name='predict',stage='predict',tasks=['predict'],input_job=str(train),
                    upstream_jobs=[str(train)],results=str(root/'results/pipeline_v2/predict'))
                jobs.submit(prediction);predict=latest(config['runtime']);predict_plan=jobs.load_plan(predict)
                task_inputs.copy_input(predict_plan)
                self.assertEqual(predict_plan['runtime'],config['runtime'])
                self.assertEqual(predict_plan['tasks'],['predict']);self.assertEqual(train_plan['tasks'],['train'])
                self.assertEqual({r['stage'] for r in history(config['runtime'])},{'prepare','train','predict'})
                self.assertTrue((Path(predict_plan['workspace'])/'07_experiments/formal/ct_holdG_B0_seed0/fold_0/checkpoint_best.pth').is_file())
                prep=work/'04_data/derived/nnUNet_preprocessed/Dataset813_ct_holdG/server_preparation_signature.json'
                marker=read_json(prep);marker['implementation_sha256']['server_prepare.py']='old-code';write_json(prep,marker)
                self.assertIn('不兼容',task_inputs.source_compatibility(prepare))
                with self.assertRaisesRegex(task_inputs.CacheMismatch,'新建任务'):jobs.submit(training)

    def test_assessment_cpu_tail_does_not_display_earlier_gpu_decision(self):
        from task_progress import resource_lines
        from server_data import write_json
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            write_json(work/'00_admin/stage-decisions/assessment_f_B0_1.json',dict(resource_snapshot={'cpu_budget':100},effective_cpu={'total_cpu_slots':90}))
            write_json(work/'00_admin/cpu-case-budget.json',dict(stage='assessment_cpu_tail',cpu_budget=4,total_cpu_slots=4,workers=1,threads_per_worker=4))
            text=' '.join(resource_lines(work,dict(current_step='assessment',status='running')))
            self.assertIn('assessment_cpu_tail',text);self.assertIn('已分配 4',text);self.assertNotIn('已分配 90',text)

    def test_feedback_applied_settings_and_sampling_both_phases(self):
        from task_progress import resource_lines
        from server_data import write_json
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            write_json(work/'00_admin/stage-decisions/formal_f_B0_1.json',dict(global_batch_size=6,reference_global_batch_size=2,
                resource_snapshot={'cpu_budget':20},effective_cpu={'total_cpu_slots':4,'rank_settings':[{'workers':1}]}))
            write_json(work/'07_experiments/formal/f_B0/ranks/feedback.rank0.json',[
                dict(window=dict(wait_fraction=.8,rank_windows=[{'rank':0},{'rank':1}]),decision=dict(
                    applied_boundary='replay_safe_batch',settings={'total_cpu_slots':12,'rank_settings':[{'workers':5}]})),
                dict(window=dict(wait_fraction=.1),decision=dict(applied_boundary='deferred_full_epoch',settings={'total_cpu_slots':16}))])
            text=' '.join(resource_lines(work,{'current_step':'train','status':'running'}))
            self.assertIn('已分配 12',text);self.assertNotIn('已分配 16',text)
            self.assertIn('训练 1500/500',text);self.assertIn('验证 300/100',text)
            self.assertIn('实际CPU用量缺少',text)

    def test_failure_uses_structured_cause_and_log_path_without_secret(self):
        from failure_summary import summarize
        from server_data import write_json
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            write_json(work/'07_experiments/tuning/x/ranks/rank-1/failure.json',dict(message='bad geometry token=private-key',classification='noncapacity'))
            text=summarize(work,RuntimeError('Command returned exit status 1'),'step.log')
            self.assertIn('bad geometry',text);self.assertIn('rank-1',text);self.assertIn('step.log',text)
            self.assertNotIn('private-key',text)

    def test_incompatible_source_is_not_offered_and_no_prepare_is_injected(self):
        import task_panel as panel
        with patch('pipeline_catalog.history',return_value=[dict(name='old',directory='old',stage='prepare',status='completed')]),             patch('task_inputs.source_compatibility',return_value='旧实现不兼容'),patch.object(panel,'ask') as ask:
            with self.assertRaisesRegex(ValueError,'先完成对应板块'):panel.choose_stage_source('runtime','prepare')
            ask.assert_not_called()

    def test_both_compression_and_cpu_case_queue_respect_fresh_threads(self):
        import cpu_case_queue as queue
        from unittest.mock import Mock
        process=Mock(pid=1);process.poll.return_value=0
        with tempfile.TemporaryDirectory() as temp,patch.object(queue.subprocess,'Popen',return_value=process) as start,patch.object(queue.time,'sleep'):
            queue.run([('a',['fixture'])],1,temp,lambda _:None,refresh=lambda:dict(workers=1,threads_per_worker=9,cpu_budget=9))
            self.assertEqual(start.call_args.kwargs['env']['WHOLE_HEART_CPU_THREADS'],'9')


if __name__=='__main__':unittest.main()
