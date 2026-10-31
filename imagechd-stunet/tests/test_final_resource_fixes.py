"""Execution boundaries for the final resource repair; CPU-only."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))

class FinalResourceFixes(unittest.TestCase):
    def test_nested_official_seg_reader_uses_same_single_slot(self):
        from read_admission import create_gate,reader_scope
        class Reader:
            def read_images(self,value):return value
            def read_seg(self,value):return self.read_images(value)
        gate=create_gate(1)
        with patch.dict(os.environ,{'VISTA_READ_GATE':str(gate)}),reader_scope(Reader):
            self.assertEqual(Reader().read_seg('payload'),'payload')

    def test_actual_read_slots_release_before_compute_finishes(self):
        from read_admission import create_gate
        from cpu_case_queue import run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gate=create_gate(2)
            script=root/'worker.py'
            script.write_text("from pathlib import Path\nimport sys,time\nfrom read_admission import read_slot\nr=Path(sys.argv[1]);n=sys.argv[2]\nwith read_slot():\n (r/(n+'.read')).touch()\n while not (r/(n+'.release')).exists():time.sleep(.01)\n (r/(n+'.read')).unlink()\n(r/(n+'.compute')).touch()\nwhile not (r/'done').exists():time.sleep(.01)\n")
            import threading
            errors=[]
            def control():
                try:
                    end=time.time()+20
                    while len(list(root.glob('*.read')))<2 and time.time()<end:time.sleep(.02)
                    self.assertEqual(len(list(root.glob('*.read'))),2)
                    time.sleep(.2);self.assertEqual(len(list(root.glob('*.read'))),2)
                    first=next(root.glob('*.read'));first.with_suffix('.release').touch()
                    while not list(root.glob('*.compute')) and time.time()<end:time.sleep(.02)
                    while len(list(root.glob('*.read')))<2 and time.time()<end:time.sleep(.02)
                    self.assertEqual(len(list(root.glob('*.read'))),2)
                    self.assertTrue(list(root.glob('*.compute')))
                except BaseException as e:errors.append(e)
                finally:
                    for i in range(4):(root/f'{i}.release').touch()
                    (root/'done').touch()
            thread=threading.Thread(target=control);thread.start()
            budget=dict(workers=4,threads_per_worker=1,cpu_budget=4,read_concurrency=2)
            with patch.dict(os.environ,{'VISTA_READ_GATE':str(gate)}):
                run([(str(i),[sys.executable,'-B',str(script),str(root),str(i)]) for i in range(4)],4,root/'logs',lambda _:None,refresh=lambda:budget)
            thread.join();self.assertFalse(errors,errors)

    def test_memory_completed_evidence_tightens_and_rejects_stale_incompatible(self):
        from case_memory import corrected_estimate
        from server_data import write_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);key=dict(stage='prepare',geometry={'shape':[8,8,8],'spacing':[1,1,1],'target_spacing':[1,1,1]},implementation={'x':'a'})
            record=dict(compatibility=key,observed_at=time.time(),peak_rss_bytes=600,peak_scope='process lifetime high-water at case boundary',status='completed',output_bytes=20)
            write_json(root/'00_admin/case-observations/a.json',record)
            value,used=corrected_estimate(root,100,[key],now=time.time())
            self.assertEqual(value,660);self.assertEqual(len(used),1)
            for change in [dict(observed_at=0),dict(compatibility=dict(key,stage='other')),dict(compatibility=dict(key,implementation={'x':'b'})),dict(peak_scope='private guess'),dict(compatibility=dict(key,geometry={'shape':[9,9,9]})),dict(status='failed'),dict(observed_at=time.time()+3600)]:
                write_json(root/'00_admin/case-observations/a.json',dict(record,**change))
                self.assertEqual(corrected_estimate(root,100,[key],now=time.time())[0],100)

    def test_supervisor_reaps_preparation_when_gpu_leader_already_dead(self):
        from persistent_prediction import stop_workers
        dead=Mock();dead.poll.return_value=-9
        prep=Mock();prep.poll.return_value=None
        row=dict(process=dead,preparation=prep,stream=Mock(),prep_stream=Mock())
        with patch('persistent_prediction.stop_owned_process') as stop:
            stop_workers([row])
        self.assertEqual([call.args[0] for call in stop.call_args_list],[dead,prep])

    def test_prefetch_pressure_action_changes_queue_not_compute(self):
        from live_resources import training_policy,feedback_policy
        settings=training_policy(dict(cpu_budget=80,memory_budget_bytes=100*2**30),1,2**20)
        w=dict(steps=20,seconds=35,wait_fraction=.8,io_pressure={'avg10':40})
        decision=feedback_policy(settings,[w,w],settings)
        self.assertEqual(decision['action'],'limit_reads')
        self.assertEqual(decision['settings']['total_cpu_slots'],settings['total_cpu_slots'])
        self.assertLess(decision['settings']['rank_settings'][0]['prefetch'],settings['rank_settings'][0]['prefetch'])

    def test_reader_factory_releases_on_success_and_exception(self):
        from read_admission import ReaderFactory
        class Reader:
            def read_images(self,value):return value
            def read_seg(self,value):raise ValueError('bad input')
        with patch('read_admission.read_slot') as slot:
            rw=ReaderFactory(Reader)();self.assertEqual(rw.read_images('payload'),'payload')
            with self.assertRaises(ValueError):rw.read_seg('payload')
            self.assertEqual(slot.return_value.__exit__.call_count,2)

class OwnedWriters(unittest.TestCase):
    def test_actual_dead_gpu_leader_does_not_leave_preparation_writer(self):
        from persistent_prediction import stop_workers
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);script=root/'writer.py'
            script.write_text("import sys,time\nfrom pathlib import Path\np=Path(sys.argv[1])\nwhile True:\n with p.open('a') as f:f.write('x');f.flush()\n time.sleep(.02)\n")
            writer=subprocess.Popen([sys.executable,'-B',str(script),str(root/'ticks')],start_new_session=True)
            leader=subprocess.Popen([sys.executable,'-c','import os;os._exit(9)'],start_new_session=True)
            try:
                leader.wait(timeout=10);end=time.time()+10
                while not (root/'ticks').exists() and time.time()<end:time.sleep(.02)
                self.assertIsNone(writer.poll());self.assertTrue((root/'ticks').exists())
                stop_workers([dict(process=leader,preparation=writer,stream=Mock(),prep_stream=Mock())])
                self.assertIsNotNone(writer.poll())
                size=(root/'ticks').stat().st_size;time.sleep(.1);self.assertEqual((root/'ticks').stat().st_size,size)
            finally:
                if writer.poll() is None:writer.kill();writer.wait()
                if leader.poll() is None:leader.kill();leader.wait()

    def test_exited_or_recycled_leader_never_signals_old_group(self):
        import gpu_queue
        dead=Mock();dead.poll.return_value=-9
        with patch.object(gpu_queue.os,'killpg',create=True) as signal_group:gpu_queue.stop_owned_process(dead)
        signal_group.assert_not_called();dead.terminate.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith('linux'),'requires actual Linux /proc and flock')
    def test_preparation_inherits_lock_and_registered_identity_survives_gpu_death(self):
        import fcntl
        from task_worker import child_options,register_child
        from task_runtime import alive
        from server_data import read_json
        from persistent_prediction import stop_workers
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lock=(root/'workflow.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.dict(os.environ,{'WHOLE_HEART_LOCK_FD':str(lock.fileno()),'WHOLE_HEART_JOB':str(root)}):
                child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True,**child_options())
                leader=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True,**child_options())
                register_child(child,'predict_preparation')
            try:
                record=read_json(root/f'processes/{child.pid}.json');self.assertTrue(alive(record))
                self.assertFalse(alive(dict(record,identity={'boot':'wrong','start':'0'})))
                lock.close();leader.kill();leader.wait(timeout=10)
                with (root/'workflow.lock').open('a+b') as contender:
                    with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                stop_workers([dict(process=leader,preparation=child,stream=Mock(),prep_stream=Mock())])
                with (root/'workflow.lock').open('a+b') as contender:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertFalse(alive(record))
            finally:
                lock.close()
                for process in (leader,child):
                    if process.poll() is None:process.kill();process.wait()

class DispatchMemory(unittest.TestCase):
    def test_real_budget_consumer_corrects_headers_from_completed_case(self):
        import types
        from task_resources import case_budget
        from case_memory import compatibility
        from server_data import write_json
        image=types.SimpleNamespace(shape=(8,8,8),header=types.SimpleNamespace(get_zooms=lambda:(1.,1.,1.)))
        facts=dict(cpu_budget=4,memory_budget_bytes=5*2**30)
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'nibabel':types.SimpleNamespace(load=lambda _:image)}),patch('task_resources.probe',return_value=facts):
            before=case_budget(temp,['a','b','c','d'],[1,1,1])
            write_json(Path(temp)/'00_admin/case-observations/a.json',dict(status='completed',compatibility=compatibility(image,[1,1,1],'prepare'),observed_at=time.time(),peak_rss_bytes=3*2**30,peak_scope='process lifetime high-water at case boundary',output_bytes=200))
            after=case_budget(temp,['b','c','d'],[1,1,1])
            self.assertEqual(before['workers'],4);self.assertEqual(after['workers'],1)
            self.assertGreater(after['estimated_bytes_per_case'],before['estimated_bytes_per_case'])
            self.assertEqual(after['memory_feedback'][0]['output_bytes'],200)


    def test_completed_peak_changes_actual_later_dispatch_and_active_reservations(self):
        import cpu_case_queue as queue
        from case_memory import corrected_estimate
        from server_data import write_json
        from live_resources import case_policy
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);key={'stage':'prepare'};processes=[];finished=[];history=[]
            class Process:
                def __init__(self,*args,**kwargs):
                    self.pid=100+len(processes);self.number=len(processes);self.polls=0;processes.append(self)
                    history.append((self.number,list(finished)))
                def poll(self):
                    self.polls+=1
                    return 0 if (self.number==0 or self.polls>=3) else None
            def completed(name):
                finished.append(name)
                if name=='0':write_json(root/'00_admin/case-observations/0.json',dict(status='completed',compatibility=key,observed_at=time.time(),peak_rss_bytes=500,peak_scope='process lifetime high-water at case boundary'))
            def refresh():
                estimate,_=corrected_estimate(root,100,[key])
                return case_policy(dict(cpu_budget=4,memory_budget_bytes=1000),4,estimate,supported_threads=1)
            with patch.object(queue.subprocess,'Popen',Process),patch.object(queue.time,'sleep'):
                queue.run([(str(i),['fixture']) for i in range(8)],4,root/'logs',completed,refresh=refresh)
            self.assertEqual([n for n,_ in history[:4]],[0,1,2,3])
            self.assertEqual(set(history[4][1]),{'0','1','2','3'})
            self.assertEqual(len(finished),8)

class FeedbackActions(unittest.TestCase):
    def test_runtime_pressure_action_is_applied_and_values_reported(self):
        from training_feedback import RollingFeedback
        from live_resources import training_policy
        resource=training_policy(dict(cpu_budget=80,memory_budget_bytes=100*2**30),1,2**20)
        class Dist:
            def all_gather_object(self,rows,local):rows[0]=local
            def broadcast_object_list(self,p,src):pass
        clock=[0.]
        with tempfile.TemporaryDirectory() as temp,patch('training_feedback.apply_pools',return_value={}) as apply,patch.object(Path,'read_text',return_value='some avg10=40 avg60=20 avg300=5 total=1'):
            feedback=RollingFeedback(resource,0,1,temp,clock=lambda:clock[0])
            for _ in range(60):clock[0]+=2;feedback.observe({'batch_wait_seconds':1.5},Mock(current_epoch=0),Dist())
            self.assertEqual(apply.call_count,1)
            decision=feedback.events[-2]['decision'];self.assertEqual(decision['action'],'limit_reads')
            self.assertEqual(feedback.events[-1]['decision']['action'],'keep')
            self.assertEqual(decision['requested_settings'],decision['applied_settings'])
            self.assertEqual(feedback.resource['total_cpu_slots'],resource['total_cpu_slots'])

class PredictionFailureLog(unittest.TestCase):
    def test_temporary_control_failure_points_to_durable_worker_log(self):
        import types
        import persistent_prediction
        from server_data import write_json,digest
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'temporary-control';directory.mkdir();log=root/'durable'/'worker.log'
            checkpoint=root/'checkpoint';checkpoint.write_bytes(b'fixture')
            model=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),targets=['a'])
            frozen=root/'08_results/evaluation/frozen_models.json';write_json(frozen,dict(models=[model]))
            write_json(directory/'request.json',dict(id=1,threads=1,records=[dict(index=0,case_id='a')]))
            torch=types.SimpleNamespace(set_num_threads=lambda _:None,cuda=types.SimpleNamespace(empty_cache=lambda:None))
            modules={'torch':torch,'prepare_nnunet':types.SimpleNamespace(configure_paths=lambda:None),
                'evaluation':types.SimpleNamespace(make_predictor=Mock(side_effect=RuntimeError('injected model failure'))),
                'evaluation_phases':types.SimpleNamespace(preprocess_case=Mock(),infer=Mock())}
            with patch.dict(os.environ,{'WHOLE_HEART_WORKSPACE':str(root),'VISTA_PREDICTION_LOG':str(log)}),patch.dict(sys.modules,modules),patch('runtime_observations.failure') as failure:
                with self.assertRaisesRegex(RuntimeError,'injected'):persistent_prediction.worker(directory,digest(frozen))
            self.assertEqual(failure.call_args.kwargs['log'],str(log))

if __name__=='__main__':unittest.main()
