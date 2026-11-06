"""Run the real supervisor loop with fake owned ranks and GPU inventories."""
import itertools
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chd_stunet.training import supervisor
from chd_stunet.training.engine import initial_state
from chd_stunet.training.checkpoints import save_checkpoint
from chd_stunet.observability.runlog import write_json


class SupervisorRelocationTests(unittest.TestCase):
    def exercise(self,mode):
        scratch=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root=Path(temporary);prepared=root/'prepared';prepared.mkdir();output=root/'run'
            settings=dict(batch_size=4,epochs=1,train_steps_per_epoch=2,validation_steps_per_epoch=1,scope='formal',architecture={})
            dataset=dict(rows={},train_ids=[],val_ids=[],identity={'adapters':[]});signature={'identity':'test-supervisor'}
            initial=[dict(uuid=u,ordinal=i,usable_bytes=100,external_pids=[777] if mode=='no_growth' and i>=2 else [])
                     for i,u in enumerate('abcde' if mode in ('replace','no_growth') else 'abcd')]
            busy=[dict(row,external_pids=[999] if mode=='none' or row['uuid']=='b' else []) for row in initial]
            groups={};jobs=[];model=torch.nn.Linear(1,1);opt=torch.optim.SGD(model.parameters(),lr=.1)
            scaler=torch.amp.GradScaler('cuda',enabled=False)

            class FakeRank:
                def __init__(self,command,**kwargs):
                    job=json.loads(Path(command[command.index('--job')+1]).read_text(encoding='utf-8'))
                    self.rank=int(command[-1]);self.job=job;self.pid=900000+len(jobs)*10+self.rank
                    if job['generation'] not in groups:
                        groups[job['generation']]={'done':False,'number':len(groups)};jobs.append(job)
                    self.group=groups[job['generation']]
                @property
                def returncode(self):return 0 if self.group['done'] else None
                def poll(self):
                    handoff=output/'control/handoff.json';stop=output/'control/stop.json'
                    if mode in ('admission','bad_data') and not self.group['number'] and not self.group['done']:
                        write_json(output/'control'/f'fatal-{self.job["generation"]}-0.json',dict(rank=0,uuid='a',
                            error='bad data' if mode=='bad_data' else 'GPU taken',error_type='ValueError' if mode=='bad_data' else 'GPUUnavailable',
                            resource_unavailable=mode=='admission'))
                        self.group['done']=True
                    action='completed' if self.group['number'] else 'relocate' if handoff.exists() else 'stopped' if stop.exists() else None
                    if action and not self.group['done']:
                        state=initial_state()
                        if action=='completed':state.update(epoch=1,attempts=2,updates=2,history=[{}],updates_by_cohort=dict(main=8,auxiliary=0))
                        else:state.update(step=1,attempts=1,updates=1,train_batches=1,updates_by_cohort=dict(main=4,auxiliary=0))
                        save_checkpoint(output/'checkpoints',model,opt,scaler,state,signature,torch.device('cpu'),
                            distributed=dict(members=self.job['members'],rng_by_rank=[{}]*len(self.job['members'])))
                        write_json(output/'control/result.json',dict(action=action,generation=self.job['generation']))
                        write_json(output/'summary.json',dict(status='relocating' if action=='relocate' else action,attempts=state['attempts']))
                        if mode=='user_during_handoff':signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                        self.group['done']=True
                    return self.returncode

            def live(_):
                if mode=='user':signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                return busy
            from chd_stunet.training.parallel_policy import layout
            def supply(*args):
                world=args[3]
                return dict(stage={'reserve_bytes':1},workers=[0]*world,layout=layout(4,world),applied_total_workers=0)
            ticks=itertools.count(0,10)
            with patch.object(supervisor,'prepare_contract',return_value=(settings,dataset,signature)), \
                 patch.object(supervisor,'freeze_code',return_value=output/'code/run.py'), \
                 patch.object(supervisor,'PatchSampler'),patch.object(supervisor,'snapshot',return_value={}), \
                 patch.object(supervisor,'supply_plan',side_effect=supply), \
                 patch.object(supervisor,'query_visible',side_effect=lambda *_:initial if not jobs else busy), \
                 patch.object(supervisor,'query_live',side_effect=live), \
                 patch.object(supervisor,'remember_children'),patch.object(supervisor,'learn_workers'), \
                 patch.object(supervisor,'close_owned'), \
                 patch.object(supervisor.subprocess,'Popen',side_effect=FakeRank), \
                 patch.object(supervisor.time,'monotonic',side_effect=lambda:next(ticks)), \
                 patch.object(supervisor.time,'sleep'),patch.object(supervisor.sys,'platform','linux'), \
                 patch.object(supervisor.torch.distributed,'is_nccl_available',return_value=True):
                result=supervisor.launch(prepared,output,gpus='auto',elastic=False,wait_seconds=0,weights='unused')
            self.saved_summary=json.loads((output/'summary.json').read_text())
            self.saved_status=json.loads((output/'status.json').read_text())
            return result,jobs

    def test_replace_even_when_opportunistic_elastic_growth_is_disabled(self):
        result,jobs=self.exercise('replace')
        self.assertEqual(result['status'],'completed');self.assertEqual(len(jobs),2)
        self.assertEqual(jobs[0]['members'],list('abcd'));self.assertEqual(jobs[1]['members'],list('acde'))

    def test_shrink_without_resetting_the_partial_cursor(self):
        result,jobs=self.exercise('shrink')
        self.assertEqual(result['status'],'completed');self.assertEqual(jobs[1]['members'],list('acd'))
        self.assertEqual(jobs[1]['supply']['layout'],[2,1,1])

    def test_zero_available_cards_stops_with_checkpoint_and_reason(self):
        result,jobs=self.exercise('none')
        self.assertEqual(result['status'],'stopped');self.assertEqual(result['reason'],'no_available_gpu')
        self.assertEqual(result['attempts'],1);self.assertEqual(len(jobs),1)

    def test_explicit_user_stop_does_not_auto_restart(self):
        result,jobs=self.exercise('user')
        self.assertEqual(result['status'],'stopped');self.assertEqual(len(jobs),1)

    def test_no_elastic_replaces_without_opportunistic_growth(self):
        result,jobs=self.exercise('no_growth')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(jobs[0]['members'],list('ab'));self.assertEqual(jobs[1]['members'],list('ac'))

    def test_admission_race_retries_without_a_checkpoint(self):
        result,jobs=self.exercise('admission')
        self.assertEqual(result['status'],'completed');self.assertEqual(jobs[1]['members'],list('cd'))

    def test_data_failure_remains_visible_even_with_other_gpu_contention(self):
        with self.assertRaisesRegex(RuntimeError,'bad data'):self.exercise('bad_data')

    def test_user_stop_while_handoff_is_committing_finalizes_saved_progress(self):
        result,jobs=self.exercise('user_during_handoff')
        self.assertEqual(result['status'],'stopped');self.assertEqual(result['attempts'],1)
        self.assertEqual(self.saved_summary['status'],'stopped');self.assertEqual(self.saved_status['state'],'stopped')
        self.assertEqual(len(jobs),1)


if __name__=='__main__':unittest.main()
