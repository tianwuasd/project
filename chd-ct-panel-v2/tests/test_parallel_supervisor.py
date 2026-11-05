import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import psutil
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chd_stunet.training.supervisor import launch,close_owned,learn_workers,restore_full_epoch
from chd_stunet.training.engine import initial_state
from chd_stunet.training.checkpoints import save_checkpoint


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';root.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=root);self.root=Path(self.temporary.name)
        self.prepared=self.root/'prepared';self.prepared.mkdir()
        self.output=self.root/'training'
    def tearDown(self):self.temporary.cleanup()

    def test_check_only_does_not_allocate_gpu_or_write_output(self):
        settings=dict(batch_size=4)
        with patch('chd_stunet.training.supervisor.prepare_contract',return_value=(settings,dict(train_ids=['t'],val_ids=['v']),{})), \
             patch('chd_stunet.training.supervisor.query_visible',side_effect=AssertionError('GPU scan forbidden')):
            result=launch(self.prepared,self.output,check_only=True)
        self.assertFalse(self.output.exists())
        self.assertFalse(result['gpu_capacity_checked'])
        self.assertEqual(result['layouts'],[[4],[2,2],[2,1,1],[1,1,1,1]])

    def test_incompatible_resume_leaves_every_old_byte_unchanged(self):
        self.output.mkdir();(self.output/'signature.json').write_text('{"identity":"old"}')
        (self.output/'status.json').write_text('unchanged')
        before={p.name:p.read_bytes() for p in self.output.iterdir()}
        with patch('chd_stunet.training.supervisor.prepare_contract',return_value=({'batch_size':4},{},{'identity':'new'})):
            with self.assertRaisesRegex(ValueError,'Resume contract differs'):
                launch(self.prepared,self.output,resume=True)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_completed_metadata_repair_needs_no_gpu_or_author_weights(self):
        self.output.mkdir();signature={'identity':'completed'}
        (self.output/'signature.json').write_text(json.dumps(signature))
        settings=dict(batch_size=1,epochs=2,scope='smoke',train_steps_per_epoch=4,validation_steps_per_epoch=1)
        state=initial_state();state.update(epoch=2,attempts=8,updates=8,history=[{},{}],
                                         updates_by_cohort=dict(main=4,auxiliary=4))
        model=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        save_checkpoint(self.output/'checkpoints',model,optimizer,torch.amp.GradScaler('cuda',enabled=False),
                        state,signature,torch.device('cpu'))
        dataset=dict(train_ids=['t'],val_ids=['v'])
        with patch('chd_stunet.training.supervisor.prepare_contract',return_value=(settings,dataset,signature)), \
             patch('chd_stunet.training.supervisor.freeze_code'), \
             patch('chd_stunet.training.supervisor.query_visible',side_effect=AssertionError('GPU scan forbidden')):
            result=launch(self.prepared,self.output,resume=True)
        self.assertEqual(result['status'],'completed')
        self.assertTrue((self.output/'checkpoints/final.json').exists())
        status=json.loads((self.output/'status.json').read_text(encoding='utf-8'))
        self.assertEqual((status['state'],status['epoch'],status['attempts']),('completed',2,8))

    def test_cleanup_terminates_owned_child_and_refuses_reused_external_identity(self):
        command=[sys.executable,'-c','import time; time.sleep(30)']
        owned_process=subprocess.Popen(command);unrelated=subprocess.Popen(command)
        try:
            identities={owned_process.pid:psutil.Process(owned_process.pid).create_time(),
                        unrelated.pid:psutil.Process(unrelated.pid).create_time()-1}
            close_owned([owned_process],identities)
            self.assertIsNotNone(owned_process.poll());self.assertIsNone(unrelated.poll())
            path=self.root/'worker.json'
            row=dict(pid=unrelated.pid,born=psutil.Process(unrelated.pid).create_time(),nonce='other',parent_pid=999)
            path=self.root/f'worker-{unrelated.pid}.json';path.write_text(json.dumps(row))
            known={};learn_workers(self.root,'this',{999},known);self.assertFalse(known)
        finally:
            for process in (owned_process,unrelated):
                if process.poll() is None:process.terminate()
                process.wait(timeout=10)

    def test_pressure_retains_complete_epoch_after_multiple_partial_pauses(self):
        from chd_stunet.observability.runlog import write_json
        from chd_stunet.training.engine import prune_checkpoints
        from chd_stunet.training.checkpoints import read_checkpoint
        folder=self.root/'checkpoints';signature={'identity':'boundary'}
        model=torch.nn.Linear(1,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        state=dict(epoch=2,phase='train',step=0,attempts=8)
        boundary=save_checkpoint(folder,model,optimizer,scaler,state,signature,torch.device('cpu'))
        write_json(folder/'epoch.json',boundary)
        for step in (1,2,3):
            state.update(step=step,attempts=8+step)
            save_checkpoint(folder,model,optimizer,scaler,state,signature,torch.device('cpu'))
            prune_checkpoints(folder)
        self.assertTrue((folder/boundary['file']).exists())
        restored=restore_full_epoch(folder,signature)
        self.assertEqual((restored['epoch'],restored['step'],restored['attempts']),(2,0,8))
        self.assertEqual(read_checkpoint(folder,signature)['state'],restored)


if __name__=='__main__':unittest.main()
