from pathlib import Path
import copy
import tempfile
import sys
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class CheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_optimizer_and_rng(self):
        from chd_stunet.training.checkpoints import save_checkpoint,read_checkpoint,restore_checkpoint
        root=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            torch.manual_seed(91)
            model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.2),torch.nn.Linear(5,1))
            optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.99,nesterov=True)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            def step(m,o):
                x=torch.randn(4,3);o.zero_grad(set_to_none=True)
                m(x).square().mean().backward();o.step();return x
            step(model,optimizer)
            state=dict(attempts=1,epoch=0,phase='train',step=1)
            save_checkpoint(folder,model,optimizer,scaler,state,{'identity':'same'},torch.device('cpu'))
            expected_x=step(model,optimizer)
            expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(optimizer.state_dict())
            restored=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.2),torch.nn.Linear(5,1))
            opt=torch.optim.SGD(restored.parameters(),lr=.01,momentum=.99,nesterov=True)
            payload=read_checkpoint(folder,{'identity':'same'})
            self.assertEqual(restore_checkpoint(payload,restored,opt,scaler,torch.device('cpu')),state)
            actual_x=step(restored,opt);torch.testing.assert_close(actual_x,expected_x,rtol=0,atol=0)
            for name,value in expected.items():torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
            for key,values in expected_opt['state'].items():
                torch.testing.assert_close(values['momentum_buffer'],opt.state_dict()['state'][key]['momentum_buffer'],rtol=0,atol=0)

    def test_failed_pointer_commit_keeps_old_generation_readable(self):
        from chd_stunet.training.checkpoints import save_checkpoint,read_checkpoint
        root=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            model=torch.nn.Linear(2,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            state=dict(attempts=0,epoch=0,phase='train',step=0);signature={'identity':'same'}
            save_checkpoint(folder,model,optimizer,scaler,state,signature,torch.device('cpu'))
            before=(Path(folder)/'latest.json').read_bytes()
            with patch('chd_stunet.training.checkpoints.write_json',side_effect=OSError('simulated crash')):
                with self.assertRaises(OSError):save_checkpoint(folder,model,optimizer,scaler,state,signature,torch.device('cpu'))
            self.assertEqual((Path(folder)/'latest.json').read_bytes(),before)
            self.assertEqual(read_checkpoint(folder,signature)['state'],state)


if __name__=='__main__':unittest.main()
