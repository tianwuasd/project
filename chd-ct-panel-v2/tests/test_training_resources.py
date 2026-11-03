from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class TrainingResourceTests(unittest.TestCase):
    def test_resume_probe_does_not_allocate_existing_momentum_twice(self):
        from chd_stunet.training.resources import capacity_probe
        model=torch.nn.Linear(2,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.99)
        model(torch.ones(1,2)).square().mean().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        batch=dict(data=np.ones((1,2),dtype=np.float32),target=[np.zeros((1,1),dtype=np.int64)],
                   available=np.ones((1,7),dtype=bool))
        settings={'amp':False}
        criterion=lambda output,target,available:output.square().mean()
        with patch('chd_stunet.training.resources.torch.zeros_like',wraps=torch.zeros_like) as allocations:
            result=capacity_probe(model,criterion,batch,settings,torch.device('cpu'),{},optimizer)
        self.assertEqual(result['existing_momentum_buffers'],2)
        self.assertEqual(result['simulated_momentum_buffers'],0)
        allocations.assert_not_called()

    def test_unsupported_driver_text_is_not_an_empty_gpu(self):
        from chd_stunet.training.resources import parse_compute_processes
        uuid='GPU-00000000-1111-2222-3333-444444444444'
        for text in ('[Not Supported]','N/A, '+uuid,'12, N/A','No running processes found'):
            with self.subTest(text=text),self.assertRaises(RuntimeError):parse_compute_processes(text,uuid,10)
        self.assertEqual(parse_compute_processes('',uuid,10),[])
        self.assertEqual(parse_compute_processes('10, '+uuid+'\n12, '+uuid,uuid,10),[12])
        self.assertEqual(parse_compute_processes('12, '+uuid,uuid.removeprefix('GPU-'),10),[12])


if __name__=='__main__':unittest.main()
