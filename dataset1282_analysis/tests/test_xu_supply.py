from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dataset1282_analysis import xu_training
from dataset1282_analysis.xu_supply import BatchSupply
from test_xu_acceleration_live import environment


class SupplyTests(unittest.TestCase):
    def test_order_and_actual_missing_file_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(2):
                np.save(root/f'{i}.npy',np.full((4,)*3,i,dtype='float32'))
            def load(batch):
                return np.load(root/f'{batch[0]}.npy',allow_pickle=False)
            with environment(XU_PREFETCH=1,XU_PREFETCH_WORKERS=2,XU_PREFETCH_BYTES=1024**3):
                supply = BatchSupply([0,1,2],1,4,load,lambda _:64)
                with supply.batches() as batches:
                    self.assertEqual(next(batches)[1].mean(),0)
                    self.assertEqual(next(batches)[1].mean(),1)
                    with self.assertRaises(FileNotFoundError):
                        next(batches)
        self.assertFalse(any(t.name.startswith('xu-cpu') for t in threading.enumerate()))

    def test_insufficient_configured_memory_fails_before_read(self):
        with environment(XU_PREFETCH=1,XU_PREFETCH_BYTES=1):
            supply = BatchSupply(['missing.npy'],1,4,lambda b:np.load(b[0]),lambda _:64)
            with supply.batches() as batches:
                with self.assertRaises(MemoryError):
                    list(batches)
        self.assertEqual(supply.stats['prefetch_batches'],0)

    def test_nonfinite_gradients_prevent_adam_mutation(self):
        model = torch.nn.Conv3d(1,11,1)
        model.spec = SimpleNamespace(side=4,batch=2)
        before = model.weight.detach().clone()
        model.bias.register_hook(lambda g:torch.full_like(g,float('inf')))
        opt = torch.optim.Adam(model.parameters())
        load = lambda name,branch:(np.zeros((4,)*3,'f4'),np.ones((4,)*3,'i1'))
        with environment(XU_PREFETCH=0):
            with self.assertRaisesRegex(FloatingPointError,'gradient'):
                xu_training._epoch(model,['a'],{'a':[]},load,'crop64','cpu',torch.Generator(),opt)
        self.assertTrue(torch.equal(before,model.weight))
        self.assertFalse(opt.state)
