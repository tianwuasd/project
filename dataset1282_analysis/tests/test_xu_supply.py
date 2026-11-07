import copy
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset1282_analysis import xu_init, xu_training
from dataset1282_analysis.xu_supply import BatchSupply


ENV = dict(XU_PREFETCH='1', XU_PREFETCH_BYTES=str(128*1024**2))


class TinyNet(torch.nn.Conv3d):
    def __init__(self, classes):
        super().__init__(1, classes, 1)
        self.spec = SimpleNamespace(side=4, batch=2)


class SupplyTests(unittest.TestCase):
    def setUp(self):
        probe = patch('dataset1282_analysis.xu_supply.snapshot',
                      return_value={'available_memory_bytes': 2*1024**3})
        self.probe = probe.start()
        self.addCleanup(probe.stop)

    def test_one_slot_order_worker_failure_and_budget(self):
        entered = threading.Event()
        threads = []

        def load(batch):
            n = batch[0]
            threads.append((n, threading.current_thread().name))
            if n == 1:
                entered.set()
            if n == 2:
                raise ValueError('bad input')
            return n

        with patch.dict(os.environ, ENV):
            supply = BatchSupply([0,1,2], 1, 4, load, lambda name:64)
            with supply.batches() as batches:
                self.assertEqual(next(batches), ([0],0))
                self.assertTrue(entered.wait(3))
                self.assertEqual([n for n,_ in threads], [0,1])
                self.assertEqual(next(batches), ([1],1))
                with self.assertRaisesRegex(ValueError, 'bad input'):
                    next(batches)
            self.assertTrue(supply.stats['applied'])
        self.assertEqual(threads[0][1], threading.current_thread().name)
        self.assertTrue(threads[1][1].startswith('xu-cpu'))
        self.assertFalse(any(t.name.startswith('xu-cpu') for t in threading.enumerate()))
        for cap, bound in [('1', lambda _:64), ('999999999', None), ('bad-input', lambda _:64)]:
            threads.clear()
            with patch.dict(os.environ, dict(ENV, XU_PREFETCH_BYTES=cap)):
                supply = BatchSupply([0,1], 1, 4, load, bound)
                with supply.batches() as batches:
                    self.assertEqual(list(batches), [([0],0),([1],1)])
            self.assertFalse(supply.stats['applied'])
            self.assertTrue(all(thread == threading.current_thread().name for _,thread in threads))
        self.probe.return_value = {'available_memory_bytes': None}
        with patch.dict(os.environ, ENV):
            supply = BatchSupply([0,1], 1, 4, load, lambda _:64)
            with supply.batches() as batches:
                list(batches)
        self.assertEqual(supply.stats['memory_denied_batches'],1)
        self.assertFalse(supply.stats['applied'])

    def test_equal_updates_for_both_types_of_branch(self):
        ids = ['a','b','c','d','e']
        raw = np.arange(64).reshape((4,4,4))
        images = {name: (raw + i).astype('float32')/32 for i,name in enumerate(ids)}
        def load(name, branch):
            return images[name], (raw % 11).astype('int8')
        load.source_voxels = lambda _:64
        cache = SimpleNamespace(read_image=lambda name,shape,box:images[name],
                                geometry=lambda name:SimpleNamespace(cache_shape=raw.shape))
        targets = SimpleNamespace(read=lambda *args:(raw%7).astype('int8'))
        book = dict(boxes={name:[] for name in ids})
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,old_threads)
        for is_init in [False, True]:
            net = TinyNet(7 if is_init else 11)
            results = []
            for enabled in ['0','1']:
                model = copy.deepcopy(net)
                opt = torch.optim.Adam(model.parameters(),lr=2e-4)
                rng = torch.Generator(device='cpu').manual_seed(91)
                stats = {}
                with patch.dict(os.environ,dict(ENV,XU_PREFETCH=enabled)):
                    if is_init:
                        loss = xu_init._epoch(model,ids,cache,targets,book,'cpu',rng,opt,stats)
                    else:
                        loss = xu_training._epoch(model,ids,{n:[] for n in ids},load,'crop64','cpu',rng,opt,stats)
                self.assertEqual(stats['prefetch_batches'], 2 if enabled=='1' else 0)
                self.assertGreater(stats['cpu_prepare_seconds'],0)
                self.assertGreaterEqual(stats['transfer_call_seconds'],0)
                results.append((loss,rng.get_state(),model.weight.detach(),opt.state[model.weight]['exp_avg']))
            self.assertEqual(results[0][0],results[1][0])
            for a,b in zip(results[0][1:],results[1][1:]):
                self.assertTrue(torch.equal(a,b))

    def test_nonfinite_gradients_prevent_adam_mutation(self):
        model = TinyNet(11)
        before = model.weight.detach().clone()
        model.bias.register_hook(lambda g: torch.full_like(g,float('inf')))
        opt = torch.optim.Adam(model.parameters())
        load = lambda name,branch:(np.zeros((4,4,4),'f4'),np.ones((4,4,4),'i1'))
        with self.assertRaisesRegex(FloatingPointError,'gradient'):
            xu_training._epoch(model,['a'],{'a':[]},load,'crop64','cpu',torch.Generator(),opt)
        self.assertTrue(torch.equal(before,model.weight))
        self.assertFalse(opt.state)

    def test_only_known_prior_stage_sources_are_accepted(self):
        current = dict(seed=3, source={'xu_supply.py':'new','xu_training.py':'changed','xu_models.py':'fixed'})
        prior = dict(seed=3, source={'xu_training.py':'2ed34c4688840f93c08992c9c623f4b88fb2b674ec0a259e12e30b3f076a38ed',
                                     'xu_models.py':'fixed'})
        self.assertTrue(xu_training._compatible_contract(prior,current))
        self.assertFalse(xu_training._compatible_contract(dict(prior,seed=4),current))
        self.assertFalse(xu_training._compatible_contract(dict(prior,source={}),current))
