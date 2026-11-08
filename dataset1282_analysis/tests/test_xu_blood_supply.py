import copy
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dataset1282_analysis import xu_blood
from dataset1282_analysis.xu_blood_grid import AxialViews
from dataset1282_analysis.xu_blood_models import SliceUNet, BloodSequence
from dataset1282_analysis.xu_blood_supply import SliceSupply, prepare_slices
from dataset1282_analysis.xu_fleet import work_limits, distribute
from dataset1282_analysis.xu_supply_budget import blood_batch_bytes, stage_minimum
from dataset1282_analysis.xu_targets import TargetStore, build_targets
from test_xu_acceleration_live import environment
from test_xu_targets import on_disk_phantom


class BloodSupplyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = on_disk_phantom(self.root/'prepared')
        self.target_dir = self.root/'targets'
        build_targets(self.cache, SimpleNamespace(width_mm=1., reviews=None, resume=False), self.target_dir)
        self.targets = TargetStore(self.target_dir, self.cache)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        # CPU 数值对照明确使用原生卷积后端；不依赖平台 oneDNN 内核。
        backend = torch.backends.mkldnn.enabled
        torch.backends.mkldnn.enabled = False
        self.addCleanup(setattr,torch.backends.mkldnn,'enabled',backend)

    def views(self):
        return AxialViews(self.cache, 'train_a', self.targets)

    def test_compressed_slice_order_values_partial_batch_and_unknown_labels(self):
        centres = np.random.default_rng(19).permutation(19)
        for temporal, batch in ((False,4),(True,1)):
            reference = self.views()
            with environment(XU_PREFETCH=1, XU_PREFETCH_BYTES=1024**3, XU_PREFETCH_WORKERS=2):
                supply = SliceSupply(self.views(),centres,batch,temporal)
                with supply.batches() as stream:
                    consumed = []
                    for ids, values in stream:
                        expected = prepare_slices(reference,ids,temporal)
                        for actual, baseline in zip(values,expected):
                            torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
                        consumed.extend(ids)
                self.assertEqual(consumed,list(centres))
                self.assertTrue(supply.stats['applied'])
                self.assertGreater(supply.stats['workers_applied'],0)
            self.assertTrue((reference.label(0)==-1).all())
        self.assertFalse(any(t.name.startswith('xu-cpu') for t in threading.enumerate()))

    def test_one_batch_memory_and_no_worker_both_finish_all_batches(self):
        budget = blood_batch_bytes(20**3,4,False)
        for workers in (0,2):
            with environment(XU_PREFETCH=1, XU_PREFETCH_BYTES=budget,
                             XU_PREFETCH_WORKERS=workers,XU_PREFETCH_DEPTH=8):
                supply = SliceSupply(self.views(),range(19),4,False)
                with supply.batches() as stream:
                    self.assertEqual([z for ids,_ in stream for z in ids],list(range(19)))
                self.assertEqual(supply.stats['workers_applied']==0,workers==0)
                if workers:
                    self.assertGreater(supply.stats['memory_denied_batches'],0)

    def test_missing_actual_target_propagates_and_releases_workers(self):
        views = self.views()
        path = self.target_dir/'cases/train_a/blood.b2nd'
        path.unlink()
        with environment(XU_PREFETCH=1,XU_PREFETCH_BYTES=1024**3,XU_PREFETCH_WORKERS=2):
            with self.assertRaisesRegex(ValueError,'Missing derived'):
                with SliceSupply(views,[2,3,4],1,False).batches() as stream:
                    list(stream)
        self.assertFalse(any(t.name.startswith('xu-cpu') for t in threading.enumerate()))

    def test_blood_single_card_and_2d_memory_budget(self):
        args = SimpleNamespace(profile='paper',prefetch=True,max_source_voxels=20**3)
        cap,_ = work_limits('blood',args,self.cache,6)
        self.assertEqual(cap,1)
        groups = distribute(['blood'],{'blood':cap},{f'cuda:{i}':100 for i in range(2,8)},lambda n,w:[10]*w)
        self.assertEqual(len(groups['blood']),1)
        self.assertEqual(stage_minimum(args,'blood'),
                         max(blood_batch_bytes(20**3,4,False),blood_batch_bytes(20**3,1,True)))

    def test_real_network_prefetch_keeps_loss_weights_and_adam(self):
        torch.manual_seed(11)
        initial = SliceUNet('smoke')
        results = []
        for workers in (0,2):
            model = copy.deepcopy(initial)
            temporal = BloodSequence(model.width)
            optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
            timing = {}
            with environment(XU_PREFETCH=int(workers>0), XU_PREFETCH_BYTES=1024**3,
                             XU_PREFETCH_WORKERS=workers):
                loss = xu_blood._epoch(self.cache,self.targets,['train_a'],model,temporal,'cpu',
                                      'encoder','paper',31,optimizer,epoch=1,total_epochs=12,telemetry=timing)
            results.append((loss,copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict())))
            self.assertGreater(timing['seconds'],0)
            self.assertGreater(timing['cpu_prepare_seconds'],0)
            self.assertEqual(timing['phase_eta_seconds'],0)
        self.assertEqual(results[0][0],results[1][0])
        for key,value in results[0][1].items():
            torch.testing.assert_close(value,results[1][1][key],rtol=0,atol=0)
        for p,state in results[0][2]['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(value,results[1][2]['state'][p][key],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
