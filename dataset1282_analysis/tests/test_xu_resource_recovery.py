import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dataset1282_analysis.xu_supply import BatchSupply
from test_xu_acceleration_live import environment


class SupplyRecoveryTests(unittest.TestCase):
    def test_ready_batch_is_consumed_before_next_batch_admission(self):
        with environment(XU_PREFETCH=1,XU_PREFETCH_WORKERS=1,XU_PREFETCH_BYTES=70*1024**2):
            supply = BatchSupply([1,2],1,4,lambda b:np.full((4,)*3,b[0],dtype='float32'),
                                 lambda name:64 if name==1 else 1024**3)
            with supply.batches() as batches:
                self.assertEqual(next(batches)[1].mean(),1)
                with self.assertRaises(MemoryError):
                    next(batches)

    def test_single_batch_budget_completes_every_ordered_real_array(self):
        from dataset1282_analysis.xu_supply_budget import batch_bytes
        side = 32
        estimate = batch_bytes(side**3,1,side)
        with environment(XU_PREFETCH=1,XU_PREFETCH_WORKERS=4,XU_PREFETCH_BYTES=estimate):
            supply = BatchSupply(list(range(12)),1,side,
                                 lambda b:np.full((side,)*3,b[0],dtype='float32'),lambda _:side**3)
            with supply.batches() as batches:
                values = [float(value.mean()) for _,value in batches]
            self.assertEqual(values,list(range(12)))
            self.assertLessEqual(supply.stats['reserved_peak_bytes'],estimate)
            self.assertGreater(supply.stats['memory_denied_batches'],0)

    def test_planner_admits_large_source_within_fair_share(self):
        from dataset1282_analysis.xu_supply_budget import prefetch_budget
        gib = 1024**3
        self.assertEqual(prefetch_budget(40*gib,4*gib,20*gib,16*gib),20*gib)
        from dataset1282_analysis.xu_resources import ResourceUnavailable
        with self.assertRaises(ResourceUnavailable):
            prefetch_budget(20*gib,4*gib,20*gib,16*gib)

    def test_actual_worker_memory_refusal_is_recovered_without_changing_batch(self):
        from dataset1282_analysis.xu_pipeline import _execute
        from dataset1282_analysis.xu_dispatch import Children
        from dataset1282_analysis.xu_reallocation import run_with_reallocation
        from dataset1282_analysis.observability.status import operation_lock
        from dataset1282_analysis.contracts import read_json
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = SimpleNamespace(resume=False,max_reallocations=1,resource_recovery=True,
                                   resource_poll_seconds=.05,prefetch_workers=4)
            with operation_lock(out,'resource-recovery-test',out) as owner:
                def operation(current):
                    children = Children(out,owner)
                    budget = 256*1024**2 if getattr(current,'force_serial',False) else 1
                    overlay = dict(XU_PREFETCH='1',XU_PREFETCH_BYTES=str(budget),XU_PREFETCH_WORKERS='1')
                    opts = SimpleNamespace(threads=1,children=children,overlay=overlay,device='cpu',assigned_devices=['cpu'])
                    try:
                        _execute('../tests/fixtures/xu_supply_child.py',[out/'complete.json'],
                                 'supply',out,owner,opts)
                    finally:
                        children.stop()
                run_with_reallocation(operation,args,out)
            self.assertEqual(read_json(out/'complete.json')['values'],[0.,1.,2.,3.])
            recovery = read_json(out/'resources/recovery.json')
            self.assertTrue(recovery['force_serial'])
            self.assertLess(recovery['prefetch_workers'],4)


if __name__=='__main__':
    unittest.main()
