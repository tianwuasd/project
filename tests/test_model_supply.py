import multiprocessing as mp
import unittest

from chd_segmentation.models import supply


def future_batch(stream,finished):
    with stream.build(0,1,0):
        finished.set()


class SupplyTests(unittest.TestCase):
    def test_current_batch_finishes_before_next_build(self):
        context = mp.get_context('spawn')
        ready = supply.CpuReady(context,1,None)
        stream = supply.OrderedSupply(context,[1],1,ready,None)
        stream.start(0,'training',[1])
        with stream.build(0,0,0):
            self.assertEqual(stream.snapshot()['active'],1)
        self.assertEqual(stream.snapshot()['ready_depth'],[1])
        stream.finish(0,'training')
        self.assertEqual(stream.snapshot()['batch'],1)
        with stream.build(0,1,0):
            pass
        stream.finish(0,'training')
        self.assertEqual(stream.snapshot()['active'],0)

    def test_duplicate_position_fails(self):
        context = mp.get_context('spawn')
        ready = supply.CpuReady(context,1,None)
        stream = supply.OrderedSupply(context,[1],1,ready,None)
        stream.start(0,'training',[1])
        with stream.build(0,0,0):
            pass
        with self.assertRaises(ValueError):
            with stream.build(0,0,0):
                pass

    def test_case_workspace_released_after_build(self):
        context = mp.get_context('spawn')
        ready = supply.CpuReady(context,1,None)
        stream = supply.OrderedSupply(context,[1],1,ready,None,capacity_bytes=16,
            estimates=[dict(case_id='sample',workspace_bytes=16)])
        stream.start(0,'read',[1])
        with stream.build(0,0,0):
            stream.reserve(0,'sample')
            self.assertEqual(stream.snapshot()['active_workspace_bytes'],16)
        self.assertEqual(stream.snapshot()['active_workspace_bytes'],0)

    def test_future_request_does_not_occupy_current_reader(self):
        context = mp.get_context('spawn')
        ready = supply.CpuReady(context,1,None)
        stream = supply.OrderedSupply(context,[1],1,ready,None)
        stream.start(0,'training',[1])
        completed = context.Event()
        child = context.Process(target=future_batch,args=(stream,completed))
        child.start()
        try:
            with stream.condition:
                pending = stream.condition.wait_for(lambda: stream.waiting[0]==1,timeout=15)
            self.assertTrue(pending)
            self.assertFalse(completed.is_set())
            with stream.build(0,0,0):
                self.assertEqual(stream.snapshot()['active'],1)
                self.assertFalse(completed.is_set())
            stream.finish(0,'training')
            child.join(15)
            self.assertEqual(child.exitcode,0)
            self.assertTrue(completed.is_set())
            stream.finish(0,'training')
        finally:
            if child.is_alive():
                child.terminate()
            child.join()


if __name__=='__main__':
    unittest.main()
