from pathlib import Path
import os
import random
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class NumberLoader:
    rows={}
    def __call__(self,seed):
        from chd_stunet.acceleration.training_cpu import deterministic_cpu_rng
        with deterministic_cpu_rng(seed):
            return dict(values=np.concatenate([np.random.rand(3),torch.rand(3).numpy(),[random.random()]]))


class QuietLog:
    def event(self,*args,**kwargs):pass


class SupplyTests(unittest.TestCase):
    def test_serial_mode_does_not_bypass_zero_cpu_or_ram_budget(self):
        from chd_stunet.acceleration.training_cpu import BatchPool
        from chd_stunet.training.contracts import recipe
        root=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            with patch('chd_stunet.acceleration.training_cpu.policy',return_value=dict(
                    workers=0,cpu_budget=0,memory_budget_bytes=0,reserve_bytes=4*1024**3)):
                with self.assertRaisesRegex(MemoryError,'Serial supply'):
                    BatchPool({'train':NumberLoader()},recipe(True),Path(temporary),QuietLog(),0)

    def test_consumption_replay_and_worker_count_do_not_change_batch(self):
        from chd_stunet.acceleration.training_cpu import BatchPool,BatchQueue
        from chd_stunet.training.contracts import recipe
        root=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            settings=recipe(True,dict(patch_size=[16]*3))
            pools=[]
            ledger=Path(temporary)/'owners';ledger.mkdir()
            env=patch.dict(os.environ,{'CHD_TRAIN_OWNER_DIR':str(ledger),'CHD_TRAIN_NONCE':'test-supply'})
            env.start();self.addCleanup(env.stop)
            try:
                for count in (0,2):
                    output=Path(temporary)/str(count);output.mkdir()
                    pools.append(BatchPool({'train':NumberLoader()},settings,output,QuietLog(),count))
                queues=[BatchQueue(p,'train',0,4) for p in pools]
                self.assertIsNone(pools[1].executor.submit(signal.raise_signal,signal.SIGINT).result(timeout=60))
                before=torch.get_rng_state().clone()
                for step in range(4):
                    serial,parallel=[q.peek() for q in queues]
                    np.testing.assert_array_equal(serial['values'],parallel['values'])
                    self.assertEqual(queues[0].cursor,step)
                    np.testing.assert_array_equal(queues[0].peek()['values'],serial['values'])
                    for q in queues:q.commit()
                # A live change after a drained epoch reopens only its pool.
                # Peek/commit after that must still follow the resumed cursor.
                receipts=list(ledger.glob('worker-*.json'))
                self.assertGreater(len(receipts),0)
                self.assertGreater(pools[1].private_credit(),0)
                unrelated=ledger/'worker-foreign.json';unrelated.write_text('{}')
                queues[1].resize_after_drain(1)
                self.assertTrue(unrelated.exists())
                self.assertTrue(all(not path.exists() for path in receipts))
                resized=BatchQueue(pools[1],'train',7,4,start=2)
                with self.assertRaisesRegex(RuntimeError,'drained'):
                    resized.resize_after_drain(2)
                expected=BatchQueue(pools[0],'train',7,4,start=2)
                while resized.cursor<4:
                    np.testing.assert_array_equal(resized.peek()['values'],expected.peek()['values'])
                    resized.commit();expected.commit()
                torch.testing.assert_close(before,torch.get_rng_state(),atol=0,rtol=0)
                replay=BatchQueue(pools[0],'train',0,4,start=1)
                expected=NumberLoader()( __import__('chd_stunet.acceleration.training_cpu',fromlist=['sequence_seed']).sequence_seed(settings['seed'],'train',0,1))
                np.testing.assert_array_equal(replay.peek()['values'],expected['values'])
                self.assertFalse(replay.ready)
            finally:
                for p in pools:p.close(force=True)


if __name__=='__main__':unittest.main()
