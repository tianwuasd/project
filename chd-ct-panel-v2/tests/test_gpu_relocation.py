"""Placement and step-cursor portability; no CUDA devices are allocated."""
from pathlib import Path
import random
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class RelocationTests(unittest.TestCase):
    def test_resource_failures_retry_but_data_errors_do_not(self):
        from chd_stunet.training.relocation import retryable_failure
        self.assertTrue(retryable_failure([{'resource_unavailable':True}],{}))
        self.assertTrue(retryable_failure([{'error_type':'OutOfMemoryError'}],{'gpu':[123]}))
        self.assertFalse(retryable_failure([{'error_type':'OutOfMemoryError'}],{}))
        self.assertFalse(retryable_failure([{'error_type':'ValueError','error':'bad data'}],{'gpu':[123]}))
    def test_replace_shrink_and_zero_preserve_survivors_and_cap(self):
        from chd_stunet.training.relocation import choose_members,foreign_on_members
        rows=[dict(uuid=u,ordinal=i,usable_bytes=100,external_pids=[]) for i,u in enumerate('abcde')]
        rows[1]['external_pids']=[999]
        self.assertEqual(choose_members(list('abcd'),rows,4),list('acde'))
        self.assertEqual(choose_members(list('abcd'),rows[:4],4),list('acd'))
        self.assertEqual(choose_members(list('abcd'),rows,1),['a'])
        self.assertEqual(choose_members([],rows,4),list('acde'))
        self.assertEqual(foreign_on_members(list('acd'),rows,{888}),{})
        self.assertEqual(foreign_on_members(list('abcd'),rows,{888}),{'b':[999]})
        self.assertEqual(foreign_on_members(list('abcd'),rows,{999}),{})
        for row in rows:row['external_pids']=[999]
        self.assertEqual(choose_members(list('abcd'),rows,4),[])

    def test_partial_train_and_validation_rng_can_move_without_changing_cursor(self):
        from chd_stunet.training.rank_runtime import restore_rank_rng
        from chd_stunet.training.checkpoints import rng_state
        for phase in ('train','validation'):
            state=dict(epoch=91,phase=phase,step=119)
            payload=dict(state=state.copy(),distributed=dict(members=list('abcd'),
                rng_by_rank=[rng_state(torch.device('cpu')) for _ in range(4)]))
            with self.assertRaises(ValueError):restore_rank_rng(payload,['a'],0,torch.device('cpu'),42)
            values=[]
            for _ in range(2):
                restore_rank_rng(payload,['a'],0,torch.device('cpu'),42,allow_partial=True)
                values.append((random.random(),float(np.random.rand()),float(torch.rand(()))))
            self.assertEqual(values[0],values[1]);self.assertEqual(payload['state'],state)


if __name__=='__main__':unittest.main()
