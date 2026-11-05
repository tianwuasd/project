from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chd_stunet.training.parallel_policy import layout,slots_for,supply_plan,expansion_baseline,expansion_regressed
from chd_stunet.training.supervisor import max_ranks
from chd_stunet.acceleration.preprocess_cpu import GIB


class ParallelPolicyTests(unittest.TestCase):
    def test_fixed_work_and_foreground_slot_survive_layout_changes(self):
        for world,expected in enumerate(([4],[2,2],[2,1,1],[1,1,1,1]),1):
            self.assertEqual(layout(4,world),expected)
            slots=[s for rank in range(world) for s in slots_for(4,world,rank)]
            self.assertEqual(slots,[0,1,2,3])
            self.assertEqual([s for s in slots if s>=round(4*(1-.33))],[3])
        with self.assertRaises(ValueError):layout(4,5)
        self.assertEqual(max_ranks('auto',4),4)
        self.assertEqual(max_ranks('8',4),4)
        with self.assertRaises(ValueError):max_ranks('0',4)

    def test_budget_is_for_whole_group_and_serial_has_real_admission(self):
        stage=dict(cpu_budget=32,memory_budget_bytes=40*GIB,reserve_bytes=10*GIB)
        settings=dict(batch_size=4,patch_size=[64]*3,train_steps_per_epoch=250)
        with patch('chd_stunet.training.parallel_policy.policy',return_value=stage):
            plan=supply_plan({},settings,{},4,maximum=5)
            self.assertLessEqual(sum(plan['workers']),5)
            self.assertLessEqual(sum(plan['workers'])+5,stage['cpu_budget'])
            allocated=sum(r['memory_budget_bytes'] for r in plan['ranks'])+GIB
            self.assertLessEqual(allocated,stage['memory_budget_bytes'])
            self.assertEqual(supply_plan({},settings,{},4,0)['workers'],[0,0,0,0])
        with patch('chd_stunet.training.parallel_policy.policy',return_value=dict(stage,cpu_budget=4)):
            with self.assertRaises(MemoryError):supply_plan({},settings,{},4,0)

    def test_natural_expansion_needs_warmup_and_three_slow_epochs(self):
        old=['a','b'];rows=[dict(epoch=i,seconds=s,members=old,global_batch=4) for i,s in enumerate([1,10,12,11])]
        self.assertIsNone(expansion_baseline(rows[:3],4,old))
        baseline=expansion_baseline(rows,4,old)
        self.assertEqual(baseline['reference_seconds'],11)
        new=[dict(seconds=s,global_batch=4) for s in [100,14,14,14]]
        self.assertFalse(expansion_regressed(baseline,new[:3],4))
        self.assertTrue(expansion_regressed(baseline,new,4))
        self.assertFalse(expansion_regressed(baseline,new,8))
        new[-1]['seconds']=12
        self.assertFalse(expansion_regressed(baseline,new,4))


if __name__=='__main__':unittest.main()
