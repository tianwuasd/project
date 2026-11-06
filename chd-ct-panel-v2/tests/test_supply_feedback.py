from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chd_stunet.acceleration.preprocess_cpu import GIB
from chd_stunet.training.supply_feedback import RankSupplyWindow, SupplyTrials, next_worker_layout


class SupplyFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.workers = [2, 2, 2, 1]
        self.queues = [dict(wait=.02, fraction=.03, service=.47)] * 3 + [
            dict(wait=.42, fraction=.66, service=.59)]
        self.sizes = [3*GIB] * 4
        self.stage = dict(cpu_budget=12, reserve_bytes=4*GIB)
        self.fresh = dict(cpu_budget=40, cpu_permitted=64, available_memory_bytes=80*GIB,
                          memory_budget_bytes=64*GIB)

    def plan(self, hard_cap=None):
        return next_worker_layout(self.workers, self.queues, self.sizes, self.stage,
                                  self.fresh, hard_cap, 250)['workers']

    def test_grow_starved_rank_after_cpu_and_memory_recover(self):
        # Boot's 12 available CPU slots could only carry 7 workers + 4 ranks
        # + supervisor. New idle/permitted CPU evidence admits one more now.
        self.assertEqual(self.plan(), [2]*4)
        self.fresh['cpu_permitted'] = 12
        self.assertEqual(self.plan(), self.workers)

    def test_fresh_quota_and_incremental_idle_are_independent(self):
        self.fresh['cgroup_cpu_quota'] = 14
        self.assertEqual(self.plan(), self.workers)
        self.fresh.update(cgroup_cpu_quota=None, cpu_budget=0)
        self.assertLessEqual(sum(self.plan()), sum(self.workers))

    def test_transfer_with_operator_cap_only_if_predicted_throughput_improves(self):
        self.assertEqual(self.plan(7), [1, 2, 2, 2])
        self.queues[:3] = [dict(wait=.02, fraction=.03, service=.58)]*3
        self.assertEqual(self.plan(7), self.workers)

    def test_grow_with_only_verified_resident_memory_credit(self):
        self.fresh.update(available_memory_bytes=10*GIB, memory_budget_bytes=8*GIB)
        self.assertEqual(self.plan(), self.workers)
        self.fresh['worker_private_credit'] = [6*GIB, 6*GIB, 6*GIB, 3*GIB]
        self.assertEqual(self.plan(), [2]*4)
        self.fresh['available_memory_bytes'] = 5*GIB
        self.assertEqual(self.plan(), self.workers)

    def test_one_bad_window_alone_does_not_trigger(self):
        samples = RankSupplyWindow()
        for _ in range(20):
            samples.observe(.01, .99, .1)
        self.assertIsNone(samples.view())
        for _ in range(20):
            samples.observe(.5, .5, .5)
        self.queues[3] = samples.view()
        self.assertEqual(self.plan(), self.workers)

    def test_unhelpful_trial_rolls_back_and_defers_repeated_restart(self):
        trial = SupplyTrials()
        def decide(layout, seconds):
            return trial.decide(layout, self.queues, self.sizes, self.stage,
                                self.fresh, None, 250, seconds)['workers']
        new = decide(self.workers, 170)
        self.assertEqual(new, [2]*4)
        self.assertEqual(decide(new, 220), new)   # warmup
        self.assertEqual(decide(new, 174), new)
        self.assertEqual(decide(new, 173), self.workers)
        for _ in range(23):
            self.assertEqual(decide(self.workers, 169), self.workers)
        self.assertEqual(decide(self.workers, 169), new)
