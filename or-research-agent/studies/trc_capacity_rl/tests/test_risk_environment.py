import itertools
import math
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from environment import Order
from scenario_environment import ScenarioConfig
from risk_environment import RiskEnv
from risk_value import conditional_saturation, RiskPatternBank
from risk_oracle import joint_capacity_risk


class RiskEnvironmentTests(unittest.TestCase):
    def test_filter_changes_at_risk_threshold_and_reconstructs_final_ledger(self):
        cfg = ScenarioConfig(slots=16, fleet_capacity=2, charger_capacity=2,
                             max_delay_slots=0, flight_multipliers=(1., 2.1),
                             charge_durations=(2, 2), probabilities=(.7, .3))
        orders = [Order('a', 0, 2.), Order('b', 2, 2.)]
        for delta, actions in [(.01, [0, 1]), (.05, [0])]:
            env = RiskEnv(orders, cfg, delta=delta)
            env.step(1)
            self.assertEqual([c['action'] for c in env.candidates()], actions)
            calls = env.oracle_calls
            env.candidates()
            self.assertEqual(env.oracle_calls, calls)
            env.step(actions[-1])
            self.assertTrue(env.audit()['feasible'])
        env.accepted[0][:] = 0
        self.assertIn('risk_ledger', env.audit()['failures'])

    def test_future_suffix_never_enters_risk_features(self):
        cfg = ScenarioConfig()
        bank = RiskPatternBank(cfg)
        a = RiskEnv([Order('a', 0, 3.)], cfg, bank)
        b = RiskEnv([Order('a', 0, 3.), Order('future', 0, 5.)], cfg, bank)
        for x, y in zip(a.candidates(), b.candidates()):
            np.testing.assert_array_equal(x['features'], y['features'])
            self.assertEqual(x['risk'], y['risk'])

    def test_state_limit_fails_closed(self):
        cfg = ScenarioConfig(slots=16, fleet_capacity=2, charger_capacity=2,
                             max_delay_slots=0, flight_multipliers=(1., 2.1),
                             charge_durations=(2, 2), probabilities=(.7, .3))
        env = RiskEnv([Order('a', 0, 2.), Order('b', 2, 2.)], cfg, max_states=1)
        env.step(1)  # Single mission has an envelope certificate.
        self.assertEqual([c['action'] for c in env.candidates()], [0])
        self.assertEqual(env.oracle_limits, 1)

    def test_cell_saturation_recursion_matches_full_label_enumeration(self):
        rng = np.random.default_rng(204)
        q = np.array([.7, .3]); delta = .13
        use = rng.integers(0, 2, size=(6, 2, 2, 9))
        cap = rng.integers(0, 5, size=(2, 9))
        expected = np.zeros((2, cap.size))
        for labels in itertools.product((0, 1), repeat=len(use)):
            load = sum(use[i, y] for i, y in enumerate(labels))
            for z in (0, 1):
                probability = math.prod((1-delta)*(y == z)+delta*q[y] for y in labels)
                expected[z] += probability*(load == cap).ravel()
        np.testing.assert_allclose(conditional_saturation(use, cap, q, delta), expected, atol=1e-14)

    def test_incremental_union_bound_covers_exact_added_joint_failure(self):
        rng = np.random.default_rng(816)
        q = np.array([.7, .3]); delta = .21
        transition = (1-delta)*np.eye(2)+delta*q[None, :]
        for _ in range(15):
            accepted = rng.integers(0, 2, size=(4, 2, 2, 6))
            new = rng.integers(0, 2, size=(2, 2, 6))
            cap = np.full((2, 6), 3)
            saturation = conditional_saturation(accepted, cap, q, delta)
            bound = float(np.sum(q[:, None]*saturation*(transition@new.reshape(2, -1))))
            before = joint_capacity_risk(accepted, cap, q, delta)['risk']
            after = joint_capacity_risk(np.concatenate([accepted, new[None]]), cap, q, delta)['risk']
            self.assertGreaterEqual(bound+1e-12, after-before)

    def test_no_risk_budget_removes_exposed_opportunity_but_keeps_zero_exposure(self):
        cfg = ScenarioConfig(slots=16, fleet_capacity=2, charger_capacity=2,
                             max_delay_slots=0, flight_multipliers=(1., 2.1),
                             charge_durations=(2, 2), probabilities=(.7, .3))
        bank = RiskPatternBank(cfg, distances=(2.,))
        env = RiskEnv([Order('a', 0, 2.)], cfg, bank)
        env.step(1)
        zero = bank.values(env.capacity, 2, env.accepted, 0.)
        ample = bank.values(env.capacity, 2, env.accepted, 1.)
        self.assertEqual(zero[2], 0.)
        self.assertEqual(ample[2], 1.)
        self.assertEqual(zero[8], 1.)


if __name__ == '__main__':
    unittest.main()
