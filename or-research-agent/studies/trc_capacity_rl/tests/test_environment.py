import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from environment import Config, ReservationEnv, Order, offline_optimum


class EnvironmentTests(unittest.TestCase):
    def test_resources_and_return_are_reserved(self):
        env = ReservationEnv([Order('a', 0, 2., 20.)], Config(slots=20))
        candidates = env.candidates()
        chosen = next(c for c in candidates if c['action'] == 1)
        self.assertEqual(chosen['usage'][0].sum(), 2)  # launch and landing
        self.assertGreater(chosen['usage'][1].sum(), 0)
        self.assertEqual(chosen['usage'][2].sum(), 2)
        env.step(1)
        self.assertTrue(env.audit()['feasible'])
        self.assertTrue(env.done)

    def test_capacity_mask_fallback_and_invalid_action(self):
        config = Config(slots=20, pad_capacity=0)
        env = ReservationEnv([Order('a', 0, 2., 20.)], config)
        self.assertEqual([c['action'] for c in env.candidates()], [0])
        with self.assertRaises(ValueError): env.step(1)
        self.assertEqual(env.step(0), 0)

    def test_no_future_information_in_candidates(self):
        a = Order('a', 0, 2., 20.)
        env1 = ReservationEnv([a, Order('x', 1, 3., 20.)], Config(slots=20))
        env2 = ReservationEnv([a, Order('y', 10, 1., 50.)], Config(slots=20))
        for c1, c2 in zip(env1.candidates(), env2.candidates()):
            np.testing.assert_equal(c1['features'], c2['features'])
            self.assertEqual(c1['reward'], c2['reward'])

    def test_offline_bound_and_independent_audit(self):
        orders = [Order('a',0,2.,20.), Order('b',0,3.,20.), Order('c',1,1.,20.)]
        config = Config(slots=20)
        oracle = offline_optimum(orders, config)
        env = ReservationEnv(orders,config)
        while not env.done: env.step(max(env.candidates(),key=lambda c:c['reward'])['action'])
        self.assertGreaterEqual(oracle['objective']+1e-6, env.reward)
        self.assertTrue(env.audit()['feasible'])
        env.capacity[0,0] = -1
        self.assertFalse(env.audit()['feasible'])

    def test_audit_rebuilds_reservations_instead_of_trusting_ledger(self):
        env = ReservationEnv([Order('a',0,2.,20.)], Config(slots=20))
        env.step(1)
        env.history[0]['usage'][:] = 0
        env.capacity[:] = env.initial  # Corrupt both ledger and balance consistently.
        self.assertFalse(env.audit()['feasible'])

    def test_audit_recomputes_delivery_and_reward(self):
        env = ReservationEnv([Order('a',0,2.,20.)], Config(slots=20))
        env.step(1)
        env.history[0]['delivery_minutes'] = 0
        env.history[0]['reward'] += 100
        self.assertFalse(env.audit()['feasible'])


if __name__ == '__main__': unittest.main()
