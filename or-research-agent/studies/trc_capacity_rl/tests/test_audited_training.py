import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_risk_development as backend
from environment import Order
from scenario_environment import ScenarioConfig
from risk_environment import RiskEnv
from run_risk_development_audited import audited_fit


class AuditedTrainingTests(unittest.TestCase):
    def test_training_and_checkpoint_validation_are_both_recorded(self):
        cfg = ScenarioConfig(slots=5, max_delay_slots=0, charge_durations=(2, 2))
        original_env, original_simulate = backend.RiskEnv, backend.simulate
        _, _, info = audited_fit({'training_day': [Order('train', 0, .5)]},
                                 {'validation_day': [Order('val', 0, .5)]},
                                 cfg, 'pattern', 11, 1)
        rows = info['episode_audits']
        self.assertEqual([(r['phase'], r['day']) for r in rows],
                         [('training', 'training_day'), ('validation', 'validation_day')])
        self.assertTrue(all(r['feasible'] and r['risk_upper'] <= .01 for r in rows))
        self.assertIs(backend.RiskEnv, original_env)
        self.assertIs(backend.simulate, original_simulate)

    def test_intermediate_validation_state_limits_are_retained(self):
        class LimitedEnv(RiskEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, max_states=1)
        cfg = ScenarioConfig(slots=16, fleet_capacity=2, charger_capacity=2,
                             max_delay_slots=0, flight_multipliers=(1., 2.1),
                             charge_durations=(2, 2))
        orders = [Order('a', 0, 2.), Order('b', 2, 2.)]
        greedy = lambda network, cs, scale: max(range(len(cs)), key=lambda i: cs[i]['reward'])
        with patch.object(backend, 'RiskEnv', LimitedEnv), patch.object(backend, 'choose', greedy):
            _, _, info = audited_fit({'train': orders}, {'validation': orders}, cfg, 'pattern', 11, 1)
        val = next(r for r in info['episode_audits'] if r['phase'] == 'validation')
        self.assertGreater(val['oracle_limits'], 0)
        self.assertTrue(val['feasible'])


if __name__ == '__main__':
    unittest.main()
