"""Online reservations with hard core scenarios and a whole-ledger risk limit.

No flight outcome is observed by this controller. The fixed finite law covers
capacity conflicts only; it is not a flight safety or realized profit model.
"""
import math
import numpy as np
from scenario_environment import ScenarioEnv, ScenarioConfig
from risk_oracle import joint_capacity_risk

RISK_CONTEXT_DIM = 10


class RiskEnv(ScenarioEnv):
    def __init__(self, orders, config=ScenarioConfig(), bank=None, delta=.01,
                 alpha=.01, max_states=100000):
        super().__init__(orders, config)
        if len(config.probabilities) != 2 or not math.isfinite(delta) or not 0 <= delta <= 1:
            raise ValueError('Two positive-probability outcomes and valid delta required')
        if not math.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError('Invalid whole-ledger risk limit')
        self.bank = bank
        self.delta, self.alpha, self.max_states = delta, alpha, max_states
        self.accepted = []
        self.risk = self.calculate_risk([])
        self.oracle_calls = self.oracle_limits = self.filtered_modes = 0
        self._cached = None

    def calculate_risk(self, accepted):
        use = np.asarray(accepted) if accepted else np.empty((0, 2, 3, self.config.slots), dtype=int)
        return joint_capacity_risk(use, self.initial[0], self.config.probabilities,
                                   self.delta, self.max_states)

    def candidates(self):
        if self.done:
            return []
        if self._cached is not None:
            return self._cached
        slot = self.orders[self.index].slot
        result = []
        for candidate in super().candidates():
            accepted = self.accepted
            if candidate['action']:
                accepted = self.accepted + [candidate['scenario_usage']]
                risk = self.calculate_risk(accepted)
                self.oracle_calls += 1
                self.oracle_limits += risk['status'] != 'exact'
                if risk['risk_upper'] > self.alpha:
                    self.filtered_modes += 1
                    continue
            else:
                risk = self.risk
            calendar = self.capacity - candidate['reservation']
            context = candidate['features'][:8]
            remaining = max(0., self.alpha - risk['risk_upper'])
            context = np.r_[context, remaining / self.alpha if self.alpha else 0., len(accepted) / 30.]
            if self.bank is None:
                values = candidate['features'][8:]
            else:
                values = self.bank.values(calendar, slot, accepted, remaining)
            result.append({**candidate, 'risk': risk,
                           'features': np.r_[context, values].astype(np.float32)})
        self._cached = result
        return result

    def step(self, action):
        chosen = next((c for c in self.candidates() if c['action'] == action), None)
        if chosen is None:
            raise ValueError('Action exceeds capacity or risk limit')
        reward = super().step(action)
        if action:
            self.accepted.append(chosen['scenario_usage'].copy())
        self.risk = chosen['risk']
        self._cached = None
        return reward

    def audit(self):
        core = super().audit()  # Rebuilds each job directly from original orders.
        accepted = [h['reservation'] for h in self.history if h['action']]
        same = len(accepted) == len(self.accepted) and all(
            np.array_equal(a, b) for a, b in zip(accepted, self.accepted))
        risk = self.calculate_risk(accepted)
        failures = list(core['failures'])
        if not same:
            failures.append('risk_ledger')
        if risk['risk_upper'] > self.alpha:
            failures.append('joint_risk_limit_or_unresolved')
        return {**core, 'feasible': not failures, 'failures': failures, 'risk': risk}
