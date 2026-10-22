"""Audit every training and checkpoint-selection day without changing old runs.

The historical backend source remains immutable. This single-process adapter
temporarily instruments its environment and validation helper, then restores
them even on error. Instrumentation consumes no random numbers.
"""
import argparse
import hashlib
from pathlib import Path
import run_risk_development as backend

ORIGINAL_FIT = backend.fit


def audited_fit(training, validation, cfg, variant, seed, epochs):
    original_env, original_simulate = backend.RiskEnv, backend.simulate
    phase = 'training'
    records = []
    lookup = {name: {tuple(o.order_id for o in sorted(orders, key=lambda o: (o.slot, o.order_id))): day
                     for day, orders in episodes.items()}
              for name, episodes in [('training', training), ('validation', validation)]}
    if any(len(lookup[name]) != len(episodes) for name, episodes in
           [('training', training), ('validation', validation)]):
        raise ValueError('Duplicate order-ID sequences cannot identify audit days')

    class AuditedEnv(original_env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit_phase = phase
            if self.done:
                self.record_completion()

        def record_completion(self):
            audit = self.audit()
            record = {'phase': self.audit_phase,
                      'day': lookup[self.audit_phase][tuple(o.order_id for o in self.orders)],
                      'orders': len(self.orders), 'air_orders': audit['air_orders'],
                      'reward': self.reward, 'feasible': audit['feasible'],
                      'failures': audit['failures'], 'risk': audit['risk']['risk'],
                      'risk_upper': audit['risk']['risk_upper'], 'risk_status': audit['risk']['status'],
                      'oracle_calls': self.oracle_calls, 'oracle_limits': self.oracle_limits,
                      'filtered_modes': self.filtered_modes}
            records.append(record)
            if not audit['feasible']:
                raise AssertionError(record)

        def step(self, action):
            reward = super().step(action)
            if self.done:
                self.record_completion()
            return reward

    def recorded_simulate(*args, **kwargs):
        nonlocal phase
        old_phase = phase
        phase = 'validation'
        try:
            return original_simulate(*args, **kwargs)
        finally:
            phase = old_phase

    backend.RiskEnv, backend.simulate = AuditedEnv, recorded_simulate
    try:
        net, bank, info = ORIGINAL_FIT(training, validation, cfg, variant, seed, epochs)
    finally:
        backend.RiskEnv, backend.simulate = original_env, original_simulate
    expected_validations = ((epochs // 40) + bool(epochs % 40)) * len(validation)
    if sum(r['phase'] == 'training' for r in records) != epochs or \
       sum(r['phase'] == 'validation' for r in records) != expected_validations:
        raise AssertionError('Incomplete episode audit coverage')
    info['episode_audits'] = records
    info['audit_summary'] = {'training_episodes': epochs, 'validation_episodes': expected_validations,
                             'oracle_limits': sum(r['oracle_limits'] for r in records),
                             'max_final_risk_upper': max((r['risk_upper'] for r in records), default=0.),
                             'all_feasible': all(r['feasible'] for r in records)}
    info['audit_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return net, bank, info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11])
    parser.add_argument('--variants', nargs='+', choices=['pattern', 'risk_insertion'], default=['pattern', 'risk_insertion'])
    backend.fit = audited_fit
    backend.main(parser.parse_args())
