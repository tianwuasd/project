"""Post hoc development calibration of frozen continuation-value scale."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from environment import Order
from scenario_environment import ScenarioConfig
from risk_value import RiskPatternBank, RiskCritic
from run_risk_development import simulate


class ScaledValue(torch.nn.Module):
    def __init__(self, network, scale):
        super().__init__()
        self.network, self.scale = network, scale

    def forward(self, features):
        return self.scale*self.network(features)


def main(name):
    if not name.replace('_', '').isalnum():
        raise ValueError('Simple immutable run name required')
    out = ROOT/'development/results'/name
    if out.exists():
        raise FileExistsError('Do not overwrite earlier diagnostics')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    source = ROOT/'development/results/risk_pilot_001'
    identity = json.loads((source/'identity.json').read_text())
    for p, sha in identity['source_sha256'].items():
        if hashlib.sha256((ROOT/'src'/p).read_bytes()).hexdigest() != sha:
            raise ValueError('Training identity changed: '+p)
    data = ROOT/'data/processed/episodes.json'
    if hashlib.sha256(data.read_bytes()).hexdigest() != identity['data_sha256']:
        raise ValueError('Data identity changed')
    raw = json.loads(data.read_text(encoding='utf-8'))
    validation = {d: [Order(**o) for o in orders] for d, orders in raw['validation'].items()}
    previous = json.loads((source/'experiment.json').read_text())['validation_rows']
    cfg = ScenarioConfig()
    grid = [0., .25, .5, .75, 1., 1.5, 2.]
    rows, summary, check_hashes = [], {}, {}
    start = time.perf_counter()
    for variant in ['pattern', 'risk_insertion']:
        bank = RiskPatternBank(cfg, variant)
        path = source/f'{variant}_11.pt'
        check_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        net = RiskCritic(bank.size)
        net.load_state_dict(torch.load(path, weights_only=True)); net.eval()
        values = {}
        for scale in grid:
            for day, orders in validation.items():
                result = simulate(orders, cfg, ScaledValue(net, scale), bank)
                if scale in (0., 1.):
                    policy = 'greedy' if scale == 0. else variant
                    old = next(r for r in previous if r['policy'] == policy and r['day'] == day)
                    if not np.isclose(result['reward'], old['reward'], atol=1e-10, rtol=0.):
                        raise AssertionError('Anchor replay changed')
                rows.append({'variant': variant, 'scale': scale, 'day': day, **result})
            values[scale] = float(np.mean([r['reward'] for r in rows if r['variant'] == variant and r['scale'] == scale]))
            print(variant, scale, values[scale], flush=True)
        selected = max(grid, key=values.get)
        summary[variant] = {'scale_means': values, 'selected_scale': selected, 'selected_validation_mean': values[selected]}
    files = [Path(__file__), ROOT/'development/value_scale_plan.md', source/'identity.json']
    result = {'purpose': 'post_hoc_validation_diagnostic_extra_tuning_not_confirmation',
              'summary': summary, 'rows': rows, 'grid': grid, 'checkpoint_sha256': check_hashes,
              'input_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'data_sha256': identity['data_sha256'], 'training_source_sha256': identity['source_sha256'],
              'anchor_checks': 'scale 0 matches greedy and scale 1 matches original TD on every day',
              'elapsed_seconds': time.perf_counter()-start}
    (out/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    main(parser.parse_args().name)
