"""Development-only matched learning and baselines under a joint risk limit."""
import argparse
import copy
import hashlib
import json
import platform
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from environment import Order
from scenario_environment import ScenarioConfig
from scenario_lookahead import ConditionalLookahead
from risk_environment import RiskEnv
from risk_value import RiskPatternBank, RiskCritic
from learning import choose

ROOT = Path(__file__).resolve().parents[1]


def simulate(orders, cfg, net=None, bank=None, price=0., lookahead=None):
    env = RiskEnv(orders, cfg, bank)
    latency = []
    while not env.done:
        start = time.perf_counter()
        cs = env.candidates()
        if lookahead is not None:
            idx = lookahead.choose(env, cs)
        elif net is not None:
            idx = choose(net, cs, cfg.reward_scale)
        else:
            scores = [c['reward'] - price*(c['reservation']/np.maximum(env.initial, 1)).sum(axis=(1, 2)).mean()
                      *(1-env.orders[env.index].slot/cfg.slots) for c in cs]
            idx = int(np.argmax(scores))
        env.step(cs[idx]['action'])
        latency.append(1000*(time.perf_counter()-start))
    audit = env.audit()
    if not audit['feasible']:
        raise AssertionError(audit)
    return {'reward': env.reward, 'orders': len(orders), 'air_orders': audit['air_orders'],
            'risk': audit['risk']['risk'], 'risk_upper': audit['risk']['risk_upper'],
            'oracle_calls': env.oracle_calls, 'oracle_limits': env.oracle_limits,
            'filtered_modes': env.filtered_modes, 'feasible': True,
            'mean_decision_ms': float(np.mean(latency)) if latency else 0.,
            'p95_decision_ms': float(np.quantile(latency, .95)) if latency else 0.}


def fit(training, validation, cfg, variant, seed, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    bank = RiskPatternBank(cfg, variant)
    net = RiskCritic(bank.size); target = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    replay = deque(maxlen=50000)
    days = sorted(training)
    steps = updates = 0
    best_score = -float('inf'); best = None; log = []
    start = time.perf_counter()
    for epoch in range(epochs):
        env = RiskEnv(training[random.choice(days)], cfg, bank)
        while not env.done:
            cs = env.candidates()
            epsilon = max(.05, .7*(1-epoch/max(1, .8*epochs)))
            idx = random.randrange(len(cs)) if random.random() < epsilon else choose(net, cs, cfg.reward_scale)
            x = cs[idx]['features']; env.step(cs[idx]['action']); steps += 1
            if env.done:
                replay.append((x, np.zeros((1, len(x)), dtype=np.float32), np.zeros(1, dtype=np.float32), True))
            else:
                nxt = env.candidates()
                replay.append((x, np.array([c['features'] for c in nxt]),
                               np.array([c['reward']/cfg.reward_scale for c in nxt], dtype=np.float32), False))
            if len(replay) >= 64 and steps % 4 == 0:
                batch = random.sample(list(replay), 64)
                current = torch.from_numpy(np.array([b[0] for b in batch]))
                future = torch.from_numpy(np.concatenate([b[1] for b in batch]))
                rewards = torch.from_numpy(np.concatenate([b[2] for b in batch]))
                with torch.no_grad():
                    online = net(future)+rewards; fixed = target(future)+rewards
                    labels = []; offset = 0
                    for b in batch:
                        length = len(b[1]); k = int(torch.argmax(online[offset:offset+length]))
                        labels.append(torch.tensor(0.) if b[3] else fixed[offset+k]); offset += length
                    labels = torch.stack(labels)
                loss = F.smooth_l1_loss(net(current), labels)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5); optimizer.step(); updates += 1
                if updates % 100 == 0:
                    target.load_state_dict(net.state_dict())
        if (epoch+1) % 40 == 0 or epoch == epochs-1:
            score = float(np.mean([simulate(o, cfg, net, bank)['reward'] for o in validation.values()]))
            log.append({'episode': epoch+1, 'validation_mean': score, 'steps': steps, 'updates': updates})
            print(variant, seed, epoch+1, round(score, 5), flush=True)
            if score > best_score:
                best = copy.deepcopy(net.state_dict()); best_score = score
    net.load_state_dict(best)
    return net, bank, {'variant': variant, 'seed': seed, 'epochs': epochs, 'steps': steps, 'updates': updates,
                       'parameter_count': sum(p.numel() for p in net.parameters()), 'log': log,
                       'best_validation_mean': best_score, 'seconds': time.perf_counter()-start}


def main(args):
    if args.epochs < 1 or not args.name.replace('_', '').isalnum():
        raise ValueError('Positive epochs and simple immutable run name required')
    out = ROOT/'development/results'/args.name
    if out.exists():
        raise FileExistsError('Existing runs are immutable')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    start = time.perf_counter(); cfg = ScenarioConfig()
    data_path = ROOT/'data/processed/episodes.json'
    raw = json.loads(data_path.read_text(encoding='utf-8'))
    training = {d: [Order(**o) for o in orders] for d, orders in raw['train'].items()}
    validation = {d: [Order(**o) for o in orders] for d, orders in raw['validation'].items()}
    identity = {'purpose': 'exploratory_validation_no_confirmation_claim', 'config': asdict(cfg),
                'delta': .01, 'whole_ledger_alpha': .01, 'epochs': args.epochs, 'seeds': args.seeds,
                'variants': args.variants, 'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
                'protocol_sha256': hashlib.sha256((ROOT/'development/risk_method_plan.md').read_bytes()).hexdigest(),
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__}}
    (out/'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    price_scores = {}; price_rows = {}
    for price in [0, 1, 2, 4, 8, 12, 20, 32]:
        price_rows[price] = [{'day': d, **simulate(o, cfg, price=price)} for d, o in validation.items()]
        price_scores[price] = float(np.mean([r['reward'] for r in price_rows[price]]))
    price = max(price_scores, key=price_scores.get)
    rows = [{'policy': name, 'seed': 0, **r} for name, p in [('greedy', 0), ('bid_price', price)] for r in price_rows[p]]
    lp_scores = {}; lp_rows = {}; lp_diagnostics = {}
    for weight in [.25, .5, 1.]:
        policy = ConditionalLookahead(training, cfg, samples=3, weight=weight)
        lp_rows[weight] = [{'day': d, **simulate(o, cfg, lookahead=policy)} for d, o in validation.items()]
        lp_scores[weight] = float(np.mean([r['reward'] for r in lp_rows[weight]]))
        lp_diagnostics[weight] = {'solves': policy.solves, 'failures': policy.failures, 'support_misses': policy.support_misses}
        print('LP', weight, lp_scores[weight], flush=True)
    weight = max(lp_scores, key=lp_scores.get)
    rows.extend({'policy': 'conditional_lp', 'seed': 0, **r} for r in lp_rows[weight])
    baselines = {'price_scores': price_scores, 'selected_price': price, 'lp_scores': lp_scores,
                 'selected_lp_weight': weight, 'lp_diagnostics': lp_diagnostics,
                 'all_price_rows': price_rows, 'all_lp_rows': lp_rows}
    (out/'baselines.json').write_text(json.dumps(baselines, indent=2, allow_nan=False), encoding='utf-8')
    logs = []
    for variant in args.variants:
        for seed in args.seeds:
            net, bank, info = fit(training, validation, cfg, variant, seed, args.epochs)
            torch.save(net.state_dict(), out/f'{variant}_{seed}.pt')
            (out/f'{variant}_{seed}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
            logs.append(info)
            rows.extend({'policy': variant, 'seed': seed, 'day': d, **simulate(o, cfg, net, bank)}
                        for d, o in validation.items())
    summary = {name: float(np.mean([r['reward'] for r in rows if r['policy'] == name]))
               for name in sorted({r['policy'] for r in rows})}
    result = {'identity': identity, 'validation_summary': summary, 'validation_rows': rows, 'training': logs,
              'selected_price': price, 'selected_lp_weight': weight, 'elapsed_seconds': time.perf_counter()-start,
              'limitations': ['Validation selected all policies; no independent confirmation data.',
                  'Synthetic binary disturbance law; empirical flight logs do not calibrate this law.',
                  'Gross proxy saving omits tail-conflict recourse cost.',
                  'Conditional LP relaxes future integrality and joint risk; current action uses exact risk filtering.',
                  'Insertion features are a union-bound heuristic, not a joint value or safety certificate.']}
    (out/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'validation_summary': summary, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11])
    parser.add_argument('--variants', nargs='+', choices=['pattern', 'risk_insertion'], default=['pattern', 'risk_insertion'])
    main(parser.parse_args())
