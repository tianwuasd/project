"""Synthetic dispatch pattern selection: exhaustive oracle, greedy and optional MILP."""
import argparse
import itertools
import json
import math
import platform
import time
from pathlib import Path


def demo_instance():
    return {'instance_type': 'synthetic_micro', 'orders': ['A', 'B', 'C'],
            'capacities': {'air_t1': 1, 'ground_t1': 2}, 'patterns': [
                {'id': 'A_air', 'order': 'A', 'cost': 2, 'usage': {'air_t1': 1}},
                {'id': 'A_ground', 'order': 'A', 'cost': 7, 'usage': {'ground_t1': 1}},
                {'id': 'B_air', 'order': 'B', 'cost': 1, 'usage': {'air_t1': 1}},
                {'id': 'B_ground', 'order': 'B', 'cost': 7, 'usage': {'ground_t1': 1}},
                {'id': 'C_ground', 'order': 'C', 'cost': 3, 'usage': {'ground_t1': 1}}]}


def validate_instance(data):
    if len(set(data['orders'])) != len(data['orders']):
        raise ValueError('Orders must be unique')
    ids = [p['id'] for p in data['patterns']]
    if len(ids) != len(set(ids)):
        raise ValueError('Pattern ids must be unique')
    for cap in data['capacities'].values():
        if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
            raise ValueError('Capacities must be finite and nonnegative')
    for p in data['patterns']:
        if p['order'] not in data['orders'] or not math.isfinite(p['cost']):
            raise ValueError('Invalid pattern order or cost')
        for resource, usage in p['usage'].items():
            if resource not in data['capacities'] or not math.isfinite(usage) or usage < 0:
                raise ValueError('Resource usage must be finite, nonnegative and defined')


def validate_solution(data, selected):
    validate_instance(data)
    by_id = {p['id']: p for p in data['patterns']}
    violations = []
    if len(set(selected)) != len(selected) or any(x not in by_id for x in selected):
        return {'feasible': False, 'violations': ['duplicate or unknown pattern'], 'objective': None}
    rows = [by_id[x] for x in selected]
    for order in data['orders']:
        if sum(p['order'] == order for p in rows) != 1:
            violations.append('coverage:' + order)
    loads = {r: sum(p['usage'].get(r, 0) for p in rows) for r in data['capacities']}
    for resource, value in loads.items():
        if value > data['capacities'][resource] + 1e-8:
            violations.append('capacity:' + resource)
    return {'feasible': not violations, 'violations': violations, 'loads': loads,
            'objective': sum(p['cost'] for p in rows)}


def exact(data, max_combinations=1000000):
    validate_instance(data)
    choices = [[p['id'] for p in data['patterns'] if p['order'] == order] for order in data['orders']]
    count = math.prod(map(len, choices))
    if count > max_combinations:
        raise ValueError('Enumeration limit exceeded; use a MILP solver')
    best, evaluated = None, 0
    for selected in itertools.product(*choices):
        check = validate_solution(data, selected)
        evaluated += 1
        if check['feasible'] and (best is None or check['objective'] < best['objective']):
            best = {'selected': list(selected), 'objective': check['objective']}
    return {'status': 'optimal' if best else 'infeasible', 'evaluated': evaluated,
            'selected': best['selected'] if best else [], 'objective': best['objective'] if best else None}


def greedy(data):
    validate_instance(data)
    loads = {r: 0 for r in data['capacities']}
    selected, cost = [], 0
    for order in data['orders']:
        candidates = sorted((p for p in data['patterns'] if p['order'] == order), key=lambda p: (p['cost'], p['id']))
        chosen = next((p for p in candidates if all(loads[r] + p['usage'].get(r, 0) <= cap for r, cap in data['capacities'].items())), None)
        if chosen is None:
            return {'status': 'no_solution_found', 'objective': None, 'selected': selected}
        selected.append(chosen['id']); cost += chosen['cost']
        for resource in loads:
            loads[resource] += chosen['usage'].get(resource, 0)
    return {'status': 'heuristic_feasible', 'objective': cost, 'selected': selected}


def milp_solve(data):
    validate_instance(data)
    import numpy as np
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    patterns = data['patterns']
    if not data['orders']:
        return {'status': 'optimal', 'objective': 0, 'selected': [], 'dual_bound': 0, 'mip_gap': 0, 'scipy': scipy.__version__}
    if not patterns:
        return {'status': 'infeasible', 'objective': None, 'selected': [], 'scipy': scipy.__version__}
    rows = [[int(p['order'] == order) for p in patterns] for order in data['orders']]
    rows += [[p['usage'].get(resource, 0) for p in patterns] for resource in data['capacities']]
    lower = [1.] * len(data['orders']) + [-np.inf] * len(data['capacities'])
    upper = [1.] * len(data['orders']) + list(data['capacities'].values())
    result = milp(c=np.array([p['cost'] for p in patterns], dtype=float), integrality=np.ones(len(patterns)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(np.array(rows, dtype=float), lower, upper),
                  options={'time_limit': 30.0, 'mip_rel_gap': 0.0})
    selected = [p['id'] for p, value in zip(patterns, result.x) if value > .5] if result.x is not None else []
    check = validate_solution(data, selected)
    status = {0: 'optimal', 1: 'limit_reached', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}.get(result.status, 'unknown')
    if result.x is not None and not check['feasible']:
        raise ValueError('Solver returned an invalid integer solution')
    return {'status': status, 'objective': check['objective'] if result.x is not None else None,
            'selected': selected, 'mip_gap': getattr(result, 'mip_gap', None),
            'dual_bound': getattr(result, 'mip_dual_bound', None), 'message': result.message, 'scipy': scipy.__version__}


def run(data):
    result = {'instance': data, 'python': platform.python_version(), 'algorithms': {},
              'scope': 'Synthetic fixed feasible-pattern selection only; not a routing or uncertainty benchmark.'}
    for name, solve in [('enumeration', exact), ('greedy', greedy), ('scipy_milp', milp_solve)]:
        start = time.perf_counter()
        try:
            answer = solve(data)
        except ImportError:
            answer = {'status': 'unavailable', 'reason': 'Optional SciPy not installed'}
        answer['elapsed_seconds'] = time.perf_counter() - start
        if answer.get('selected'):
            answer['validation'] = validate_solution(data, answer['selected'])
        result['algorithms'][name] = answer
    oracle = result['algorithms']['enumeration']
    solved = result['algorithms']['scipy_milp']
    if solved['status'] == 'optimal' and not math.isclose(oracle['objective'], solved['objective'], abs_tol=1e-7):
        raise ValueError('MILP disagrees with exhaustive oracle')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance'); parser.add_argument('--output')
    args = parser.parse_args()
    data = json.loads(Path(args.instance).read_text(encoding='utf-8-sig')) if args.instance else demo_instance()
    result = run(data)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding='utf-8')
    print(text)
