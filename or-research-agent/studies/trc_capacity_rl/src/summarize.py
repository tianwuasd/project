"""Summarize held-out days without treating training seeds as independent days."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def day_means(rows, field='reward'):
    groups = defaultdict(list)
    for row in rows:
        if field in row:
            groups[row['day']].append(row[field])
    return {day: float(np.mean(values)) for day, values in sorted(groups.items())}


def interval(values, block=1):
    """Circular day-block bootstrap; exploratory interval for a short dependent series."""
    values = np.asarray(values)
    rng = np.random.default_rng(20261004)
    starts = rng.integers(0, len(values), (10000, int(np.ceil(len(values)/block))))
    indices = (starts[:,:,None] + np.arange(block)) % len(values)
    samples = values[indices.reshape(10000,-1)[:,:len(values)]].mean(axis=1)
    return [float(x) for x in np.quantile(samples,[.025,.975])]


def summarize():
    source = json.loads((ROOT/'results/experiment.json').read_text())
    policies = list(dict.fromkeys(r['policy'] for r in source['test']))
    grouped = {p: [r for r in source['test'] if r['policy']==p] for p in policies}
    daily = {p:day_means(rows) for p,rows in grouped.items()}
    summary = {}
    for p, rows in grouped.items():
        vals = list(daily[p].values())
        seeds = sorted(set(r['seed'] for r in rows))
        seed_means = [float(np.mean([r['reward'] for r in rows if r['seed']==s])) for s in seeds]
        summary[p] = {'mean_daily_saving':float(np.mean(vals)),
                      'seed_mean_range':[min(seed_means),max(seed_means)],
                      'mean_air_orders':float(np.mean([r['air_orders'] for r in rows])) if 'air_orders' in rows[0] else None,
                      'mean_of_daily_decision_ms':float(np.mean([r['mean_decision_ms'] for r in rows])) if 'mean_decision_ms' in rows[0] else None}
    comparisons = {}
    for p in ['greedy','bid_price','sampled_rollout','unconstrained']:
        differences = [daily['monotone'][d]-daily[p][d] for d in daily[p]]
        comparisons[p] = {'mean_difference':float(np.mean(differences)),
                          'percent_of_comparator_mean':100*float(np.mean(differences))/summary[p]['mean_daily_saving'],
                          'paired_day_percentile_ci95':interval(differences),
                          'circular_3_observed_day_block_ci95':interval(differences,3)}
    stress = {}
    for fleet in [2,6]:
        stress[str(fleet)] = {p:float(np.mean(list(day_means([r for r in source['stress'] if r['fleet']==fleet and r['policy']==p]).values())))
                             for p in ['greedy','bid_price','monotone','unconstrained']}
    # This diagnostic is explicitly post hoc; it does not replace the full test.
    busy_days = {r['day'] for r in grouped['greedy'] if r['orders']>1}
    sensitivity = {p:float(np.mean([v for d,v in days.items() if d in busy_days])) for p,days in daily.items()}
    result = {'policies':summary,'monotone_minus_comparator':comparisons,'fleet_stress':stress,
              'post_hoc_non_singleton_days':{'n':len(busy_days),'means':sensitivity},
              'inference_note':'Seeds averaged within day. Intervals condition on five fitted models, do not capture all training uncertainty, and are descriptive with only 17 nonempty days. Block length 3 denotes consecutive observed days, not calendar days.'}
    (ROOT/'results/summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['greedy','sampled_rollout','bid_price','unconstrained','monotone','clairvoyant_milp']
    fig, ax = plt.subplots(figsize=(8.2,4.5),layout='constrained')
    labels=['Greedy','Sampled rollout','Bid price','RL: signed weights','RL: monotone','Offline upper bound']
    ax.barh(labels,[summary[p]['mean_daily_saving'] for p in names],color=['#a4b8c4']*3+['#d6a767','#287b8e','#59636b'])
    ax.set_xlabel('Mean daily generalized cost saving (simulated units)')
    ax.set_title('Held-out LaDe replay: 17 days, 911 orders')
    ax.invert_yaxis()
    for i,p in enumerate(names):
        v=summary[p]['mean_daily_saving'];ax.text(v+1,i,f'{v:.2f}',va='center')
    ax.set_xlim(0,115)
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/test_results.png',dpi=200);plt.close(fig)
    return result


if __name__=='__main__':
    print(json.dumps(summarize(),indent=2))
