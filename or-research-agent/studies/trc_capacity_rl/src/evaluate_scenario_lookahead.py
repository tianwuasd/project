"""Validation-only tuning of a conditional LP baseline, preserving every row."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import scipy
from environment import Order
from scenario_environment import ScenarioConfig,ScenarioEnv
from scenario_lookahead import ConditionalLookahead

ROOT=Path(__file__).resolve().parents[1]


def main(name,samples,weights):
    out=ROOT/'development/results'/name
    if out.exists():raise FileExistsError('Choose a new run name to preserve earlier evidence')
    out.mkdir(parents=True)
    source=ROOT/'data/processed/episodes.json';raw=json.loads(source.read_text(encoding='utf-8'))
    train={d:[Order(**o) for o in orders] for d,orders in raw['train'].items()}
    val={d:[Order(**o) for o in orders] for d,orders in raw['validation'].items()}
    cfg=ScenarioConfig();rows=[];start=time.perf_counter();counters=[]
    identity={'config':cfg.__dict__,'samples':samples,'weights':weights,'scipy':scipy.__version__,
        'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
    (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    for weight in weights:
        policy=ConditionalLookahead(train,cfg,samples,weight)
        for day,orders in val.items():
            env=ScenarioEnv(orders,cfg);lat=[]
            while not env.done:
                t=time.perf_counter();cs=env.candidates();idx=policy.choose(env,cs)
                lat.append((time.perf_counter()-t)*1000);env.step(cs[idx]['action'])
            audit=env.audit()
            if not audit['feasible']:raise AssertionError(audit)
            rows.append({'day':day,'weight':weight,'reward':env.reward,'air_orders':audit['air_orders'],
                'feasible':True,'mean_decision_ms':float(np.mean(lat)) if lat else 0.,
                'p95_decision_ms':float(np.quantile(lat,.95)) if lat else 0.})
        counters.append({'weight':weight,'solves':policy.solves,'fallbacks':policy.failures,'empirical_support_misses':policy.support_misses})
        print('weight',weight,'validation',float(np.mean([r['reward'] for r in rows if r['weight']==weight])),
              'solves',policy.solves,'fallbacks',policy.failures,flush=True)
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summary={str(w):float(np.mean([r['reward'] for r in rows if r['weight']==w])) for w in weights}
    result={'purpose':'exploratory_validation_tuning','identity':identity,'validation_summary':summary,
            'rows':rows,'counters':counters,'elapsed_seconds':time.perf_counter()-start,
            'limitations':'Training-path two-stage LP relaxation, not an implementable future policy; same-slot suffix conditional on observed count. Outside empirical count support predicts no additional same-slot jobs; see counters.'}
    (out/'experiment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FINISHED',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--weights',nargs='+',type=float,default=[.25,.5,1.]);a=p.parse_args()
    if not a.name.replace('_','').replace('-','').isalnum():raise ValueError('Use simple run name')
    main(a.name,a.samples,a.weights)
