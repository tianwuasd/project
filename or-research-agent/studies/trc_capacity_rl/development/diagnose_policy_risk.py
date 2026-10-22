"""Development-only risk diagnostic and frozen-policy feasibility-filter pilot."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from environment import Order
from scenario_environment import ScenarioConfig,ScenarioEnv,patterns
from risk_oracle import joint_capacity_risk
from pattern_value import PatternBank
from run_scenario_development import Critic
from learning import choose


def summarize_risks(rows,alpha=.01):
    """Keep unresolved queries visible; a missing probability is never zero."""
    exact=[r for r in rows if r['status']=='exact']
    return {'records':len(rows),'unresolved_count':len(rows)-len(exact),
            'mean_risk_exact_only':float(np.mean([r['risk'] for r in exact])) if exact else None,
            'mean_risk_upper_all_records':float(np.mean([r['risk_upper'] for r in rows])) if rows else None,
            'exact_records_over_limit':sum(r['risk']>alpha for r in exact),
            'records_not_certified_within_limit':sum(r['risk_upper']>alpha for r in rows)}


def replay(orders,cfg,net=None,bank=None,price=0.,delta=None,alpha=.01):
    env=ScenarioEnv(orders,cfg,feature_bank=bank);accepted=[];latency=[];rejects=limits=0
    while not env.done:
        start=time.perf_counter();cs=env.candidates();valid=[]
        if delta is not None:
            for c in cs:
                if c['action']==0:valid.append(c);continue
                risk=joint_capacity_risk(np.array(accepted+[c['scenario_usage']]),env.initial[0],cfg.probabilities,delta)
                if risk['status']!='exact':limits+=1
                if risk['risk_upper']<=alpha:valid.append(c)
                else:rejects+=1
            cs=valid
        if net is not None:idx=choose(net,cs,cfg.reward_scale)
        else:
            scores=[c['reward']-price*(c['reservation']/np.maximum(env.initial,1)).sum(axis=(1,2)).mean()*
                    (1-env.orders[env.index].slot/cfg.slots) for c in cs]
            idx=int(np.argmax(scores))
        chosen=cs[idx]
        if chosen['action']:accepted.append(chosen['scenario_usage'])
        latency.append((time.perf_counter()-start)*1000);env.step(chosen['action'])
    audit=env.audit()
    if not audit['feasible']:raise AssertionError(audit)
    shape=(0,2,3,cfg.slots)
    # Independently rebuild the binary occupancy from original orders/actions,
    # instead of using accumulated candidate arrays for the final risk audit.
    rebuilt=[]
    for order,h in zip(env.orders,env.history):
        if h['action']:
            rebuilt.append(next(p for p in patterns(order,cfg) if p['action']==h['action'])['scenario_usage'])
    tensor=np.array(rebuilt) if rebuilt else np.empty(shape,dtype=int)
    if accepted and not np.array_equal(np.array(accepted),tensor):raise AssertionError('Risk ledger mismatch')
    return {'reward':env.reward,'air_orders':len(rebuilt),'orders':len(orders),'core_audit':True,
            'mean_decision_ms':float(np.mean(latency)) if latency else 0.,'risk_filtered_modes':rejects,
            'oracle_state_limits':limits},tensor


def main(name):
    out=ROOT/'development/results'/name
    if out.exists():raise FileExistsError('Use a new diagnostic name')
    out.mkdir(parents=True);torch.set_num_threads(1);cfg=ScenarioConfig();bank=PatternBank(cfg)
    path=ROOT/'data/processed/episodes.json';raw=json.loads(path.read_text(encoding='utf-8'))
    validation={d:[Order(**o) for o in orders] for d,orders in raw['validation'].items()}
    seeds=[11,22,33,44,55];policies=[('greedy',0,None,0.),('bid_price',0,None,2.)]
    checkpoint_root=ROOT/'development/results/pilot_002'
    identity=json.loads((checkpoint_root/'identity.json').read_text())
    checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [checkpoint_root/'identity.json',
                                 *[checkpoint_root/f'pattern_{seed}.pt' for seed in seeds]]}
    for source,sha in identity['source_sha256'].items():
        if hashlib.sha256((ROOT/'src'/source).read_bytes()).hexdigest()!=sha:
            raise ValueError('Original training source changed: '+source)
    for seed in seeds:
        net=Critic(bank.size,'pattern');net.load_state_dict(torch.load(checkpoint_root/f'pattern_{seed}.pt',weights_only=True));net.eval()
        policies.append(('pattern',seed,net,0.))
    deltas=[0.,.001,.01,.05,.1,1.];rows=[];filter_rows=[];start=time.perf_counter()
    caps=np.repeat(np.array([cfg.pad_capacity,cfg.fleet_capacity,cfg.charger_capacity])[:,None],cfg.slots,axis=1)
    for name,seed,net,price in policies:
        for day,orders in validation.items():
            base,use=replay(orders,cfg,net,bank if net else None,price)
            for delta in deltas:
                begin=time.perf_counter();risk=joint_capacity_risk(use,caps,cfg.probabilities,delta)
                rows.append({'policy':name,'seed':seed,'day':day,'delta':delta,**base,**risk,'risk_compute_ms':1000*(time.perf_counter()-begin)})
            filtered,use=replay(orders,cfg,net,bank if net else None,price,delta=.01,alpha=.01)
            risk=joint_capacity_risk(use,caps,cfg.probabilities,.01)
            if risk['risk_upper']>.01:raise AssertionError('Accepted policy violates declared model risk')
            filter_rows.append({'policy':name,'seed':seed,'day':day,**filtered,**risk})
        print('finished',name,seed,flush=True)
    summary={}
    for name in ['greedy','bid_price','pattern']:
        by_delta={}
        for d in deltas:
            sub=[r for r in rows if r['policy']==name and r['delta']==d]
            by_delta[str(d)]={**summarize_risks(sub),
                             'mean_oracle_ms':float(np.mean([r['risk_compute_ms'] for r in sub]))}
        sub=[r for r in filter_rows if r['policy']==name]
        summary[name]={'unfiltered':by_delta,'filtered_at_delta_001':{'mean_saving':float(np.mean([r['reward'] for r in sub])),
            'max_risk_upper':max(r['risk_upper'] for r in sub),'mean_risk':float(np.mean([r['risk'] for r in sub])),
            'mean_decision_ms':float(np.mean([r['mean_decision_ms'] for r in sub])),
            'state_limits':sum(r['oracle_state_limits'] for r in sub)}}
    result={'purpose':'development_risk_diagnostic_not_method_superiority','deltas':deltas,'filtered_delta':.01,'joint_risk_limit':.01,
        'summary':summary,'unfiltered_rows':rows,'filtered_rows':filter_rows,'elapsed_seconds':time.perf_counter()-start,
        'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'checkpoint_sha256':checkpoint_sha256,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/risk_oracle.py']},
        'limitations':['All weather laws synthetic and share unchanged one-flight marginals.',
            'Gross expected saving has no penalty/recourse model for tail conflict outcomes; not realized operating profit.',
            'Policies pretrained/selected under the earlier model; no new learning or new confirmation data.',
            'Price and lookahead have not been retuned for the new risk-constrained problem. No superiority conclusion.']}
    (out/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'summary':summary,'elapsed_seconds':result['elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    if not a.name.replace('_','').isalnum():raise ValueError('Simple run name required')
    main(a.name)
