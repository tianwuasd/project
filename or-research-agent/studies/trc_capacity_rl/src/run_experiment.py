"""Frozen training/validation/test protocol; results are never tuned on test days."""
import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from environment import Config,Order,ReservationEnv,offline_optimum
from learning import ValueNetwork,train,choose

ROOT=Path(__file__).resolve().parents[1]


def load_splits():
    raw=json.loads((ROOT/'data/processed/episodes.json').read_text(encoding='utf-8'))
    return {k:{d:[Order(**o) for o in orders] for d,orders in rows.items()} for k,rows in raw.items()}


def policy_score(c,env,price):
    # Tunable separable resource price, same resource quantities as the RL model.
    consumption=(c['usage']/np.maximum(env.initial,1)).sum()
    time_factor=1-env.orders[env.index].slot/env.config.slots
    return c['reward']-price*consumption*time_factor


def simulate(orders,config,policy='greedy',network=None,price=0.,training=None,rng=None):
    env=ReservationEnv(orders,config);latencies=[]
    while not env.done:
        t=time.perf_counter();cs=env.candidates()
        if policy=='ground':index=0
        elif network is not None:index=choose(network,cs,config.reward_scale)
        elif policy=='rollout':
            scores=np.array([c['reward'] for c in cs]);current=env.orders[env.index].slot
            selected=rng.choice(list(training),size=3,replace=True)
            for d in selected:
                future=[o for o in training[d] if o.slot>current]
                for j,c in enumerate(cs):
                    look=ReservationEnv(future,config);look.capacity=env.capacity-c['usage']
                    while not look.done:
                        opt=max(look.candidates(),key=lambda x:x['reward']);look.step(opt['action'])
                    scores[j]+=look.reward/len(selected)
            index=int(np.argmax(scores))
        else:index=max(range(len(cs)),key=lambda j:policy_score(cs[j],env,price))
        latencies.append(time.perf_counter()-t);env.step(cs[index]['action'])
    audit=env.audit()
    if not audit['feasible']:raise ValueError(audit)
    return {'reward':env.reward,'air_orders':audit['air_orders'],'ground_orders':audit['ground_orders'],
            'orders':len(orders),'feasible':audit['feasible'],'mean_decision_ms':1000*float(np.mean(latencies)) if latencies else 0,
            'p95_decision_ms':1000*float(np.quantile(latencies,.95)) if latencies else 0}


def experiment_identity(config,epochs):
    import scipy
    return {'config':asdict(config),'epochs':epochs,
            'processed_data_sha256':hashlib.sha256((ROOT/'data/processed/episodes.json').read_bytes()).hexdigest(),
            'source_sha256':{name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                             for name in ['environment.py','learning.py','prepare_data.py','run_experiment.py']},
            'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__}}


def validate_checkpoint(info,identity,seed,monotone,epochs):
    if info.get('identity')!=identity or info.get('seed')!=seed or info.get('monotone')!=monotone or info.get('training_episodes')!=epochs:
        raise ValueError('Checkpoint identity changed or missing; preserve prior results and use --retrain explicitly')


def experiment(epochs=400,seeds=(11,22,33,44,55),retrain=False):
    start=time.perf_counter();splits=load_splits();cfg=Config();torch.set_num_threads(1)
    identity=experiment_identity(cfg,epochs)
    out=ROOT/'results';out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    prices=[0,1,2,4,8,12,20,32]
    price_scores={p:float(np.mean([simulate(o,cfg,price=p)['reward'] for o in splits['validation'].values()])) for p in prices}
    price=max(prices,key=lambda p:price_scores[p]);print('selected validation price',price,flush=True)
    networks={};logs=[]
    for variant in ['monotone','unconstrained']:
        for seed in seeds:
            checkpoint=out/'checkpoints'/f'{variant}_{seed}.pt';logpath=checkpoint.with_suffix('.json')
            if checkpoint.exists() and logpath.exists() and not retrain:
                info=json.loads(logpath.read_text());net=ValueNetwork(3*cfg.slots,variant=='monotone')
                validate_checkpoint(info,identity,seed,variant=='monotone',epochs)
                net.load_state_dict(torch.load(checkpoint,weights_only=True))
            else:
                net,info=train(splits['train'],splits['validation'],cfg,seed,variant=='monotone',epochs)
                info['identity']=identity
                torch.save(net.state_dict(),checkpoint);logpath.write_text(json.dumps(info,indent=2),encoding='utf-8')
            networks[(variant,seed)]=net;logs.append(info)
            print(variant,seed,'validation',round(info['best_validation_reward'],3),'steps',info['steps'],flush=True)
    rows=[]
    # Policies/hyperparameters now frozen. Same held-out days for every comparison.
    for d,orders in splits['test'].items():
        oracle=offline_optimum(orders,cfg)
        for name,p in [('ground',0.),('greedy',0.),('bid_price',price)]:
            rows.append({'day':d,'policy':name,'seed':0,**simulate(orders,cfg,name,price=p)})
        day_rng=np.random.default_rng(int(d)+20261004)
        rows.append({'day':d,'policy':'sampled_rollout','seed':0,**simulate(orders,cfg,'rollout',training=splits['train'],rng=day_rng)})
        rows.append({'day':d,'policy':'clairvoyant_milp','seed':0,'reward':oracle['objective'],'bound':oracle['upper_bound'],'status':oracle['status']})
        for (variant,seed),net in networks.items():rows.append({'day':d,'policy':variant,'seed':seed,**simulate(orders,cfg,network=net)})
        print('evaluated day',d,'orders',len(orders),flush=True)
    # Stress test with frozen weights; policy is never retrained on these test outcomes.
    stress=[]
    for fleet in [2,6]:
        stressed=replace(cfg,fleet_capacity=fleet)
        for d,orders in splits['test'].items():
            for name,p in [('greedy',0.),('bid_price',price)]:
                stress.append({'fleet':fleet,'day':d,'policy':name,'seed':0,**simulate(orders,stressed,price=p)})
            for (variant,seed),net in networks.items():
                stress.append({'fleet':fleet,'day':d,'policy':variant,'seed':seed,**simulate(orders,stressed,network=net)})
    result={'config':asdict(cfg),'epochs':epochs,'training_seeds':list(seeds),'selected_bid_price':price,'validation_price_scores':price_scores,
            'training':logs,'test':rows,'stress':stress,'elapsed_seconds':time.perf_counter()-start,
            'versions':identity['versions'],'checkpoint_identity':identity,
            'processed_data_sha256':hashlib.sha256((ROOT/'data/processed/episodes.json').read_bytes()).hexdigest(),
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
            'interpretation':'Public ground-delivery traces replayed under synthetic drone and cost assumptions; not causal deployment evidence.'}
    (out/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('FINISHED',round(result['elapsed_seconds'],1),'seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=400);p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);p.add_argument('--retrain',action='store_true');a=p.parse_args()
    experiment(a.epochs,tuple(a.seeds),a.retrain)
