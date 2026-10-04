"""Exploratory development only. Never evaluates a fresh confirmation set."""
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
from scenario_environment import CONTEXT_DIM,ScenarioConfig,ScenarioEnv,enumerate_optimum
from pattern_value import PatternBank
from learning import choose

ROOT=Path(__file__).resolve().parents[1]


class Critic(nn.Module):
    def __init__(self,dimension,variant):
        super().__init__();self.variant=variant
        if variant=='mlp':
            self.network=nn.Sequential(nn.Linear(dimension+CONTEXT_DIM,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        else:
            self.weights=nn.Sequential(nn.Linear(CONTEXT_DIM,32),nn.Tanh(),nn.Linear(32,dimension))
            nn.init.zeros_(self.weights[-1].weight);nn.init.constant_(self.weights[-1].bias,-4.)
            self.intercept=nn.Sequential(nn.Linear(CONTEXT_DIM,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,x):
        if self.variant=='mlp':return self.network(x).squeeze(-1)
        feat=x[:,CONTEXT_DIM:] if self.variant=='pattern' else 1-torch.exp(-x[:,CONTEXT_DIM:])
        return (F.softplus(self.weights(x[:,:CONTEXT_DIM]))*feat).sum(dim=1)+self.intercept(x[:,:CONTEXT_DIM]).squeeze(-1)


def simulate(orders,cfg,net=None,bank=None,price=0.,mode='joint'):
    env=ScenarioEnv(orders,cfg,mode,bank);latency=[]
    while not env.done:
        start=time.perf_counter();cs=env.candidates()
        if net is not None:idx=choose(net,cs,cfg.reward_scale)
        else:
            # Mean across joint scenarios avoids multiplying price by scenario count.
            # Uniform averaging here is a tunable resource-price heuristic, not reward expectation.
            scores=[c['reward']-price*(c['reservation']/np.maximum(env.initial,1)).sum(axis=(1,2)).mean()*
                    (1-env.orders[env.index].slot/cfg.slots) for c in cs]
            idx=int(np.argmax(scores))
        latency.append((time.perf_counter()-start)*1000)
        env.step(cs[idx]['action'])
    audit=env.audit()
    if not audit['feasible']:raise AssertionError(audit)
    return {'reward':env.reward,'orders':len(orders),'air_orders':audit['air_orders'],
            'feasible':True,'mean_decision_ms':float(np.mean(latency)) if latency else 0.,
            'p95_decision_ms':float(np.quantile(latency,.95)) if latency else 0.}


def fit(training,validation,cfg,variant,seed,epochs):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    bank=PatternBank(cfg) if variant=='pattern' else None
    dim=bank.size if bank else len(cfg.probabilities)*3*cfg.slots
    net=Critic(dim,variant);target=copy.deepcopy(net)
    opt=torch.optim.Adam(net.parameters(),lr=.001);replay=deque(maxlen=50000)
    days=list(training);steps=updates=0;log=[];best=None;best_score=-float('inf')
    for epoch in range(epochs):
        env=ScenarioEnv(training[random.choice(days)],cfg,feature_bank=bank)
        while not env.done:
            cs=env.candidates();eps=max(.05,.7*(1-epoch/max(1,.8*epochs)))
            idx=random.randrange(len(cs)) if random.random()<eps else choose(net,cs,cfg.reward_scale)
            x=cs[idx]['features'];env.step(cs[idx]['action']);steps+=1
            if env.done:
                replay.append((x,np.zeros((1,len(x)),dtype=np.float32),np.zeros(1,dtype=np.float32),True))
            else:
                nxt=env.candidates()
                replay.append((x,np.array([c['features'] for c in nxt]),
                               np.array([c['reward']/cfg.reward_scale for c in nxt],dtype=np.float32),False))
            if len(replay)>=64 and steps%4==0:
                batch=random.sample(list(replay),64)
                current=torch.from_numpy(np.array([b[0] for b in batch]))
                future=torch.from_numpy(np.concatenate([b[1] for b in batch]))
                rewards=torch.from_numpy(np.concatenate([b[2] for b in batch]))
                with torch.no_grad():
                    online=net(future)+rewards;fixed=target(future)+rewards;ys=[];start=0
                    for b in batch:
                        length=len(b[1]);k=int(torch.argmax(online[start:start+length]))
                        ys.append(torch.tensor(0.) if b[3] else fixed[start+k]);start+=length
                    labels=torch.stack(ys)
                loss=F.smooth_l1_loss(net(current),labels)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();updates+=1
                if updates%100==0:target.load_state_dict(net.state_dict())
        if (epoch+1)%40==0 or epoch==epochs-1:
            score=float(np.mean([simulate(o,cfg,net,bank)['reward'] for o in validation.values()]))
            log.append({'training_episode':epoch+1,'validation_mean':score,'steps':steps,'updates':updates})
            print(variant,seed,epoch+1,round(score,4),flush=True)
            if score>best_score:best=copy.deepcopy(net.state_dict());best_score=score
    net.load_state_dict(best)
    return net,bank,{'variant':variant,'seed':seed,'epochs':epochs,'steps':steps,'updates':updates,
                     'parameter_count':sum(p.numel() for p in net.parameters()),'best_validation_mean':best_score,'log':log}


def main(args):
    if args.epochs<1:raise ValueError('Training budget must be positive')
    if not args.name.replace('_','').replace('-','').isalnum():raise ValueError('Use a simple run name')
    out=ROOT/'development'/'results'/args.name
    if out.exists():raise FileExistsError('Development records are immutable; choose another run name')
    out.mkdir(parents=True);start=time.perf_counter();torch.set_num_threads(1)
    path=ROOT/'data/processed/episodes.json';raw=json.loads(path.read_text(encoding='utf-8'))
    # Old test data already observed historically, but this run uses only train/validation.
    training={day:[Order(**o) for o in orders] for day,orders in raw['train'].items()}
    validation={day:[Order(**o) for o in orders] for day,orders in raw['validation'].items()}
    cfg=ScenarioConfig()
    identity={'purpose':'exploratory_development_no_confirmation_claim','config':asdict(cfg),
              'epochs':args.epochs,'seeds':args.seeds,'variants':args.variants,
              'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
              'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}}
    (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    prices=[0,1,2,4,8,12,20,32]
    price_scores={p:float(np.mean([simulate(o,cfg,price=p)['reward'] for o in validation.values()])) for p in prices}
    price=max(prices,key=lambda p:price_scores[p]);print('validation price',price,'reward',price_scores[price],flush=True)
    rows=[];logs=[]
    for name,p,mode in [('greedy',0,'joint'),('bid_price',price,'joint'),('envelope_greedy',0,'envelope')]:
        for day,orders in validation.items():rows.append({'day':day,'policy':name,'seed':0,**simulate(orders,cfg,price=p,mode=mode)})
    # Save baseline results before potentially long training.
    (out/'baselines.json').write_text(json.dumps({'price_scores':price_scores,'selected_price':price,'rows':rows},indent=2),encoding='utf-8')
    for variant in args.variants:
        for seed in args.seeds:
            net,bank,info=fit(training,validation,cfg,variant,seed,args.epochs)
            torch.save(net.state_dict(),out/f'{variant}_{seed}.pt')
            (out/f'{variant}_{seed}.json').write_text(json.dumps(info,indent=2),encoding='utf-8');logs.append(info)
            for day,orders in validation.items():rows.append({'day':day,'policy':variant,'seed':seed,**simulate(orders,cfg,net,bank)})
    names=sorted({r['policy'] for r in rows})
    summary={name:float(np.mean([r['reward'] for r in rows if r['policy']==name])) for name in names}
    micro_cfg=ScenarioConfig(slots=16,fleet_capacity=2,charger_capacity=2,max_delay_slots=0,
                flight_multipliers=(1.,2.1),charge_durations=(2,2),probabilities=(.5,.5))
    micro={mode:enumerate_optimum([Order('a',0,2.),Order('b',2,2.)],micro_cfg,mode) for mode in ('joint','envelope')}
    result={'identity':identity,'validation_summary':summary,'validation_rows':rows,'training':logs,
            'selected_price':price,'microexample':micro,'elapsed_seconds':time.perf_counter()-start,
            'limitations':['Validation selected both checkpoints and price: exploratory, not independent evidence.',
                'Synthetic common disturbance; no real weather/flight calibration.',
                'Rolling optimization and independent-residual stress baselines remain to be implemented.']}
    (out/'experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'validation_summary':summary,'elapsed_seconds':result['elapsed_seconds'],'microexample':micro}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--seeds',nargs='+',type=int,default=[11]);p.add_argument('--variants',nargs='+',choices=['calendar','pattern','mlp'],default=['calendar','pattern','mlp'])
    main(p.parse_args())
