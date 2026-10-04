"""Finite joint-scenario reservations; synthetic uncertainty, fixed forecast set.

Launch decisions are common to every scenario. No realized weather or future
order is supplied to the policy. Guarantees only cover the listed scenarios.
"""
from dataclasses import dataclass, asdict
import copy
import math
import numpy as np
from environment import Config, Order

CONTEXT_DIM=8


@dataclass(frozen=True)
class ScenarioConfig(Config):
    flight_multipliers: tuple = (1., 1.5)
    charge_durations: tuple = (2, 3)
    probabilities: tuple = (.7, .3)


def validate(config):
    vectors=(config.flight_multipliers,config.charge_durations,config.probabilities)
    if not vectors[0] or len({len(v) for v in vectors})!=1:
        raise ValueError('Scenario dimensions must agree and be nonempty')
    if any(not math.isfinite(v) or v<=0 for vec in vectors for v in vec):
        raise ValueError('Scenario values must be positive and finite')
    if not math.isclose(sum(config.probabilities),1.,abs_tol=1e-9):
        raise ValueError('Scenario probabilities must sum to one')
    integers=(config.slots,config.slot_minutes,*config.charge_durations)
    if any(int(v)!=v or v<1 for v in integers):raise ValueError('Invalid time discretization')
    for v in (config.pad_capacity,config.fleet_capacity,config.charger_capacity,config.max_delay_slots):
        if not math.isfinite(v) or int(v)!=v or v<0:raise ValueError('Invalid capacity/delay')
    for v in (config.drone_speed_km_min,config.ground_speed_km_min,config.road_factor,
              config.max_distance_km,config.reward_scale):
        if not math.isfinite(v) or v<=0:raise ValueError('Invalid physical scale')


def patterns(order,config):
    """All physics-feasible common launch modes, before capacity filtering."""
    if order.distance_km>config.max_distance_km:return []
    result=[];n=len(config.probabilities)
    for delay in range(config.max_delay_slots+1):
        usage=np.zeros((n,3,config.slots),dtype=np.int16)
        rewards=[];times=[];valid=True
        for w,(mult,charge) in enumerate(zip(config.flight_multipliers,config.charge_durations)):
            flight=max(1,math.ceil(mult*order.distance_km/(config.drone_speed_km_min*config.slot_minutes)))
            launch=order.slot+delay;landing=launch+2*flight+1;ready=landing+charge
            delivery=(delay+flight)*config.slot_minutes+2.
            if ready>config.slots or delivery>order.deadline_minutes:
                valid=False;break
            usage[w,0,launch]+=1;usage[w,0,landing]+=1
            usage[w,1,launch:ready]=1;usage[w,2,landing:ready]=1
            ground=2*config.road_factor*order.distance_km/config.ground_speed_km_min+4
            air=2*order.distance_km*mult/config.drone_speed_km_min+4+.15*delay*config.slot_minutes
            rewards.append(ground-air);times.append((launch,landing,ready,delivery))
        if valid:
            result.append({'action':delay+1,'scenario_usage':usage,
                'reward':float(np.dot(config.probabilities,rewards)),
                'scenario_rewards':tuple(rewards),'times':tuple(times)})
    return result


class ScenarioEnv:
    def __init__(self,orders,config=ScenarioConfig(),mode='joint',feature_bank=None):
        validate(config)
        if mode not in ('joint','envelope'):raise ValueError('Unknown reservation mode')
        self.config=config;self.mode=mode;self.feature_bank=feature_bank
        self.orders=sorted(orders,key=lambda o:(o.slot,o.order_id))
        if len({o.order_id for o in self.orders})!=len(self.orders):raise ValueError('Duplicate order IDs')
        if any(o.slot<0 or o.slot>=config.slots or int(o.slot)!=o.slot or
               o.distance_km<0 or not math.isfinite(o.distance_km) or
               not math.isfinite(o.deadline_minutes) or o.deadline_minutes<=0 for o in self.orders):
            raise ValueError('Invalid order')
        base=np.repeat(np.array([config.pad_capacity,config.fleet_capacity,config.charger_capacity])[:,None],config.slots,axis=1)
        count=len(config.probabilities) if mode=='joint' else 1
        self.initial=np.repeat(base[None,:,:],count,axis=0)
        self.capacity=self.initial.copy();self.index=0;self.reward=0.;self.history=[]

    @property
    def done(self):return self.index>=len(self.orders)

    def features(self,capacity,slot):
        t=slot/self.config.slots
        revealed=self.orders[:self.index+1]
        in_slot=sum(o.slot==slot for o in revealed)
        recent=[o for o in revealed if o.slot>=slot-4]
        context=np.array([t,math.sin(math.pi*t),math.cos(math.pi*t),1.,
            in_slot/10.,len(revealed)/100.,len(recent)/20.,
            np.mean([o.distance_km for o in recent])/self.config.max_distance_km if recent else 0.],dtype=np.float32)
        if self.feature_bank is not None:
            if self.mode!='joint':raise ValueError('Pattern bank requires joint-scenario calendar')
            features=self.feature_bank.values(capacity,slot)
        else:
            features=capacity.astype(np.float32)/np.maximum(self.initial,1)
            features[:,:,:slot]=0;features=features.ravel()
        return np.r_[context,features].astype(np.float32)

    def candidates(self):
        if self.done:return []
        slot=self.orders[self.index].slot
        result=[{'action':0,'reward':0.,'reservation':np.zeros_like(self.capacity),
                 'features':self.features(self.capacity,slot)}]
        for p in patterns(self.orders[self.index],self.config):
            reservation=p['scenario_usage'] if self.mode=='joint' else p['scenario_usage'].max(axis=0,keepdims=True)
            if np.any(reservation>self.capacity):continue
            result.append({**p,'reservation':reservation,
                           'features':self.features(self.capacity-reservation,slot)})
        return result

    def step(self,action):
        if self.done:raise ValueError('Episode finished')
        chosen=next((c for c in self.candidates() if c['action']==action),None)
        if chosen is None:raise ValueError('Infeasible action')
        self.capacity-=chosen['reservation'];self.reward+=chosen['reward']
        self.history.append({'order':asdict(self.orders[self.index]),'action':int(action),
                             'reward':chosen['reward'],'reservation':chosen['reservation'].copy()})
        self.index+=1
        return chosen['reward']

    def audit(self):
        """Independent reconstruction; does not call patterns/candidates or trust ledgers."""
        cfg=self.config;n=len(cfg.probabilities);failures=[]
        load=np.zeros((n,3,cfg.slots),dtype=np.int64)
        reserved=np.zeros_like(self.initial);total=0.
        if len(self.history)!=self.index:failures.append('history_length')
        for i,h in enumerate(self.history):
            if i>=len(self.orders):failures.append('extra_order');continue
            o=self.orders[i];act=h['action'];actual=np.zeros_like(load);reward=0.
            if h['order']!=asdict(o):failures.append('order_identity')
            if act:
                if not isinstance(act,int) or not 1<=act<=cfg.max_delay_slots+1:
                    failures.append('action');continue
                wait=act-1;start=o.slot+wait
                for w in range(n):
                    leg=max(1,int(math.ceil(o.distance_km*cfg.flight_multipliers[w]/(cfg.slot_minutes*cfg.drone_speed_km_min))))
                    end=start+leg+leg+1;available=end+cfg.charge_durations[w]
                    if available>cfg.slots:failures.append('horizon');continue
                    if o.distance_km>cfg.max_distance_km:failures.append('range')
                    if (wait+leg)*cfg.slot_minutes+2>o.deadline_minutes:failures.append('deadline')
                    actual[w,0,start]+=1;actual[w,0,end]+=1
                    for t in range(start,available):actual[w,1,t]+=1
                    for t in range(end,available):actual[w,2,t]+=1
                    reward+=cfg.probabilities[w]*(2*o.distance_km*(cfg.road_factor/cfg.ground_speed_km_min-
                            cfg.flight_multipliers[w]/cfg.drone_speed_km_min)-.15*wait*cfg.slot_minutes)
            expected=actual if self.mode=='joint' else actual.max(axis=0,keepdims=True)
            if not np.array_equal(h['reservation'],expected):failures.append('reservation_ledger')
            if not math.isclose(reward,h['reward'],abs_tol=1e-7):failures.append('reward_ledger')
            load+=actual;reserved+=expected;total+=reward
        base=self.initial[0]
        if np.any(load>base[None,:,:]):failures.append('scenario_capacity')
        if np.any(reserved>self.initial):failures.append('reservation_capacity')
        if not np.array_equal(self.initial-reserved,self.capacity):failures.append('calendar_ledger')
        if not math.isclose(total,self.reward,abs_tol=1e-6):failures.append('reward_total')
        return {'feasible':not failures,'failures':sorted(set(failures)),
                'air_orders':sum(h['action']>0 for h in self.history),
                'reward_recomputed':total,'scenario_peak_loads':load.max(axis=2).tolist()}


def enumerate_optimum(orders,config=ScenarioConfig(),mode='joint'):
    """Small-instance exhaustive oracle, with future orders; not an online policy."""
    if len(orders)>8:raise ValueError('Exhaustive oracle limited to eight orders')
    leaves=0;best=-float('inf');actions=None
    def visit(env):
        nonlocal leaves,best,actions
        if env.done:
            leaves+=1
            if not env.audit()['feasible']:raise AssertionError('Oracle produced invalid schedule')
            if env.reward>best:best=env.reward;actions=[h['action'] for h in env.history]
            return
        for c in env.candidates():
            nxt=copy.deepcopy(env);nxt.step(c['action']);visit(nxt)
    visit(ScenarioEnv(orders,config,mode))
    return {'objective':best,'actions':actions,'enumerated_leaves':leaves,'status':'exhaustive_optimal'}
