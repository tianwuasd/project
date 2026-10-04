"""Online single-depot round-trip delivery with exact resource reservations."""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class Config:
    slots: int = 60
    slot_minutes: int = 5
    pad_capacity: int = 1
    fleet_capacity: int = 4
    charger_capacity: int = 2
    charge_slots: int = 2
    max_delay_slots: int = 3
    max_distance_km: float = 6.
    drone_speed_km_min: float = .8
    ground_speed_km_min: float = .35
    road_factor: float = 1.35
    reward_scale: float = 20.


@dataclass(frozen=True)
class Order:
    order_id: str
    slot: int
    distance_km: float
    deadline_minutes: float = 30.


class ReservationEnv:
    def __init__(self, orders, config=Config()):
        self.config = config
        self.orders = sorted(orders, key=lambda x: (x.slot, x.order_id))
        if any(o.slot < 0 or o.slot >= config.slots or o.distance_km < 0 or not math.isfinite(o.distance_km) for o in orders):
            raise ValueError('Invalid order geometry or time')
        caps = [config.pad_capacity, config.fleet_capacity, config.charger_capacity]
        if any(c < 0 or int(c) != c for c in caps): raise ValueError('Invalid capacity')
        self.initial = np.repeat(np.array(caps,dtype=np.int32)[:,None],config.slots,axis=1)
        self.capacity = self.initial.copy()
        self.index = 0; self.reward = 0.; self.history = []

    @property
    def done(self): return self.index >= len(self.orders)

    def features(self, capacity, slot):
        fraction = slot / self.config.slots
        context = np.array([fraction, math.sin(math.pi*fraction), math.cos(math.pi*fraction), 1.], dtype=np.float32)
        norm = capacity.astype(np.float32) / np.maximum(self.initial,1)
        norm[:,:slot] = 0
        return np.r_[context,norm.ravel()].astype(np.float32)

    def candidates(self):
        if self.done: return []
        order = self.orders[self.index]; cfg = self.config
        zero = np.zeros_like(self.capacity)
        result = [{'action':0,'reward':0.,'usage':zero,'features':self.features(self.capacity,order.slot),'delivery_minutes':None}]
        if order.distance_km > cfg.max_distance_km: return result
        flight = max(1, math.ceil((order.distance_km/cfg.drone_speed_km_min)/cfg.slot_minutes))
        outbound_minutes = flight*cfg.slot_minutes + 2.
        # Return and complete charging before the drone can serve a new sortie.
        for delay in range(cfg.max_delay_slots+1):
            launch = order.slot + delay
            landing = launch + 2*flight + 1
            ready = landing + cfg.charge_slots
            delivery = delay*cfg.slot_minutes + outbound_minutes
            if ready >= cfg.slots or delivery > order.deadline_minutes: continue
            usage = np.zeros_like(self.capacity)
            usage[0,launch] += 1; usage[0,landing] += 1
            usage[1,launch:ready] = 1
            usage[2,landing:ready] = 1
            if np.any(usage>self.capacity): continue
            # Generalized transport effort in cost units; not empirical monetary costs.
            ground_cost = 2*cfg.road_factor*order.distance_km/cfg.ground_speed_km_min + 4
            air_cost = 2*order.distance_km/cfg.drone_speed_km_min + 4 + .15*delay*cfg.slot_minutes
            reward = ground_cost-air_cost
            result.append({'action':delay+1,'reward':reward,'usage':usage,
                           'features':self.features(self.capacity-usage,order.slot),
                           'delivery_minutes':delivery,'launch':launch,'landing':landing,'ready':ready})
        return result

    def step(self, action):
        if self.done: raise ValueError('Episode already finished')
        candidate = next((c for c in self.candidates() if c['action']==action),None)
        if candidate is None: raise ValueError('Action is infeasible')
        self.capacity -= candidate['usage']; self.reward += candidate['reward']
        self.history.append({'order':asdict(self.orders[self.index]),'action':int(action),'reward':candidate['reward'],
                             'usage':candidate['usage'].copy(),'delivery_minutes':candidate['delivery_minutes']})
        self.index += 1
        return candidate['reward']

    def audit(self):
        # Reconstruct physics from the immutable original orders and chosen actions.
        # Do not call candidates() or trust recorded usage/reward/delivery times.
        cfg=self.config;load=np.zeros_like(self.initial);failures=[];reward=0.
        if len(self.history)!=self.index: failures.append('history_length')
        for i,h in enumerate(self.history):
            if i>=len(self.orders): failures.append('extra_order');continue
            order=self.orders[i];action=h['action'];use=np.zeros_like(load);r=0.;delivery=None
            if h['order']!=asdict(order): failures.append('order_identity')
            if action:
                if not 1<=action<=cfg.max_delay_slots+1: failures.append('action');continue
                delay=action-1
                flight=max(1,int(math.ceil(order.distance_km/(cfg.drone_speed_km_min*cfg.slot_minutes))))
                depart=order.slot+delay;land=depart+flight+flight+1;ready=land+cfg.charge_slots
                delivery=(delay+flight)*cfg.slot_minutes+2.
                if order.distance_km>cfg.max_distance_km: failures.append('range')
                if delivery>order.deadline_minutes: failures.append('deadline')
                if ready>=cfg.slots: failures.append('horizon');continue
                for slot in (depart,land): use[0,slot]+=1
                for slot in range(depart,ready): use[1,slot]+=1
                for slot in range(land,ready): use[2,slot]+=1
                r=2*order.distance_km*(cfg.road_factor/cfg.ground_speed_km_min-1/cfg.drone_speed_km_min)-.15*delay*cfg.slot_minutes
            if delivery!=h['delivery_minutes']: failures.append('delivery_ledger')
            if not math.isclose(r,h['reward'],abs_tol=1e-7): failures.append('reward_ledger')
            if not np.array_equal(use,h['usage']): failures.append('reservation_ledger')
            load+=use;reward+=r
        if np.any(load>self.initial) or np.any(self.capacity<0): failures.append('capacity')
        if not np.array_equal(self.initial-load,self.capacity): failures.append('ledger')
        if not math.isclose(reward,self.reward,abs_tol=1e-6): failures.append('reward_total')
        return {'feasible':not failures,'failures':failures,'air_orders':sum(h['action']>0 for h in self.history),
                'ground_orders':sum(h['action']==0 for h in self.history),'reward_recomputed':reward}


def offline_optimum(orders,config=Config()):
    """Perfect-information upper bound; never used by the online learned policy."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    if not orders: return {'objective':0.,'status':0,'upper_bound':0.}
    columns=[]; objective=[]; order_rows=[]
    for i,order in enumerate(orders):
        env=ReservationEnv([order],config)
        for c in env.candidates()[1:]:
            columns.append(c['usage'].ravel());objective.append(c['reward']);order_rows.append(i)
    if not columns: return {'objective':0.,'status':0,'upper_bound':0.}
    cover=np.zeros((len(orders),len(columns)))
    cover[order_rows,np.arange(len(columns))]=1
    matrix=np.vstack([cover,np.array(columns).T])
    capacities=ReservationEnv([],config).initial.ravel()
    result=milp(-np.array(objective),integrality=np.ones(len(columns)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,-np.inf,np.r_[np.ones(len(orders)),capacities]),
        options={'time_limit':30.,'mip_rel_gap':0.0001})
    return {'objective':float(-result.fun) if result.fun is not None else None,'status':int(result.status),
            'upper_bound':float(-result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else None,
            'mip_gap':float(getattr(result,'mip_gap',np.nan))}
