"""Conditional training-day LP lookahead, an optimistic two-stage heuristic.

Only the current action is applied. Sampled future requests come from training
days, never the evaluation day's unrevealed suffix. Future LPs relax integrality
and know their sampled paths; their values are not implementable future policies.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scenario_environment import patterns


class ConditionalLookahead:
    def __init__(self,training,config,samples=3,weight=1.):
        if samples<1 or samples>len(training):raise ValueError('Invalid number of forecast days')
        self.config=config;self.samples=samples;self.weight=weight;self.days=sorted(training)
        self.forecasts={};self.descriptors={};self.slot_counts={};self.failures=0;self.solves=0;self.support_misses=0
        for day in self.days:
            orders=sorted(training[day],key=lambda o:(o.slot,o.order_id))
            columns=[];rewards=[];owner=[];slots=[];ranks=[]
            counts=np.zeros(config.slots,dtype=int)
            for j,order in enumerate(orders):
                counts[order.slot]+=1
                for p in patterns(order,config):
                    columns.append(p['scenario_usage'].ravel());rewards.append(p['reward']);owner.append(j);slots.append(order.slot);ranks.append(counts[order.slot])
            n=len(columns)
            cover=sparse.csc_matrix((np.ones(n),(owner,np.arange(n))),shape=(len(orders),n))
            usage=sparse.csc_matrix(np.array(columns).T) if n else sparse.csc_matrix((len(config.probabilities)*3*config.slots,0))
            self.forecasts[day]=(np.array(slots),np.array(rewards),sparse.vstack([cover,usage],format='csc'),len(orders),np.array(ranks))
            self.slot_counts[day]=counts
            self.descriptors[day]=np.array([self.describe(orders,t) for t in range(config.slots)])

    @staticmethod
    def describe(orders,slot):
        past=[o for o in orders if o.slot<slot]
        recent=sum(o.slot>=slot-4 for o in past)
        mean_distance=np.mean([o.distance_km for o in past]) if past else 0.
        return np.array([len(past)/10.,recent/5.,mean_distance/3.])

    def choose(self,env,candidates):
        if env.mode!='joint' or env.config!=self.config:raise ValueError('Lookahead configuration mismatch')
        if len(candidates)==1:return 0
        slot=env.orders[env.index].slot
        # This slice is the revealed prefix, even when there are same-slot requests.
        prefix=env.orders[:env.index+1]
        observed=self.describe(prefix,slot)
        seen=sum(o.slot==slot for o in prefix)
        compatible=[d for d in self.days if self.slot_counts[d][slot]>=seen]
        if not compatible:
            # Observed count outside empirical support: predict no more in this
            # slot, use nearest past profiles for later slots, and disclose it.
            compatible=self.days;self.support_misses+=1
        ranked=sorted(compatible,key=lambda d:(float(np.sum((self.descriptors[d][slot]-observed)**2)),d))[:self.samples]
        scores=np.array([c['reward'] for c in candidates])
        for day in ranked:
            slots,rewards,matrix,norders,ranks=self.forecasts[day]
            keep=(slots>slot)|((slots==slot)&(ranks>seen))
            if not keep.any():continue
            a=matrix[:,keep];objective=-rewards[keep]
            for j,candidate in enumerate(candidates):
                rhs=np.r_[np.ones(norders),(env.capacity-candidate['reservation']).ravel()]
                result=linprog(objective,A_ub=a,b_ub=rhs,bounds=(0.,1.),method='highs',options={'time_limit':2.})
                self.solves+=1
                if not result.success:
                    # Do not silently mix a failed candidate with valid future scores.
                    self.failures+=1
                    return int(np.argmax([c['reward'] for c in candidates]))
                scores[j]+=self.weight*(-float(result.fun))/len(ranked)
        return int(np.argmax(scores))
