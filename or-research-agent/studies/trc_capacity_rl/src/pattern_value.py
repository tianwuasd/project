"""Joint-resource insertion features and nonnegative contextual value weights."""
import numpy as np
from environment import Order
from scenario_environment import patterns, validate


class PatternBank:
    def __init__(self,config,distances=(1.,2.,3.,4.,5.,6.)):
        validate(config)
        if not distances or any(not np.isfinite(d) or d<=0 for d in distances):
            raise ValueError('Template distances must be positive and finite')
        self.config=config;self.distances=tuple(distances)
        self.size=config.slots*len(distances)
        self.release=np.repeat(np.arange(config.slots),len(distances))
        indices=[];units=[];groups=[]
        for slot in range(config.slots):
            for j,distance in enumerate(distances):
                for p in patterns(Order('template',slot,distance),config):
                    flat=p['scenario_usage'].ravel();idx=np.flatnonzero(flat)
                    indices.append(idx);units.append(flat[idx]);groups.append(slot*len(distances)+j)
        length=max((len(i) for i in indices),default=1)
        self.indices=np.zeros((len(indices),length),dtype=np.int64)
        self.units=np.ones((len(indices),length),dtype=np.float32)
        self.mask=np.zeros((len(indices),length),dtype=bool)
        self.groups=np.array(groups,dtype=np.int64)
        for row,(idx,unit) in enumerate(zip(indices,units)):
            self.indices[row,:len(idx)]=idx;self.units[row,:len(idx)]=unit;self.mask[row,:len(idx)]=True

    def values(self,capacity,slot):
        expected=(len(self.config.probabilities),3,self.config.slots)
        if capacity.shape!=expected or np.any(capacity<0):raise ValueError('Invalid calendar')
        result=np.zeros(self.size,dtype=np.float32)
        if len(self.groups):
            ratios=np.where(self.mask,capacity.ravel()[self.indices]/self.units,np.inf)
            modes=np.minimum(1.,ratios.min(axis=1))
            np.maximum.at(result,self.groups,modes)
        # More orders may still be revealed in the current discrete slot.
        # Observed within-slot progress is context, not the unknown total count.
        result[self.release<slot]=0
        return result
