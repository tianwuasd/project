"""Exact tiny-instance support/risk diagnostic at fixed one-flight marginals.

No flight model calibration or dispatch advantage claim. The controller chooses
one static set of reservations before realizing any common/private disturbance.
"""
import itertools
import json
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from environment import Order
from scenario_environment import ScenarioConfig,patterns


def outcome_probability(outcomes,q,delta):
    """Stationary resampling channel: retain common label or redraw its prior.

    P(Y_i=j|Z=z)=(1-delta)1[j=z]+delta*q[j]. Private redraws are
    conditionally independent. Marginal P(Y_i=j)=q[j] at every delta.
    """
    if not 0<=delta<=1 or len(q)!=2 or not math.isclose(sum(q),1.):raise ValueError('Invalid law')
    return sum(q[z]*math.prod((1-delta)*(y==z)+delta*q[y] for y in outcomes) for z in (0,1))


def exact_audit(modes,capacities,q,delta):
    if len(modes)>18:raise ValueError('Exhaustive diagnostic limited to 18 accepted flights')
    total=violation=0.;positive_support=bad_support=0;bad_outcomes=[]
    for labels in itertools.product((0,1),repeat=len(modes)):
        probability=outcome_probability(labels,q,delta)
        if probability<=0:continue
        total+=probability;positive_support+=1
        load=np.zeros_like(capacities)
        for mode,label in zip(modes,labels):load+=mode['scenario_usage'][label]
        if np.any(load>capacities):
            violation+=probability;bad_support+=1
            bad_outcomes.append({'labels':labels,'probability':probability})
    if not math.isclose(total,1.,abs_tol=1e-10):raise AssertionError('Probabilities do not sum to one')
    return {'violation_probability':violation,'robust_feasible':bad_support==0,
            'positive_probability_outcomes':positive_support,'bad_outcomes':bad_outcomes}


def run():
    cfg=ScenarioConfig(slots=16,max_delay_slots=0,fleet_capacity=2,charger_capacity=2,
                       flight_multipliers=(1.,2.1),charge_durations=(2,2),probabilities=(.7,.3))
    modes=[patterns(o,cfg)[0] for o in [Order('first',0,2.),Order('second',2,2.)]]
    capacities=np.repeat(np.array([1,2,2])[:,None],cfg.slots,axis=1)
    rows=[]
    for delta in [0.,.001,.01,.05,.1,.25,.5,1.]:
        audit=exact_audit(modes,capacities,cfg.probabilities,delta)
        closed_form=cfg.probabilities[0]*cfg.probabilities[1]*delta*(2-delta)
        if not math.isclose(audit['violation_probability'],closed_form,abs_tol=1e-12):raise AssertionError('Risk formula mismatch')
        # Enumerate to verify both marginal invariance and correlation.
        probs={(a,b):outcome_probability((a,b),cfg.probabilities,delta) for a,b in itertools.product((0,1),repeat=2)}
        means=[sum(p for labels,p in probs.items() if labels[i]==1) for i in (0,1)]
        rho=(probs[(1,1)]-means[0]*means[1])/(cfg.probabilities[0]*cfg.probabilities[1])
        if not np.allclose(means,[.3,.3]):raise AssertionError('Marginals changed')
        if not math.isclose(rho,(1-delta)**2,abs_tol=1e-12):raise AssertionError('Correlation formula mismatch')
        rows.append({'delta':delta,'pair_correlation':rho,'one_flight_adverse_probability':means[0],**audit,
                     'satisfies_one_percent_joint_risk':audit['violation_probability']<=.01})
    result={'purpose':'exact_synthetic_counterexample_not_performance_claim',
            'common_probabilities':cfg.probabilities,'rows':rows,
            'conclusion':'Every positive resampling probability gives full product support. Robust feasibility then equals per-flight envelope constraints, despite arbitrarily high correlation.',
            'independent_outcome_violation_probability':rows[-1]['violation_probability']}
    out=Path(__file__).resolve().parent/'results/correlation_001'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Preserve previous diagnostics')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
