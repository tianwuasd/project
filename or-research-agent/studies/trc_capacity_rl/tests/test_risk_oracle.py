import itertools
from fractions import Fraction
import math
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from risk_oracle import joint_capacity_risk


def brute(usage,capacities,q,delta):
    n=len(usage);risk=0.
    for labels in itertools.product((0,1),repeat=n):
        probability=sum(q[z]*math.prod((1-delta)*(y==z)+delta*q[y] for y in labels) for z in (0,1))
        load=sum((usage[i,y] for i,y in enumerate(labels)),start=np.zeros_like(capacities))
        if np.any(load>capacities):risk+=probability
    return risk


class RiskOracleTests(unittest.TestCase):
    def test_two_job_closed_form(self):
        use=np.zeros((2,2,1,8),dtype=int)
        use[0,0,0,3]=use[0,1,0,5]=use[1,0,0,5]=use[1,1,0,7]=1
        for delta in [0.,.001,.1,1.]:
            result=joint_capacity_risk(use,np.ones((1,8),dtype=int),(.7,.3),delta)
            self.assertEqual(result['status'],'exact')
            self.assertAlmostEqual(result['risk'],.21*delta*(2-delta),places=12)

    def test_matches_independent_enumeration_for_coupled_multislot_usage(self):
        rng=np.random.default_rng(814)
        for _ in range(20):
            n=int(rng.integers(1,8));use=np.zeros((n,2,2,12),dtype=int)
            for i in range(n):
                start=int(rng.integers(0,7))
                for y in (0,1):
                    end=min(12,start+int(rng.integers(1,6)))
                    use[i,y,0,start:end]=1;use[i,y,1,end-1]=1
            cap=np.repeat(np.array([3,2])[:,None],12,axis=1)
            q=(.4,.6);delta=float(rng.uniform())
            result=joint_capacity_risk(use,cap,q,delta)
            self.assertAlmostEqual(result['risk'],brute(use,cap,q,delta),places=11)

    def test_retirement_keeps_common_factor_dependence(self):
        # Each disjoint pair fails only under common outcome 1. Averaging each
        # segment first would incorrectly multiply .7*.7 rather than keep .7.
        use=np.zeros((4,2,1,4),dtype=int)
        use[0,1,0,0]=use[1,1,0,0]=1
        use[2,1,0,3]=use[3,1,0,3]=1
        result=joint_capacity_risk(use,np.ones((1,4),int),(.7,.3),0.)
        self.assertAlmostEqual(result['risk'],.3)

    def test_invariant_to_job_order_and_empty_jobs(self):
        use=np.zeros((3,2,1,4),dtype=int);use[0,1,0,1]=use[1,0,0,1]=1
        cap=np.ones((1,4),int)
        a=joint_capacity_risk(use,cap,(.7,.3),.2)
        b=joint_capacity_risk(use[[2,1,0]],cap,(.7,.3),.2)
        self.assertAlmostEqual(a['risk'],b['risk'])
        empty=joint_capacity_risk(np.zeros((0,2,1,4),int),cap,(.7,.3),.2)
        self.assertEqual(empty['risk'],0.)

    def test_state_limit_never_returns_a_false_certificate(self):
        use=np.ones((6,2,1,3),dtype=int);use[:,1,0,2]=0
        result=joint_capacity_risk(use,np.array([[6,6,2]]),(.5,.5),.5,max_states=4)
        self.assertEqual(result['status'],'state_limit')
        self.assertIsNone(result['risk'])
        self.assertEqual(result['risk_upper'],1.)

    def test_invalid_arrays_and_law_rejected(self):
        use=np.zeros((2,2,1,3));cap=np.ones((1,3))
        with self.assertRaises(ValueError):joint_capacity_risk(use,cap,(.7,.3),-.1)
        with self.assertRaises(ValueError):joint_capacity_risk(use,cap,(.7,.2),.1)
        with self.assertRaises(ValueError):joint_capacity_risk(use+np.nan,cap,(.7,.3),.1)
        with self.assertRaises(ValueError):joint_capacity_risk(use-1,cap,(.7,.3),.1)
        with self.assertRaises(ValueError):joint_capacity_risk(use+1e20,cap,(.7,.3),.1)

    def test_upper_bound_covers_rational_probability(self):
        rng=np.random.default_rng(918)
        for n in range(2,9):
            use=rng.integers(0,2,size=(n,2,2,4));cap=np.full((2,4),max(1,n//2))
            q=(.7,.3);delta=.013
            exact_q=[Fraction(v) for v in q];total=sum(exact_q);exact_q=[x/total for x in exact_q]
            d=Fraction(delta);risk=Fraction(0)
            for labels in itertools.product((0,1),repeat=n):
                load=sum((use[i,y] for i,y in enumerate(labels)),start=np.zeros_like(cap))
                if np.any(load>cap):
                    risk+=sum(exact_q[z]*math.prod((1-d)*int(y==z)+d*exact_q[y] for y in labels) for z in (0,1))
            result=joint_capacity_risk(use,cap,q,delta)
            self.assertGreaterEqual(Fraction(result['risk_upper']),risk)

if __name__=='__main__':unittest.main()
