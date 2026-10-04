import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from environment import Order
from scenario_environment import ScenarioConfig, ScenarioEnv, enumerate_optimum
from pattern_value import PatternBank
from scenario_lookahead import ConditionalLookahead


class ScenarioChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=ScenarioConfig(slots=16,fleet_capacity=2,charger_capacity=2,
            max_delay_slots=0,flight_multipliers=(1.,2.1),charge_durations=(2,2),probabilities=(.5,.5))
        self.orders=[Order('a',0,2.),Order('b',2,2.)]

    def test_strict_envelope_gap_and_exact_enumeration(self):
        joint=ScenarioEnv(self.orders,self.cfg,'joint')
        envelope=ScenarioEnv(self.orders,self.cfg,'envelope')
        joint.step(1);envelope.step(1)
        self.assertIn(1,[c['action'] for c in joint.candidates()])
        self.assertEqual([0],[c['action'] for c in envelope.candidates()])
        joint.step(1);envelope.step(0)
        self.assertTrue(joint.audit()['feasible'])
        self.assertTrue(envelope.audit()['feasible'])
        self.assertAlmostEqual(joint.reward,2*envelope.reward)
        for mode,env in [('joint',joint),('envelope',envelope)]:
            exact=enumerate_optimum(self.orders,self.cfg,mode)
            self.assertAlmostEqual(exact['objective'],env.reward)

    def test_audit_catches_forged_physics_even_if_ledger_balanced(self):
        env=ScenarioEnv(self.orders,self.cfg);env.step(1)
        env.capacity+=env.history[0]['reservation']
        env.history[0]['reservation'][:]=0
        self.assertFalse(env.audit()['feasible'])

    def test_future_suffix_does_not_change_current_actions_or_features(self):
        short=ScenarioEnv(self.orders[:1],self.cfg)
        long=ScenarioEnv(self.orders+[Order('future',10,3.)],self.cfg)
        for a,b in zip(short.candidates(),long.candidates()):
            self.assertEqual(a['action'],b['action'])
            np.testing.assert_array_equal(a['features'],b['features'])
            self.assertEqual(a['reward'],b['reward'])

    def test_completion_at_horizon_allowed(self):
        cfg=ScenarioConfig(slots=5,max_delay_slots=0,flight_multipliers=(1.,),
            charge_durations=(2,),probabilities=(1.,))
        env=ScenarioEnv([Order('last',0,1.)],cfg)
        self.assertEqual([0,1],[c['action'] for c in env.candidates()])
        env.step(1);self.assertTrue(env.audit()['feasible'])

    def test_robust_deadline_and_random_full_episode_audits(self):
        env=ScenarioEnv([Order('deadline',0,2.,deadline_minutes=8.)],self.cfg)
        self.assertEqual([0],[c['action'] for c in env.candidates()])
        rng=np.random.default_rng(17)
        for mode in ('joint','envelope'):
            for _ in range(15):
                orders=[Order(str(i),int(rng.integers(0,12)),float(rng.uniform(.1,6))) for i in range(15)]
                e=ScenarioEnv(orders,self.cfg,mode)
                while not e.done:
                    actions=e.candidates();e.step(actions[int(rng.integers(len(actions)))]['action'])
                self.assertTrue(e.audit()['feasible'])

    def test_invalid_scenarios_rejected(self):
        for changes in ({'probabilities':(.2,.2)}, {'flight_multipliers':(-1.,2.)},
                        {'charge_durations':(0,2)}, {'probabilities':(1.,)},
                        {'flight_multipliers':(float('nan'),2.)}):
            from dataclasses import replace
            with self.assertRaises(ValueError): ScenarioEnv([],replace(self.cfg,**changes))

    def test_pattern_features_monotone_and_need_all_resources(self):
        bank=PatternBank(self.cfg,distances=(2.,))
        env=ScenarioEnv([],self.cfg)
        rng=np.random.default_rng(23)
        for _ in range(15):
            low=rng.integers(0,env.initial+1)
            high=np.maximum(low,rng.integers(0,env.initial+1))
            self.assertTrue(np.all(bank.values(low,0)<=bank.values(high,0)))
        full=bank.values(env.initial,0)
        no_pad=env.initial.copy();no_pad[:,0,:]=0
        no_fleet=env.initial.copy();no_fleet[:,1,:]=0
        self.assertGreater(full.sum(),0)
        self.assertEqual(bank.values(no_pad,0).sum(),0)
        self.assertEqual(bank.values(no_fleet,0).sum(),0)
        self.assertEqual(bank.values(env.initial,self.cfg.slots-1).sum(),0)

    def test_same_slot_future_pattern_has_value(self):
        cfg=ScenarioConfig(slots=5,max_delay_slots=0,flight_multipliers=(1.,),
                           charge_durations=(2,),probabilities=(1.,))
        bank=PatternBank(cfg,distances=(1.,4.))
        env=ScenarioEnv([Order('a',0,1.),Order('b',0,4.)],cfg,feature_bank=bank)
        before=bank.values(env.initial,0)
        env.step(1)
        after=bank.values(env.capacity,0)
        self.assertGreater(before.sum(),after.sum())

    def test_same_slot_conditional_lp_uses_training_tail_only(self):
        cfg=ScenarioConfig(slots=5,max_delay_slots=0,flight_multipliers=(1.,),
                           charge_durations=(2,),probabilities=(1.,))
        training={'day':[Order('a',0,1.),Order('b',0,4.)]}
        policy=ConditionalLookahead(training,cfg,samples=1)
        current=Order('x',0,1.)
        short=ScenarioEnv([current],cfg)
        long=ScenarioEnv([current,Order('z',0,4.)],cfg)
        # Forecasted valuable second job should cause rejection, regardless of
        # the actual unknown evaluation suffix (even when there is no suffix).
        self.assertEqual(policy.choose(short,short.candidates()),0)
        self.assertEqual(policy.choose(long,long.candidates()),0)
        self.assertEqual(policy.failures,0)

if __name__=='__main__':unittest.main()
