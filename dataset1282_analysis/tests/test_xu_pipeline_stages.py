from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset1282_analysis import xu_pipeline_stages as stages
from dataset1282_analysis.xu_workflow import main as xu_main


class ParsedStage(Exception):
    """Signals that the single-stage command accepted its complete argument list."""


def options(**overrides):
    fields = dict(prepared=Path('prepared'), profile='smoke', device='cpu', supervision='marginal',
                  technical_subset=True, seed=17, crop64=None, crop128=None, reviews=None)
    return SimpleNamespace(**(fields | overrides))


def schedule(init, blood, **overrides):
    counts = {kind: dict(train=t, development=d) for kind, (t, d) in [('init', init), ('blood', blood)]}
    with patch.object(stages, 'auxiliary_counts', return_value=counts):
        sequence = list(stages.iter_stages(object(), options(**overrides), Path('run')))
    return {step.name: step for step in sequence}


def value_of(step, flag):
    return step.argv[step.argv.index(flag)+1]


class StageTests(unittest.TestCase):
    def test_complete_schedule_routes_to_existing_commands_in_dependency_order(self):
        plan = schedule((1, 1), (1, 1), reviews=Path('reviewed.json'))
        self.assertEqual(list(plan), [
            'crop64', 'crop128', 'localize', 'audit-roi', 'all64', 'all128',
            'evaluate-all64', 'evaluate-all128', 'four-way', 'targets', 'init64', 'init128',
            'six-way', 'blood', 'with-blood', 'vessels'])
        with patch('dataset1282_analysis.xu_geometry.PreparedCache', side_effect=ParsedStage):
            for name, step in plan.items():
                with self.subTest(name=name):
                    self.assertIsNone(step.skip_reason)
                    self.assertTrue(step.required)
                    self.assertEqual(Path(value_of(step, '--output')), Path('run')/name)
                    with self.assertRaises(ParsedStage):
                        xu_main(list(step.argv))
        self.assertEqual(value_of(plan['localize'], '--padding-percent'), '20')
        self.assertEqual(value_of(plan['blood'], '--supervision'), 'marginal')
        self.assertEqual(Path(value_of(plan['all64'], '--boxes')), Path('run/audit-roi/boxes.json'))
        self.assertEqual(Path(value_of(plan['targets'], '--reviews')), Path('reviewed.json'))
        self.assertNotIn('--blood', plan['six-way'].argv)
        for flag in ('--init64', '--init128', '--blood', '--evaluate'):
            self.assertIn(flag, plan['with-blood'].argv)
        self.assertEqual(Path(value_of(plan['vessels'], '--fusion')), Path('run/with-blood'))

    def test_insufficient_targets_select_only_supported_models(self):
        for init, blood, last in [((3, 1), (2, 2), 'with-blood'),
                                  ((4, 1), (2, 0), 'six-way'),
                                  ((0, 1), (0, 0), 'four-way')]:
            with self.subTest(init=init, blood=blood):
                plan = schedule(init, blood, profile='paper', technical_subset=False)
                self.assertIsNone(plan['four-way'].skip_reason)
                self.assertEqual(Path(value_of(plan['vessels'], '--fusion')), Path('run')/last)
                if init[0] < 4:
                    self.assertEqual(plan['init64'].skip_reason['eligible'],
                                     {'train': init[0], 'development': init[1]})
                    self.assertEqual(plan['six-way'].skip_reason, plan['init128'].skip_reason)
                    self.assertNotIn('--init64', plan['with-blood'].argv)
                if blood[1] == 0:
                    self.assertEqual(plan['blood'].skip_reason, plan['with-blood'].skip_reason)
                    self.assertEqual(plan['blood'].skip_reason['required'], dict(train=1, development=1))

    def test_frozen_locators_must_be_paired_and_are_used_for_voting_and_fusion(self):
        with self.assertRaisesRegex(ValueError, 'pair'):
            next(stages.iter_stages(object(), options(crop64=Path('one.pt')), Path('run')))
        plan = schedule((1, 1), (1, 1), crop64=Path('small.pt'), crop128=Path('large.pt'))
        for branch, filename in [('crop64', 'small.pt'), ('crop128', 'large.pt')]:
            self.assertEqual(plan[branch].skip_reason['reason'], 'supplied_frozen_locator')
            for stage in ('localize', 'four-way', 'six-way', 'with-blood'):
                self.assertEqual(value_of(plan[stage], '--'+branch), filename)

    def test_corrupt_targets_are_errors_and_complete_arm_filters_both_auxiliaries(self):
        class Cache:
            supervision = {
                'a': dict(present_classes=list(range(1,11)), known_absent_classes=[], unknown_classes=[]),
                'b': dict(present_classes=list(range(1,10)), known_absent_classes=[], unknown_classes=[10]),
                'v': dict(present_classes=list(range(1,11)), known_absent_classes=[], unknown_classes=[])}

            def case_ids(self, role, complete_only=False):
                if role == 'development':
                    return ('v',)
                return ('a',) if complete_only else ('a', 'b')

        with (patch.object(stages, 'TargetStore') as store,
              patch.object(stages, 'blood_ids', return_value=(1,))):
            store.return_value.has.return_value = True
            counts = stages.auxiliary_counts(Cache(), 'targets', 'complete')
            self.assertEqual(counts, dict(init=dict(train=1, development=1),
                                          blood=dict(train=1, development=1)))
            self.assertEqual(stages.auxiliary_counts(Cache(),'targets','marginal')['blood']['train'],2)
            store.return_value.has.return_value = False
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                stages.auxiliary_counts(Cache(), 'targets', 'marginal')
