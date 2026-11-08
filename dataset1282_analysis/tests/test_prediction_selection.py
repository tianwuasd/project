import unittest
from pathlib import Path

from dataset1282_analysis.prediction.heldout import GROUPS, select_cases
from dataset1282_analysis.prediction.source import safe_path


def cohort(rows):
    return {'cases': rows}


def row(role, patient, *, included=True, index_scan=True):
    return {'role': role, 'patient': patient, 'included': included,
            'index_scan': index_scan}


class PredictionSelectionTests(unittest.TestCase):
    def test_selects_each_frozen_heldout_group(self):
        for group in GROUPS:
            with self.subTest(group=group):
                cases = cohort({'held': row(group, f'p-{group}')})
                self.assertEqual(select_cases(cases, [group]),
                                 {'held': cases['cases']['held']})

    def test_rejects_training_and_development_groups(self):
        for role in ('train', 'development'):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError, 'Prediction admits only frozen held-out groups'):
                    select_cases(cohort({'leak': row(role, 'p1')}), [role])

    def test_rejects_patient_shared_with_training_or_development(self):
        cases = cohort({'held': row('source_test', 'p1'),
                        'train': row('train', 'p1')})
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            select_cases(cases, ['source_test'])

    def test_rejects_duplicate_patient_in_selected_groups(self):
        cases = cohort({'a': row('target_primary', 'p1'),
                        'b': row('target_secondary', 'p1')})
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            select_cases(cases, ['target_primary', 'target_secondary'])

    def test_excludes_nonincluded_and_nonindex_cases(self):
        for flags in ({'included': False}, {'index_scan': False}):
            with self.subTest(flags=flags):
                cases = cohort({'held': row('source_test', 'p1', **flags)})
                with self.assertRaisesRegex(ValueError, 'Select unique'):
                    select_cases(cases, ['source_test'])

    def test_rejects_duplicate_case_selection(self):
        cases = cohort({'held': row('source_test', 'p1')})
        with self.assertRaisesRegex(ValueError, 'Select unique'):
            select_cases(cases, ['source_test'], ['held', 'held'])

    def test_safe_path_rejects_parent_escape(self):
        with self.assertRaisesRegex(ValueError, 'Linked or escaping'):
            safe_path(Path(__file__).parent, '../outside.json')


if __name__ == '__main__':
    unittest.main()
