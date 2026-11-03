"""Partial annotations are a supervision contract, never invented background."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class PartialPreprocessingTests(unittest.TestCase):
    def cohort_rows(self):
        from chd_stunet.preprocessing.cohort import frozen_roles
        ids = [f'complete_{i:03}' for i in range(96)] + ['ct_1170']
        roles = frozen_roles(ids)
        return ([dict(case_id=c, missing_label_ids=[], roles=roles[c]) for c in ids]
                + [dict(case_id=f'partial_{i:02}', missing_label_ids=[5], roles={}) for i in range(13)])

    def test_partial_method_is_a_separate_registered_protocol(self):
        from chd_stunet.preprocessing.registry import get_method, get_adapter
        clean, partial = get_method('stunet-s'), get_method('stunet-s-partial')
        self.assertNotEqual(clean['protocol'], partial['protocol'])
        self.assertEqual(clean['expected_included'], 96)
        self.assertEqual(partial['expected_included'], 109)
        self.assertIs(get_adapter('stunet-s'), get_adapter('stunet-s-partial'))

    def test_split_keeps_main_roles_and_marks_auxiliary_train_only(self):
        from chd_stunet.preprocessing.cohort import select_cohort
        from chd_stunet.preprocessing.registry import get_method
        rows = self.cohort_rows()
        original = copy.deepcopy(rows)
        main_only, _ = select_cohort(rows, get_method('stunet-s'))
        kept, excluded = select_cohort(rows, get_method('stunet-s-partial'))
        main = [r for r in kept if r['cohort_group'] == 'main']
        aux = [r for r in kept if r['cohort_group'] == 'auxiliary']
        self.assertEqual((len(main), len(aux), len(excluded)), (96, 13, 1))
        self.assertEqual({r['case_id']: r['roles'] for r in main},
                         {r['case_id']: r['roles'] for r in main_only})
        self.assertTrue(all(r['roles'] == {'0': 'aux_train', '1': 'aux_train', '2': 'aux_train'} for r in aux))
        self.assertEqual(excluded[0]['case_id'], 'ct_1170')
        self.assertEqual(rows, original, 'Selection must not mutate the frozen audit rows')

    def test_other_missing_patterns_do_not_silently_enter_auxiliary(self):
        from chd_stunet.preprocessing.cohort import select_cohort
        from chd_stunet.preprocessing.registry import get_method
        for missing in ([1], [1, 5], [5, 7]):
            rows = self.cohort_rows()
            rows[-1]['missing_label_ids'] = missing
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'Cohort changed'):
                select_cohort(rows, get_method('stunet-s-partial'))

    def test_metadata_keeps_unknown_myo_and_background_ambiguity(self):
        from chd_stunet.preprocessing.annotations import make_annotation
        partial = make_annotation([5])
        self.assertEqual(partial['annotated_class_ids'], [1, 2, 3, 4, 6, 7])
        self.assertEqual(partial['unannotated_class_ids'], [5])
        self.assertEqual(partial['background_semantics'], 'background_or_unannotated')
        self.assertTrue(partial['requires_partial_loss'])
        self.assertFalse(make_annotation([])['requires_partial_loss'])

    def test_resampling_accepts_native_missing_myo_but_rejects_lost_known_label(self):
        from chd_stunet.preprocessing.annotations import make_annotation, validate_resampled_labels
        native = {0, 1, 2, 3, 4, 6, 7, 8}
        annotation = make_annotation([5])
        validate_resampled_labels(native, native | {-1}, annotation)
        with self.assertRaisesRegex(ValueError, 'removed target'):
            validate_resampled_labels(native, native - {7}, annotation)
        with self.assertRaisesRegex(ValueError, 'invented'):
            validate_resampled_labels(native, native | {5}, annotation)
        with self.assertRaisesRegex(ValueError, 'annotation contract'):
            validate_resampled_labels(native, native, make_annotation([]))

    def test_partial_case_roles_cannot_be_misrepresented_as_validation_or_main(self):
        from chd_stunet.preprocessing.annotations import make_annotation, validate_case_roles
        annotation = make_annotation([5])
        roles = {str(f): 'aux_train' for f in range(3)}
        validate_case_roles(annotation, 'auxiliary', roles)
        for group, wrong in [('main', roles), ('auxiliary', {}),
                             ('auxiliary', dict(roles, **{'0': 'val'}))]:
            with self.subTest(group=group, roles=wrong), self.assertRaisesRegex(ValueError,'aux_train'):
                validate_case_roles(annotation, group, wrong)

    def test_auxiliary_never_changes_intensity_or_spacing_fit(self):
        from chd_stunet.preprocessing.fingerprint import make_plan
        from chd_stunet.preprocessing.registry import get_method
        main = [dict(case_id='main', roles={'0': 'train'}, cohort_group='main',
                     spacing_zyx=[.75, .3, .3], cropped_shape_zyx=[12, 14, 16])]
        # Also defend against a mistakenly assigned ordinary train role.
        aux = dict(case_id='aux', roles={'0': 'train'}, cohort_group='auxiliary',
                   spacing_zyx=[99, 99, 99], cropped_shape_zyx=[999, 999, 999])
        samples = {'main': np.arange(100, dtype=np.float32), 'aux': np.array([1e9])}
        baseline, plan = make_plan(main, samples, 0, get_method('stunet-s'))
        combined, aux_plan = make_plan(main + [aux], samples, 0, get_method('stunet-s-partial'))
        self.assertEqual(combined['fit_ids'], ['main'])
        self.assertEqual(combined['foreground_intensity_properties_per_channel'],
                         baseline['foreground_intensity_properties_per_channel'])
        self.assertEqual(plan['target_spacing_before_transpose'], aux_plan['target_spacing_before_transpose'])


if __name__ == '__main__':
    unittest.main()
