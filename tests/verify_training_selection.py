import unittest

from chd_segmentation.training.stopping import resolve_budget, selection_options, update_selection
from chd_segmentation.models.roles import technical_names
from chd_segmentation.models.medsam2.intensity import resolve_intensity
from chd_segmentation.training.attempts import is_technical, remaining_attempts, claim_attempt, finish_attempt


class TrainingSelection(unittest.TestCase):
    def test_formal_stu_technical_settings_do_not_require_tool(self):
        config = dict(model='stunet', scope='formal', technical=dict(roi_xyz=None, allow_role_overlap=False))
        self.assertFalse(is_technical(config))
        self.assertIsNone(remaining_attempts(config, 'train'))
        self.assertIsNone(claim_attempt(config, 'train'))
        self.assertIsNone(finish_attempt(config, 'train', None))
        self.assertTrue(is_technical(dict(config, scope='technical')))
        self.assertTrue(is_technical(dict(model='q_route1', technical=True)))
    def test_model_defaults(self):
        self.assertEqual(resolve_budget({}, 'q_route1', 'localizer')['updates'], 5000)
        self.assertEqual(resolve_budget({}, 'medsam2', 'sequence')['epochs'], 75)
        self.assertEqual(selection_options({}, 'stunet')['patience'], 20)

    def test_best_and_patience_are_independent(self):
        options = selection_options({'enabled': True, 'patience': 1}, 'q_route1')
        first = update_selection(None, 1., 1000, options)
        second = update_selection(first, .99999, 2000, options)
        self.assertTrue(second['improved'])
        self.assertTrue(second['stop'])
        self.assertEqual(second['best_position'], 2000)
        self.assertEqual(second['early_stop_reference'], 1.)

    def test_q_epoch_requires_explicit_conversion(self):
        with self.assertRaisesRegex(ValueError, 'updates_per_epoch'):
            resolve_budget({'epochs': 2}, 'q_route1', 'base')
        self.assertEqual(resolve_budget({'epochs': 2, 'updates_per_epoch': 5},
                                       'q_route1', 'base')['updates'], 10)

    def test_technical_roles_do_not_train_validation(self):
        roles = {'base': dict(train=['ct_1001'], validation=['ct_1012'], test=[])}
        self.assertEqual(technical_names(roles, 'base', 'train', ['ct_1001', 'ct_1012']), ['ct_1001'])
        with self.assertRaisesRegex(ValueError, '角色'):
            technical_names({}, 'base', 'train', ['ct_1001', 'ct_1012'])

    def test_medsam_auto_requires_known_intensity(self):
        with self.assertRaisesRegex(ValueError, 'intensity_semantics'):
            resolve_intensity({}, {})
        actual = resolve_intensity({'intensity_mode': 'normalized_percentile'}, {})
        self.assertEqual(actual['mode'], 'normalized_percentile')
        known = resolve_intensity({}, dict(semantics='windowed_uint8', gray_window=[-100., 400.]))
        self.assertEqual(known['mode'], 'windowed_uint8')


if __name__ == '__main__':
    unittest.main()
