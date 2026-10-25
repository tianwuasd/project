import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
from model_catalog import parse_models, parse_folds, load_selection, save_selection
from server_data import write_json


class ModelSelectionTests(unittest.TestCase):
    def test_single_multi_and_rejection(self):
        self.assertEqual(parse_models('2'), ['B1'])
        self.assertEqual(parse_models('b1,B0,b1'), ['B0', 'B1'])
        for invalid in ['', [], 'SAM', 'B0,xxx']:
            with self.assertRaises(ValueError):
                parse_models(invalid)

    def test_persisted_selection_cannot_change(self):
        with tempfile.TemporaryDirectory() as temp:
            save_selection(temp, ['ct_holdG'], ['B1'])
            self.assertEqual(load_selection(temp), {'folds': ['ct_holdG'], 'methods': ['B1']})
            with self.assertRaises(ValueError):
                save_selection(temp, ['ct_holdG'], ['B0', 'B1'])

    def test_legacy_full_default_cannot_be_silently_narrowed(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(len(load_selection(temp)['folds']), 5)
            write_json(Path(temp) / '07_experiments/formal/ct_holdG_B0_seed0/run.json', {})
            with self.assertRaises(ValueError):
                save_selection(temp, ['ct_holdG'], ['B0'])


if __name__ == '__main__':
    unittest.main()
