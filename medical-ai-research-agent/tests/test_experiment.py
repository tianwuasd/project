import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/medical-ai-research/scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    import experiment
except ImportError:
    experiment = None


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(experiment, '实验模块尚未实现')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        path = self.root / 'data.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['patient_id', 'site', 'label', 'age', 'signal'])
            for i in range(90):
                for visit in range(2):
                    w.writerow([f'p{i}', ['A', 'B', 'C'][i % 3], i % 2, 20 + i % 35, (i % 2) + visit * 0.1])
        manifest = {'data_kind': 'synthetic', 'patient_column': 'patient_id', 'target': 'label',
                    'site_column': 'site', 'external_site': 'C', 'seed': 11, 'bootstrap': 40,
                    'prediction_time': 'before surgery', 'outcome': 'synthetic binary label',
                    'features': [{'name': n, 'available': 'pre_index', 'source': 'synthetic'} for n in ['age', 'signal']],
                    'feature_sets': {'age_only': ['age'], 'all': ['age', 'signal']}}
        mp = self.root / 'manifest.json'
        mp.write_text(json.dumps(manifest), encoding='utf-8')
        return path, mp, manifest

    def test_patient_isolation_external_holdout_and_actual_results(self):
        data, manifest, _ = self.fixture()
        out = self.root / 'result'
        result = experiment.run_experiment(data, manifest, out)
        split = json.loads((out / 'split.json').read_text())
        sets = [set(split[k]) for k in ['train', 'validation', 'test']]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(sets[2]), 30)
        self.assertEqual(result['data_kind'], 'synthetic')
        self.assertEqual(len(result['models']), 6)
        self.assertEqual(result['test_patients'], 30)
        self.assertTrue(all('test' in row and 'auroc_ci95' in row['test'] for row in result['models']))
        self.assertTrue((out / 'predictions.csv').exists())
        with self.assertRaises(FileExistsError):
            experiment.run_experiment(data, manifest, out)

    def test_future_features_and_id_features_are_rejected(self):
        _, _, m = self.fixture()
        m['features'][0]['available'] = 'post_outcome'
        with self.assertRaisesRegex(ValueError, 'available|时点'):
            experiment.validate_manifest(m)
        m['features'][0] = {'name': 'patient_id', 'available': 'pre_index', 'source': 'id'}
        with self.assertRaises(ValueError):
            experiment.validate_manifest(m)

    def test_cross_hospital_patient_and_single_class_rejected(self):
        ids = np.array(['a', 'a', 'b', 'c'])
        sites = np.array(['A', 'B', 'A', 'B'])
        with self.assertRaisesRegex(ValueError, '中心|site'):
            experiment.make_split(ids, np.array([0, 0, 1, 1]), sites, 'B', 1)
        with self.assertRaises(ValueError):
            experiment.make_split(np.array(['a', 'b', 'c']), np.array([0, 0, 0]), np.array(['A']*3), None, 1)

    def test_inconsistent_patient_outcomes_rejected(self):
        with self.assertRaisesRegex(ValueError, '结局|outcome'):
            experiment.make_split(np.array(['a', 'a', 'b']), np.array([0, 1, 1]), np.array(['A']*3), None, 1)

    def test_duplicate_header_cannot_replace_patient_identity(self):
        data, manifest, _ = self.fixture()
        lines = data.read_text(encoding='utf-8').splitlines()
        lines[0] += ',patient_id'
        for i in range(1, len(lines)):
            lines[i] += f',fake_row_{i}'
        data.write_text('\n'.join(lines), encoding='utf-8')
        out = self.root / 'bad_result'
        with self.assertRaisesRegex(ValueError, '重复|duplicate'):
            experiment.run_experiment(data, manifest, out)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
