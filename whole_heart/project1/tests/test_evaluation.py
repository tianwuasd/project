"""用微型三维数组检查评价约定，无需 GPU 或真实病例。"""
import sys
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
from evaluation import largest_components, score_case, require_completed
from evaluation_summary import paired_summary, summarize
import evaluation as ev
from server_data import write_json, read_json, digest


class EvaluationTests(unittest.TestCase):
    def test_summary_writes_reports_and_keeps_infinite_distances(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / '08_results/evaluation'
            models = [dict(fold=fold, method=method, targets=['Case0001'])
                      for fold in ['ct_holdG', 'mr_holdE'] for method in ['B0', 'B1']]
            write_json(folder / 'frozen_models.json', dict(models=models))
            sha = digest(folder / 'frozen_models.json')
            prediction = root / 'synthetic_output.bin'
            prediction.write_bytes(b'synthetic')
            for model in models:
                score = dict(macro_dice=.5 if model['method'] == 'B0' else .7,
                             both_empty_classes=0, one_empty_classes=1, defined_classes=7,
                             classes={name: dict(dice=.5, hd_mm='inf', hd95_mm='inf', status='one_empty')
                                      for name in ['Myo', 'LA', 'LV', 'RA', 'RV', 'AO', 'PA']})
                write_json(folder / f"{model['fold']}_{model['method']}/Case0001.json",
                           dict(status='completed', signature=dict(frozen_sha256=sha),
                                modality='CT' if model['fold'].startswith('ct') else 'MRI',
                                outputs=dict(raw=dict(path=str(prediction), sha256=digest(prediction))),
                                results=dict(raw=score, lcc=score)))
            summarize(root)
            report = root / '09_reports/final_evaluation'
            result = read_json(report / 'summary.json')['results']['CT_raw']
            self.assertEqual(result['methods']['B0']['pooled_class_hd_mm']['mean'], 'inf')
            self.assertAlmostEqual(result['paired']['mean_delta'], .2)
            self.assertTrue((report / 'README.md').exists())
            self.assertEqual(len((report / 'class_metrics.csv').read_text(encoding='utf-8-sig').splitlines()), 57)

    def test_no_checkpoint_recovery_preserves_other_runs_and_archive(self):
        import server_train
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / '07_experiments/formal/ct_holdG_B1_seed0'
            write_json(out / 'run.json', dict(status='failed'))
            other = root / '07_experiments/formal/ct_holdG_B0_seed0/fold_0/checkpoint_final.pth'
            write_json(other, {})
            with patch.object(server_train, 'ROOT', root):
                archive = Path(server_train.archive_uncheckpointed(out))
                self.assertTrue((archive / 'run.json').exists())
                self.assertFalse(out.exists())
                self.assertTrue(other.exists())
                with self.assertRaises(ValueError):
                    server_train.archive_uncheckpointed(other.parents[1])
                write_json(out / 'run.json', dict(status='failed'))
                write_json(root / '08_results/evaluation/frozen_models.json', {})
                with self.assertRaises(ValueError):
                    server_train.archive_uncheckpointed(out)

    def test_all_ten_required_and_frozen_checkpoint_cannot_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / '05_src').mkdir()
            write_json(root / '06_configs/pip_freeze_training.txt', {})
            write_json(root / '06_configs/conda_explicit_training.txt', {})
            splits = {fold: dict(dataset_id=701+i, target=['Case0001']) for i, fold in enumerate(ev.FOLDS)}
            write_json(root / '04_data/manifests/splits_v1.json', dict(folds=splits))
            for fold, split in splits.items():
                prep = root / '04_data/derived/nnUNet_preprocessed' / f"Dataset{split['dataset_id']:03d}_{fold}"
                write_json(prep / 'Project1Plans.json', {})
                for method in ['B0', 'B1']:
                    out = root / '07_experiments/formal' / f'{fold}_{method}_seed0'
                    signature = dict(fold=fold, method=method, seed=0,
                                     plan_sha256=digest(prep / 'Project1Plans.json'),
                                     splits_sha256=digest(root / '04_data/manifests/splits_v1.json'),
                                     environment_sha256=digest(root / '06_configs/pip_freeze_training.txt'),
                                     conda_environment_sha256=digest(root / '06_configs/conda_explicit_training.txt'),
                                     code_sha256={})
                    write_json(out / 'run.json', dict(status='completed', formal_training=True,
                                                      signature=signature, split=split, initial_weight_sha256='same'))
                    write_json(out / 'fold_0/checkpoint_best.pth', {})
                    write_json(out / 'fold_0/checkpoint_final.pth', {})
            with patch.multiple(ev, ROOT=root, CODE_ROOT=root):
                last = out / 'run.json'
                complete = read_json(last)
                write_json(last, dict(complete, status='running'))
                with self.assertRaises(ValueError):
                    ev.freeze_models()
                self.assertFalse((root / '08_results/evaluation/frozen_models.json').exists())
                write_json(last, complete)
                models, sha = ev.freeze_models()
                self.assertEqual(len(models), 10)
                self.assertEqual(ev.freeze_models()[1], sha)
                write_json(out / 'fold_0/checkpoint_best.pth', {'changed': True})
                with self.assertRaises(ValueError):
                    ev.freeze_models()

    def test_native_prediction_resume_and_corruption_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / 'image.nii.gz'
            reference = root / 'label.nii.gz'
            shape, affine = (6, 7, 8), np.diag([1.2, 1.5, 2., 1.])
            labels = np.zeros(shape, dtype=np.uint16)
            labels[1:3, 2:4, 3:5] = 850
            nib.save(nib.Nifti1Image(np.ones(shape), affine), image)
            nib.save(nib.Nifti1Image(labels, affine), reference)
            case = dict(case_id='Case0001', modality='CT', image=str(image), label=str(reference),
                        image_sha256=digest(image), label_sha256=digest(reference))
            write_json(root / '04_data/manifests/case_inventory.json', dict(cases=[case]))
            # 模拟nnU-Net读写器的轴转换，检验不能直接把内部数组当原图坐标。
            class Reader:
                def read_images(self, paths):
                    return np.ones((1, 8, 7, 6)), {}
                def write_seg(self, seg, path, properties):
                    nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0).astype(np.uint8), affine), path)
            class Predictor:
                def predict_single_npy_array(self, *args):
                    return ev.map_labels(labels).transpose(2, 1, 0)
            model = dict(fold='ct_holdG', method='B0', targets=['Case0001'])
            with patch.object(ev, 'ROOT', root), patch.object(ev, 'make_predictor', return_value=(Predictor(), Reader())) as factory:
                ev.evaluate_model(model, 'frozen')
                ev.evaluate_model(model, 'frozen')
                self.assertEqual(factory.call_count, 1)
                record = read_json(root / '08_results/evaluation/ct_holdG_B0/Case0001.json')
                self.assertEqual(record['results']['raw']['macro_dice'], 1)
                saved = nib.load(record['outputs']['raw']['path'])
                self.assertTrue(np.array_equal(np.asanyarray(saved.dataobj), labels))
                Path(record['outputs']['raw']['path']).write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    ev.evaluate_model(model, 'frozen')

    def test_lcc_is_per_class_and_26_connected(self):
        a = np.zeros((6, 6, 6), dtype=np.uint8)
        a[0, 0, 0] = a[1, 1, 1] = a[5, 5, 5] = 1
        a[4, 0, 0] = 2
        result = largest_components(a)
        self.assertEqual(int((result == 1).sum()), 2)
        self.assertEqual(int((result == 2).sum()), 1)
        self.assertEqual(a[5, 5, 5], 1)

    def test_empty_failure_is_not_dropped(self):
        a = np.zeros((4, 4, 4), dtype=np.uint8)
        b = a.copy()
        b[1, 1, 1] = 1
        scores = score_case(a, b, np.eye(4))
        self.assertEqual(scores['classes']['Myo']['hd_mm'], 'inf')
        self.assertEqual(scores['macro_dice'], 0)
        self.assertEqual(scores['defined_classes'], 1)
        self.assertEqual(scores['both_empty_classes'], 6)

    def test_incomplete_model_blocks_target_evaluation(self):
        with self.assertRaises(ValueError):
            require_completed({'status': 'running', 'formal_training': True})

    def test_bootstrap_weights_centers_equally(self):
        rows = []
        for fold, n, delta in [('A', 2, .1), ('B', 8, .3)]:
            for i in range(n):
                for method in ['B0', 'B1']:
                    rows.append(dict(fold=fold, case_id=str(i), method=method,
                                     macro_dice=.5 + (delta if method == 'B1' else 0)))
        result = paired_summary(rows, samples=50)
        self.assertAlmostEqual(result['mean_delta'], .2)
        self.assertTrue(np.allclose(result['ci95'], [.2, .2]))
        with self.assertRaises(ValueError):
            paired_summary(rows[:-1], samples=50)


if __name__ == '__main__':
    unittest.main()
