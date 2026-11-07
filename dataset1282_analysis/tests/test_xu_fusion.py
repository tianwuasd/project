import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dataset1282_analysis.xu_fusion import blood_grow, grid_prediction, vote

MAPPING = dict(LV=1, RV=2, LA=3, RA=4, MYO=5, AO=6, PA=7, PV=8, SVC=9, IVC=10)


class FusionTests(unittest.TestCase):
    def test_single_class_blood_head_never_fills_or_erases_entire_roi(self):
        labels = np.zeros((16, 16, 16), np.uint8)
        labels[4:8, 4:8, 4:8] = 6
        labels[5:7, 5:7, 5:7] = 5
        for sole_class in (0, 1, 2):
            with self.subTest(sole_class=sole_class):
                collapsed = np.full(labels.shape, sole_class, dtype=np.uint8)
                result, qc = blood_grow(labels, collapsed, 5)
                np.testing.assert_array_equal(result, labels)
                self.assertEqual(qc["status"], "abstained_missing_blood_class")
                self.assertEqual(qc["changed"], 0)

    def test_whole_volume_localizer_is_cropped_at_target_centres(self):
        labels = np.broadcast_to(np.arange(4)[:, None, None], (4, 3, 2)).astype(
            np.uint8
        )
        box = ((1, 5), (1, 3), (0, 2))
        sampled = grid_prediction(labels, (8, 4, 2), None, box, (4, 2, 1))
        np.testing.assert_array_equal(sampled[:, 0, 0], [0, 1, 1, 2])
        with self.assertRaises(ValueError):
            grid_prediction(labels, (8, 4, 2), box, None, (8, 4, 2))

    def test_votes_and_abstention_on_heterogeneous_auxiliary_classes(self):
        def v(values):
            return np.array(values, "u1").reshape(-1, 1, 1)

        preds = dict(
            crop64=v([6, 9, 2, 4]),
            crop128=v([6, 9, 2, 3]),
            all64=v([1, 9, 2, 3]),
            all128=v([1, 9, 2, 4]),
        )
        baseline, stats = vote(preds)
        self.assertEqual(baseline.ravel().tolist(), [1, 9, 2, 4])
        preds.update(init64=v([6, 0, 6, 0]), init128=v([6, 0, 6, 0]))
        labels, stats = vote(preds, MAPPING)
        # Proximal boosts AO when fine predictors support it. Distal PV/SVC can
        # remain despite both proximal models predicting coarse background.
        self.assertEqual(labels.ravel().tolist(), [6, 9, 2, 4])
        self.assertEqual(stats["proximal_abstaining_votes"], 2)
        del preds["init128"]
        with self.assertRaises(ValueError):
            vote(preds, MAPPING)

    def test_boundary_and_myocardium_block_growing(self):
        samples = np.zeros((3, 3, 9), "u1")
        samples[1, 1, 0] = 6
        samples[1, 1, 8] = 7
        samples[1, 1, 2] = 5
        blood = np.zeros_like(samples)
        blood[1, 1, :] = 1
        blood[1, 1, 5] = 2
        labels, qc = blood_grow(samples, blood, 5)
        self.assertEqual(labels[1, 1].tolist(), [6, 6, 5, 0, 0, 0, 7, 7, 7])
        self.assertGreater(qc["unseeded_blood"], 0)
        no_seed, detail = blood_grow(samples, np.zeros_like(samples), 5)
        np.testing.assert_array_equal(samples, no_seed)

    def test_vessel_radius_graph_relabels_small_supported_fragment(self):
        from dataset1282_analysis.xu_vessels import refine_vessels

        array = np.zeros((130, 20, 20), "u1")
        array[5:125, 2:7, 2:7] = 6
        array[122:125, 2:7, 2:7] = 7
        array[20:100, 12:17, 12:17] = 7
        result, qc = refine_vessels(array, (0.7, 0.8, 1.1), MAPPING)
        self.assertEqual(qc["status"], "applied")
        self.assertTrue(np.all(result[122:125, 2:7, 2:7] == 6))
        self.assertTrue(qc["vessels"]["AO"]["paths"][0]["capacity_mm"] > 0)
        self.assertEqual(qc["landmark_orientation"], "unavailable_local_array_only")

    def test_touching_principal_vessels_and_true_absence_are_preserved(self):
        from dataset1282_analysis.xu_vessels import refine_vessels

        arr = np.zeros((7, 5, 5), "u1")
        arr[1:4, 2, 2] = 6
        arr[4:6, 2, 2] = 7
        for source, status in (
            (arr, "abstained_anchors_connected"),
            (arr * (arr != 7), "abstained_missing_or_fragmented_anchors"),
        ):
            result, report = refine_vessels(source, (1, 1, 1), MAPPING)
            self.assertEqual(report["status"], status)
            np.testing.assert_array_equal(result, source)

    def test_enclosed_tiny_islands_change_only_with_unambiguous_contact(self):
        from dataset1282_analysis.xu_fusion import _enclosed_islands

        source = np.ones((20, 20, 20), np.uint8)
        source[0:10, :, :] = 3
        source[15, 15, 15] = 3
        source[10, 8, 8] = 3
        blood = np.ones_like(source)
        result, report = _enclosed_islands(source, blood, 5)
        self.assertEqual(report["reassigned"], 1)
        self.assertEqual(result[15, 15, 15], 1)
        self.assertEqual(result[10, 8, 8], 3)
        blood[15, 15, 14] = 2
        result, report = _enclosed_islands(source, blood, 5)
        self.assertEqual(result[15, 15, 15], 3)

    def test_committed_roi_identity_and_exported_bytes_are_rechecked(self):
        from dataset1282_analysis.contracts import read_json, atomic_json, sha256_file
        from dataset1282_analysis.xu_fusion import _finish, _completed

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "cases").mkdir()
            a = p / "cases" / "toy.npy"
            b = p / "cases" / "toy.nii.gz"
            a.write_bytes(b"roi")
            b.write_bytes(b"native")
            job = dict(command="fuse", request="request", evaluate=False)
            records = {
                "toy": dict(
                    box=[[2, 4]] * 3,
                    label_sha256=sha256_file(a),
                    native_sha256=sha256_file(b),
                )
            }
            _finish(p, SimpleNamespace(request_digest="request"), job, records, False)
            _completed(p, job, ("toy",), True)
            a.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "prediction changed"):
                _completed(p, job, ("toy",), True)
            marker = read_json(p / "complete.json")
            marker["cases"]["toy"]["box"] = [[0, 10]] * 3
            atomic_json(p / "complete.json", marker)
            with self.assertRaisesRegex(ValueError, "prediction set"):
                _completed(p, job, ("toy",), False)


class SharedPredictionTests(unittest.TestCase):
    def setUp(self):
        from dataset1282_analysis.contracts import atomic_json, sha256_file
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.shared = self.root/'shared'
        self.identity = dict(branch='crop64', checkpoint='weights', cases=['toy'])
        self.calls = 0

        def produce(path):
            self.calls += 1
            path.mkdir()
            np.save(path/'toy.npy', np.arange(8, dtype='uint8').reshape(2,2,2))
            files = {'toy': sha256_file(path/'toy.npy')}
            atomic_json(path/'complete.json', dict(checkpoint=self.identity['checkpoint'], files=files))
            return files
        self.produce = produce

    def run_branch(self, destination, producer=None):
        from dataset1282_analysis.xu_fusion import _shared_branch
        return _shared_branch(self.shared, self.identity, self.root/destination, ['toy'],
                              producer or self.produce)

    def test_identical_inputs_reuse_labels_and_changed_inputs_predict_again(self):
        from dataset1282_analysis.contracts import sha256_file
        first, miss = self.run_branch('first')
        with patch('dataset1282_analysis.xu_fusion.os.link', side_effect=OSError('different disk')):
            second, hit = self.run_branch('second')
        self.assertEqual((miss['outcome'], hit['outcome'], self.calls), ('miss','hit',1))
        self.assertEqual((first, miss['key']), (second, hit['key']))
        self.assertEqual(first['toy'], sha256_file(self.root/'second/toy.npy'))
        self.identity['geometry'] = 'another-grid'
        _, distinct = self.run_branch('third')
        self.assertNotEqual(distinct['key'], hit['key'])
        self.assertEqual((distinct['outcome'], self.calls), ('miss',2))

    def test_active_writer_and_interrupted_publish_are_safe(self):
        from dataset1282_analysis.contracts import digest
        from dataset1282_analysis.observability.status import operation_lock
        key = digest(self.identity)
        with operation_lock(self.shared/'locks'/key, 'owner'):
            with self.assertRaisesRegex(RuntimeError, 'lock|active'):
                self.run_branch('overlap')
        def interrupted(path):
            self.produce(path)
            raise RuntimeError('injected failure')
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            self.run_branch('first', interrupted)
        self.assertFalse((self.shared/'entries'/key).exists())
        self.assertFalse((self.root/'first/complete.json').exists())
        self.assertEqual(self.run_branch('second')[1]['outcome'], 'miss')

    def test_changed_shared_bytes_and_manifest_are_rejected(self):
        from dataset1282_analysis.contracts import atomic_json, read_json
        _, info = self.run_branch('first')
        entry = self.shared/'entries'/info['key']
        marker = entry/'manifest.json'
        original = read_json(marker)
        atomic_json(marker, dict(original, digest='changed'))
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.run_branch('second')
        atomic_json(marker, original)
        (entry/'toy.npy').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Changed shared'):
            self.run_branch('second')
        self.assertEqual(self.calls, 1)
