import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset1282_analysis.contracts import sha256_file
from dataset1282_analysis.xu_blood_io import case_readers
from dataset1282_analysis.xu_blood_grid import AxialViews
from dataset1282_analysis.xu_blood_supply import SliceSupply
from dataset1282_analysis.xu_targets import TargetStore, build_targets
from test_xu_targets import on_disk_phantom


class BloodReaderTests(unittest.TestCase):
    def test_real_artifacts_exact_slices_threads_and_cleanup(self):
        task_root = Path(r'D:\projects\back\t17\dataset1282-blood-io')
        task_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=task_root) as directory:
            root = Path(directory)
            cache = on_disk_phantom(root/'prepared')
            target_root = root/'targets'
            build_targets(cache, type('Args', (), {'width_mm': 1., 'reviews': None, 'resume': False})(), target_root)
            store = TargetStore(target_root, cache)
            image_path = cache._artifact('process-train_a', f'{cache._folder}/train_a.b2nd')
            target_path = target_root/'cases/train_a/blood.b2nd'
            original_hashes = (sha256_file(image_path), sha256_file(target_path))
            baseline = AxialViews(cache, 'train_a', store)
            with case_readers(cache, store, 'train_a') as (image_reader, target_reader):
                views = AxialViews(image_reader, 'train_a', target_reader)
                self.assertEqual(views.cache_shape, cache.geometry('train_a').cache_shape)
                for box in (((0, 7), (1, 13), (2, 15)), ((4, 16), (0, 9), (7, 20))):
                    expected_image = cache.read_image('train_a', box=box)
                    expected_target = store.read('train_a', 'blood', box=box)
                    np.testing.assert_array_equal(image_reader.read_image('train_a', box=box), expected_image)
                    np.testing.assert_array_equal(target_reader.read('train_a', 'blood', box=box), expected_target)
                from concurrent.futures import ThreadPoolExecutor
                boxes = [((i, i+3), (2, 11), (1, 18)) for i in range(0, 12)]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    observed = list(pool.map(lambda b: target_reader.read('train_a', 'blood', box=b), boxes))
                for box, actual in zip(boxes, observed):
                    np.testing.assert_array_equal(actual, store.read('train_a', 'blood', box=box))
                centres = np.arange(views.zyx[0])[:3]
                supply = SliceSupply(views, centres, 2, False)
                with supply.batches() as batches:
                    batches = list(batches)
                actual_images = np.concatenate([batch[1][0].numpy() for batch in batches])
                actual_labels = np.concatenate([batch[1][1].numpy() for batch in batches])
                np.testing.assert_array_equal(actual_images[:, 0], np.stack([baseline.image(int(z)) for z in centres]))
                np.testing.assert_array_equal(actual_labels, np.stack([baseline.label(int(z)) for z in centres]))
                self.assertEqual(image_reader.stats['opens'], 2)
                self.assertEqual(image_reader.stats['hash_checks'], 1)
            self.assertEqual(image_reader._arrays, {})
            self.assertEqual(original_hashes, (sha256_file(image_path), sha256_file(target_path)))
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                image_reader.read_image('train_a', box=((0, 1), (0, 1), (0, 1)))

    def test_changed_artifact_rejected(self):
        task_root = Path(r'D:\projects\back\t17\dataset1282-blood-io')
        task_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=task_root) as directory:
            root = Path(directory)
            cache = on_disk_phantom(root/'prepared')
            target_root = root/'targets'
            build_targets(cache, type('Args', (), {'width_mm': 1., 'reviews': None, 'resume': False})(), target_root)
            store = TargetStore(target_root, cache)
            path = target_root/'cases/train_a/blood.b2nd'
            saved = path.read_bytes()
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                with case_readers(cache, store, 'train_a'):
                    pass
            path.write_bytes(saved)


if __name__ == '__main__':
    unittest.main()
