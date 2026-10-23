"""Synthetic tests for resource controls, automatic geometry and durable jobs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

try:
    import nibabel as nib
    import numpy as np
    import scipy
    import psutil
except ImportError as exc:
    raise unittest.SkipTest('Install requirements-preprocess.txt') from exc

from aaoca_preprocessing.config import Config
from aaoca_preprocessing.image import inspect_image
from aaoca_preprocessing.pipeline import plan, run, suggest_spacing
from aaoca_preprocessing.resources import allocation
from aaoca_preprocessing import jobs
from aaoca_preprocessing.benchmark import benchmark


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'input'
        self.src.mkdir()

    def image(self, name='a.nii.gz', spacing=(.6, .6, .25), shape=(11, 13, 17)):
        x, y, z = np.indices(shape)
        data = (300*x + 50*y - 80*z - 1000).astype(np.float32)
        image = nib.Nifti1Image(data, np.diag([*spacing, 1]))
        image.header.set_xyzt_units('mm')
        nib.save(image, self.src / name)
        return self.src / name

    def test_automatic_spacing_uses_dataset_median_and_no_intensity_fit(self):
        self.image('a.nii.gz', (.5, .5, .25))
        self.image('b.nii.gz', (.6, .6, .3))
        self.image('c.nii.gz', (.7, .7, .25))
        result = suggest_spacing(self.src)
        np.testing.assert_allclose(result['spacing_mm'], [.6, .6, .25])
        self.assertEqual(result['source_count'], 3)
        self.assertEqual(result['shape_basis'], 'full_FOV_not_cropped')
        self.assertFalse(result['anisotropy_adjusted'])

    def test_plan_reports_downsampling_and_reversible_coordinate_mapping(self):
        self.image()
        row = plan(self.src, Config(target_spacing=(.5, .5, .5)))['cases'][0]
        self.assertEqual(row['downsample_axes_RAS'], ['S'])
        np.testing.assert_allclose(np.array(row['original_index_to_output_index']) @
                                   np.array(row['output_index_to_original_index']), np.eye(4), atol=1e-8)

    def test_threaded_slabs_and_compression_levels_preserve_pixels(self):
        self.image()
        arrays = []
        for threads, level in [(1, 6), (2, 1), (3, 3)]:
            output = self.root / f'output{threads}'
            result = run(self.src, output, Config(target_spacing=(.5, .5, .5),
                         compute_threads=threads, compression_level=level, chunk_depth=3))
            self.assertEqual(result['completed'], 1, result)
            record = json.loads(next((output / 'cases').glob('*/record.json')).read_text())
            self.assertIn('decode_load', record['timings_seconds'])
            self.assertIn('save', record['timings_seconds'])
            arrays.append(nib.load(output / record['output']).get_fdata())
        for array in arrays[1:]:
            np.testing.assert_array_equal(array, arrays[0])

    def test_input_gzip_is_decompressed_once(self):
        source = self.image()
        import gzip
        original = gzip.open
        calls = []
        def counted(path, *args, **kwargs):
            if Path(path) == source:
                calls.append(path)
            return original(path, *args, **kwargs)
        with patch('aaoca_preprocessing.image.gzip.open', side_effect=counted):
            result = run(self.src, self.root / 'output', Config())
        self.assertEqual(result['completed'], 1)
        self.assertEqual(len(calls), 1)

    def test_cooperative_stop_preserves_pending_work_and_can_resume(self):
        self.image('a.nii.gz'); self.image('b.nii.gz')
        stopped = threading.Event()
        def event(case_id, stage, **fields):
            if stage == 'compute':
                stopped.set()
        output = self.root / 'output'
        result = run(self.src, output, Config(chunk_depth=1), stop=stopped.is_set, event=event)
        self.assertEqual(result['state'], 'stopped')
        self.assertGreater(result['pending'], 0)
        self.assertEqual(result['failed'], 0)
        self.assertEqual(run(self.src, output, Config(chunk_depth=1))['completed'], 2)

    def test_memory_queue_admits_cases_by_sum_not_maximum_times_workers(self):
        for i in range(3): self.image(f'{i}.nii.gz')
        info = inspect_image(self.src / '0.nii.gz', Config())
        budget = info['estimated_working_bytes'] * 1.2 / 2**30
        result = run(self.src, self.root / 'output', Config(workers=3, memory_budget_gib=budget))
        self.assertEqual(result['completed'], 3)
        self.assertEqual(result['resources']['max_active_cases'], 1)

    def test_cgroup_parent_limits_cpu_and_memory(self):
        group = self.root / 'cgroup'; (group / 'team/job').mkdir(parents=True)
        membership = self.root / 'membership'; membership.write_text('0::/team/job\n')
        (group / 'team/cpu.max').write_text('200000 100000')
        (group / 'team/memory.max').write_text(str(1024**3))
        (group / 'team/memory.current').write_text(str(256*1024**2))
        report = allocation(Config(workers=8), group, membership)
        self.assertLessEqual(report['cpu_budget'], 2)
        self.assertLessEqual(report['available_memory_bytes'], 768*1024**2)
        self.assertLessEqual(report['case_slots'], 2)

    def test_combined_prepare_requires_explicit_spacing(self):
        self.image()
        entry = Path(__file__).resolve().parents[1] / 'preprocess_aaoca.py'
        result = subprocess.run([sys.executable, str(entry), 'prepare', '--input', str(self.src),
                                 '--output', str(self.root / 'output')], capture_output=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)

    def test_resource_settings_validate(self):
        for kwargs in [{'read_workers': 0}, {'compute_threads': 0}, {'compression_level': 10},
                       {'memory_budget_gib': -1}]:
            with self.assertRaises(ValueError): Config(**kwargs)

    def test_anisotropy_adjustment_uses_tenth_percentile_only_when_both_conditions_hold(self):
        for index, z in enumerate([2, 5, 8]):
            self.image(f'{index}.nii.gz', (.5, .5, z), (30, 30, 3))
        result = suggest_spacing(self.src)
        self.assertTrue(result['anisotropy_adjusted'])
        np.testing.assert_allclose(result['spacing_mm'], [.5, .5, 2.6])

    def test_auto_prepare_records_selection_and_uses_fixed_hu(self):
        self.image()
        entry = Path(__file__).resolve().parents[1] / 'preprocess_aaoca.py'
        output = self.root / 'auto'
        result = subprocess.run([sys.executable, str(entry), 'prepare', '--input', str(self.src),
                                 '--output', str(output), '--spacing', 'auto'], capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((output / 'dataset.json').read_text())
        np.testing.assert_allclose(record['config']['nominal_spacing_mm'], [.6, .6, .25])
        self.assertEqual(record['spacing_selection']['source_count'], 1)
        self.assertEqual(record['config']['formula'], 'clip(HU, -1000, 1000) / 1000')

    def wait_job(self, output):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = jobs.status(output)
            if not result['process_alive']:
                return result
            time.sleep(.1)
        jobs.stop(output)
        self.fail('synthetic job did not exit before timeout')

    def test_background_stop_resume_and_verified_reuse(self):
        self.image()
        output = self.root / 'background'
        result = jobs.start(self.src, output, Config(target_spacing=(.5, .5, .5)))
        self.assertEqual(result['state'], 'submitted')
        with self.assertRaises(RuntimeError): jobs.resume(output)
        jobs.stop(output)
        stopped = self.wait_job(output)
        self.assertEqual(stopped['worker']['state'], 'stopped', stopped)
        jobs.resume(output)
        finished = self.wait_job(output)
        self.assertEqual(finished['worker']['completed'], 1, finished)
        jobs.resume(output)
        reused = self.wait_job(output)
        self.assertEqual(reused['worker']['reused'], 1, reused)

    def test_background_rejects_changed_input_and_frozen_configuration(self):
        source = self.image()
        output = self.root / 'background'
        jobs.start(self.src, output, Config(target_spacing=(.5, .5, .5)))
        self.wait_job(output)
        source.write_bytes(source.read_bytes() + b'changed')
        jobs.resume(output)
        failed = self.wait_job(output)
        self.assertEqual(failed['worker']['state'], 'failed')
        self.assertIn('input changed', failed['worker']['error'])
        with patch.object(Config, 'signature', return_value='changed-code'):
            with self.assertRaisesRegex(ValueError, 'implementation'):
                jobs.resume(output)

    def test_stale_pid_is_not_treated_as_this_job(self):
        self.assertFalse(jobs.alive({'pid': psutil.Process().pid, 'created': 1}))

    def test_frozen_source_metadata_is_rechecked_at_case_execution(self):
        source = self.image()
        manifest = self.root / 'frozen.json'
        manifest.write_text(json.dumps([{'input': source.name, 'size': source.stat().st_size + 1,
                                        'mtime_ns': source.stat().st_mtime_ns}]))
        result = run(self.src, self.root / 'output', Config(), manifest)
        self.assertEqual(result['failed'], 1)
        self.assertIn('frozen job', result['errors'][0]['error'])

    def test_benchmark_selects_bounded_cases_and_checks_pixels(self):
        for i in range(4): self.image(f'{i}.nii.gz', shape=(10, 10, 8+i))
        report = benchmark(self.src, self.root / 'bench', Config(target_spacing=(.5, .5, .5)),
                           samples=3, worker_options=(1, 2))
        self.assertEqual(report['source_count'], 3)
        self.assertTrue(report['numerically_identical'])
        self.assertEqual(len(report['trials']), 2)
        self.assertGreater(report['trials'][0]['performance']['wall_seconds'], 0)


if __name__ == '__main__': unittest.main()
