import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess

import psutil
import numpy as np
import nibabel as nib

from chd_stunet.observability.runlog import write_json
from chd_stunet.acceleration.preprocess_cpu import interpreter_command
from chd_stunet.acceleration.prediction import process_options, stop_owned, desktop_identity, can_admit, refresh_budget, admitted_threads
from chd_stunet.prediction.export import valid_result
from chd_stunet.prediction.model import file_sha, runtime_versions
from chd_stunet.prediction.run import assert_inactive, validate_job, wait_expired, ensemble_results, worker_failure, reap_orphans
from chd_stunet.training.contracts import digest


class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ.get('CHD_TEST_SCRATCH', 'D:/projects/back/t17/chd-prediction-20261108/tests'))
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_live_launch_is_refused(self):
        write_json(self.root/'launch.json', {'pid': os.getpid(), 'born': psutil.Process().create_time()})
        with self.assertRaisesRegex(RuntimeError, '运行'):
            assert_inactive(self.root)
        write_json(self.root/'launch.json', {'pid': os.getpid(), 'born': 0})
        assert_inactive(self.root)

    def test_completed_file_tamper_is_refused(self):
        write_json(self.root/'result.json', {'case_id': 'real-file'})
        write_json(self.root/'complete.json', {'identity': 'test', 'files': {'result.json': file_sha(self.root/'result.json')}})
        self.assertEqual(valid_result(self.root, 'test')['case_id'], 'real-file')
        write_json(self.root/'result.json', {'case_id': 'changed'})
        with self.assertRaisesRegex(ValueError, '损坏'):
            valid_result(self.root, 'test')

    def test_job_change_is_refused(self):
        job = {'schema': 1, 'versions': runtime_versions(), 'options': {'tta': True}}
        job['identity'] = digest(job)
        validate_job(job)
        job['options']['tta'] = False
        with self.assertRaisesRegex(ValueError, '签名'):
            validate_job(job)

    def test_real_child_identity_and_owned_stop(self):
        env = dict(os.environ)
        code = 'import os,time; print(os.getpid(),flush=True); time.sleep(60)'
        process = subprocess.Popen([interpreter_command(env), '-c', code], env=env,
                                   stdout=subprocess.PIPE, text=True, **process_options())
        born = psutil.Process(process.pid).create_time()
        self.assertEqual(int(process.stdout.readline()), process.pid)
        stop_owned({'process': process, 'born': born, 'stream': process.stdout})
        self.assertIsNotNone(process.poll())

    def test_timeout_counts_only_waiting_for_unfinished_work(self):
        self.assertFalse(wait_expired(False, False, 90, 0, 400))
        self.assertFalse(wait_expired(True, True, 90, 0, 400))
        self.assertFalse(wait_expired(True, False, 0, 0, 400))
        self.assertTrue(wait_expired(True, False, 90, 0, 400))

    def test_idle_stage_recovers_cpu_and_memory_capacity(self):
        old = dict(workers=1, cpu_budget=1, memory_budget_bytes=30)
        fresh = dict(workers=3, cpu_budget=8, memory_budget_bytes=100)
        self.assertEqual(admitted_threads(old, fresh, [], 4), 1)
        self.assertEqual(admitted_threads(old, fresh, [dict(threads=1)], 4), 0)
        self.assertFalse(can_admit(old, fresh, [], 60, 1))
        renewed = refresh_budget(old, fresh, busy=False)
        self.assertTrue(can_admit(renewed, fresh, [], 60, 4))
        self.assertEqual(refresh_budget(old, fresh, busy=True), old)

    def test_fatal_worker_record_has_priority_over_resource_recovery(self):
        write_json(self.root/'failure.json', {'kind': 'fatal', 'error': 'invalid_model'})
        for code in (None, 0, 1, -15):
            with self.assertRaisesRegex(RuntimeError, 'invalid_model'):
                worker_failure(self.root, code, allow_success=True, supervisor_stopped=True)
        write_json(self.root/'failure.json', {'kind': 'resource', 'error': 'allocation'})
        self.assertEqual(worker_failure(self.root, 1)['kind'], 'resource')

    def test_unknown_exit_is_fatal_and_successful_export_is_allowed(self):
        with self.assertRaises(RuntimeError):
            worker_failure(self.root, 1, allow_success=True)
        self.assertIsNone(worker_failure(self.root, 0, allow_success=True))

    def test_orphan_cleanup_terminates_only_verified_child(self):
        directory = self.root/'control/gpu-real-child'
        directory.mkdir(parents=True)
        env = dict(os.environ)
        process = subprocess.Popen([interpreter_command(env), '-c',
            'import os,time; print(os.getpid(),flush=True); time.sleep(60)', '--directory', str(directory)],
            env=env, stdout=subprocess.PIPE, text=True, **process_options())
        born = psutil.Process(process.pid).create_time()
        self.addCleanup(stop_owned, {'process': process, 'born': born, 'stream': process.stdout})
        self.assertEqual(int(process.stdout.readline()), process.pid)
        owner = dict(pid=process.pid, born=born, executable=psutil.Process(process.pid).exe(),
                     identity='test', directory=str(directory))
        write_json(directory/'owner.json', owner)
        self.assertEqual(len(reap_orphans(self.root, 'test')), 1)
        process.wait(timeout=10)
        self.assertIsNotNone(process.poll())

    def test_orphan_cleanup_refuses_unrelated_process(self):
        directory = self.root/'control/gpu-unrelated'
        directory.mkdir(parents=True)
        owner = dict(pid=os.getpid(), born=psutil.Process().create_time(), executable=psutil.Process().exe(),
                     identity='test', directory=str(directory))
        write_json(directory/'owner.json', owner)
        with self.assertRaisesRegex(RuntimeError, '身份'):
            reap_orphans(self.root, 'test')

    def test_all_resampling_dependencies_are_bound(self):
        for name in ('scikit-image', 'acvl-utils'):
            job = {'versions': runtime_versions(), 'options': {}}
            job['versions'][name] = 'unreviewed-version'
            job['identity'] = digest(job)
            with self.assertRaisesRegex(ValueError, name):
                validate_job(job)

    def test_new_ct_rejects_conflicting_header_and_affine(self):
        from chd_stunet.prediction.inputs import new_cases
        image = nib.Nifti1Image(np.ones((7, 8, 9), dtype=np.float32), np.diag([2., 3., 4., 1.]))
        image.header.set_xyzt_units('mm')
        image.header.set_zooms((1., 1., 1.))
        path = self.root/'conflict.nii.gz'
        nib.save(image, path)
        with self.assertRaisesRegex(ValueError, '冲突'):
            new_cases([path], [])

    def test_desktop_exception_rejects_formal_and_other_processes(self):
        with self.assertRaises(ValueError):
            desktop_identity(os.getpid(), False)
        with self.assertRaises(ValueError):
            desktop_identity(os.getpid(), True)

    def test_idle_worker_retains_ram_without_occupying_case_slot(self):
        stage = dict(workers=1, cpu_budget=4, memory_budget_bytes=100)
        idle = [dict(threads=0, estimate=20, private=20)]
        self.assertTrue(can_admit(stage, stage, idle, 70, 4))
        self.assertFalse(can_admit(stage, stage, idle, 90, 4))

    def test_probability_ensemble_uses_common_native_grid(self):
        arrays = []
        for fold, label in enumerate((1, 2, 2)):
            probability = np.zeros((8, 3, 4, 5), dtype=np.float32)
            probability[label] = 1
            path = self.root/f'probability_{fold}.npy'
            np.save(path, probability)
            arrays.append(dict(case_id='synthetic', fold=fold, shape=[3, 4, 5],
                               affine=np.eye(4).tolist(), probabilities=str(path), reference_image='unit fixture'))
        job = {'models': [0, 1, 2], 'identity': 'unit fixture'}
        ensemble_results(self.root, job, arrays)
        self.assertTrue(np.all(np.asarray(nib.load(self.root/'ensemble/synthetic.nii.gz').dataobj) == 2))
        arrays[0]['affine'][0][0] = 2
        with self.assertRaisesRegex(ValueError, '共同网格'):
            ensemble_results(self.root, job, arrays)


if __name__ == '__main__':
    unittest.main()
