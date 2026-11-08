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
from chd_stunet.acceleration.prediction import process_options, stop_owned, desktop_identity, can_admit
from chd_stunet.prediction.export import valid_result
from chd_stunet.prediction.model import file_sha
from chd_stunet.prediction.run import assert_inactive, validate_job, wait_expired, ensemble_results
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
        job = {'schema': 1, 'versions': {}, 'options': {'tta': True}}
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
