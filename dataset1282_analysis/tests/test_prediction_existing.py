from dataclasses import asdict
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import blosc2
import numpy as np

from dataset1282_analysis.contracts import atomic_json, digest, read_json, sha256_file
from dataset1282_analysis.observability.status import operation_lock
from dataset1282_analysis.prediction import PROTOCOL
from dataset1282_analysis.prediction.cli import execute
from dataset1282_analysis.prediction.source import freeze_source, implementation
from dataset1282_analysis.prediction.workflow import case_record
from dataset1282_analysis.xu_geometry import PreparedCache


class ExistingPredictionTests(unittest.TestCase):
    def test_prior_models_through_isolated_cpu_worker(self):
        existing = os.environ.get('DATASET1282_PREDICTION_TEST_TRAINING')
        if not existing:
            self.skipTest('Pass a completed synthetic training directory for process validation')
        root = Path(existing).resolve()
        record = read_json(root/'run.json')
        self.assertEqual(record['scope'],'technical_subset')
        cache = PreparedCache(record['prepared'],allow_technical_subset=True)
        original_name = cache.case_ids('development')[0]
        geo = cache.geometry(original_name)
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch).resolve()
            source = freeze_source(root,cache,out,'vessels')
            job = dict(protocol=PROTOCOL,prepared=str(cache.run_dir),source=digest(source),
                       mode='vessels',technical_subset=True,implementation=implementation(),
                       cases={'synthetic_held':dict(role='target_primary',shape=list(geo.native_shape_xyz),
                                                   sha256={'image':'synthetic-ct'})})
            atomic_json(out/'run.json',job)
            folder = out/'prepared/synthetic_held'
            folder.mkdir(parents=True)
            # 使用既有合成输入验证进程串联，正式留出组保持未打开状态。
            array = blosc2.asarray(cache.read_image(original_name)[None],
                                  urlpath=str(folder/'image.b2nd'),mode='w')
            del array
            atomic_json(folder/'geometry.json',asdict(geo))
            atomic_json(folder/'complete.json',dict(job=digest(job),case='synthetic_held',
                source_sha256='synthetic-ct',artifacts={f:sha256_file(folder/f)
                                                     for f in ('image.b2nd','geometry.json')}))
            args = SimpleNamespace(device='cpu',threads=1,wait_seconds=1,resume=False)
            with operation_lock(out,'xu-model-run',out) as owner:
                execute(args,out,owner,job,'predict')
                complete = read_json(out/'complete.json')
                self.assertEqual(complete['job'],digest(job))
                self.assertEqual(complete['rejected'],0)
                self.assertEqual(case_record(out,'synthetic_held',job)['state'],'predicted')
                before = {p:sha256_file(p) for p in (out/'cases').glob('*') if p.is_file()}
                execute(args,out,owner,job,'predict')
                self.assertEqual(before,{p:sha256_file(p) for p in before})
            reports = list((out/'resources').glob('predict-launch-*.json'))
            self.assertEqual(len(reports),2)
            self.assertTrue(all(read_json(p)['cuda_visible_devices']=='-1' for p in reports))
            self.assertTrue(np.load(out/'cases/synthetic_held.npy',allow_pickle=False).size)


if __name__ == '__main__':
    unittest.main()
