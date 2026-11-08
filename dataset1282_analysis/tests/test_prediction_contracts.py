from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
import unittest

from dataset1282_analysis.contracts import atomic_json, sha256_file
from dataset1282_analysis.prediction.cli import validate_output
from dataset1282_analysis.prediction.heldout import prepare_memory
from dataset1282_analysis.prediction.source import inspect_source, freeze_source
from dataset1282_analysis.prediction.compatibility import source_schema
from dataset1282_analysis.xu_pipeline import _contract
from test_xu_targets import on_disk_phantom


class PredictionContractTests(unittest.TestCase):
    def test_prior_completed_training_read_only(self):
        existing = os.environ.get('DATASET1282_PREDICTION_TEST_TRAINING')
        if not existing:
            self.skipTest('Pass a completed technical training directory to verify old artifacts')
        from dataset1282_analysis.contracts import read_json
        from dataset1282_analysis.xu_geometry import PreparedCache
        root = Path(existing).resolve()
        record = read_json(root/'run.json')
        self.assertEqual(record['scope'],'technical_subset')
        cache = PreparedCache(record['prepared'],allow_technical_subset=True)
        files = [root/'run.json',*(root/'receipts').glob('*.json'),*root.glob('*/best.pt')]
        before = {str(p):sha256_file(p) for p in files}
        with tempfile.TemporaryDirectory() as scratch:
            identity,paths = inspect_source(root,cache,'vessels')
            frozen = freeze_source(root,cache,Path(scratch).resolve(),'vessels')
            self.assertEqual(identity,frozen)
            self.assertEqual(frozen['run'],record)
            self.assertIn('blood',paths)
        self.assertEqual(before,{str(p):sha256_file(p) for p in files})

    def test_unreviewed_legacy_source_rejected(self):
        from dataset1282_analysis.xu_pipeline import _sources
        record = dict(protocol='xu-model-run-2',scope='technical_subset',implementation=_sources(True))
        with self.assertRaisesRegex(ValueError,'no reviewed'):
            source_schema(record,_sources(True))

    def test_unfinished_actual_training_contract_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch).resolve()
            cache = on_disk_phantom(base/'prepared')
            args = SimpleNamespace(technical_subset=True,profile='smoke',supervision='marginal',
                device='cpu',devices=None,exclude_physical_gpus=[],resource_policy='v2',
                prefetch=True,prefetch_mib=None,threads=1,seed=2023,crop64=None,crop128=None,reviews=None)
            record = _contract(cache,args)
            atomic_json(base/'training/run.json',record)
            before = sha256_file(base/'training/run.json')
            with self.assertRaisesRegex(ValueError,'no completed receipt'):
                inspect_source(base/'training',cache,'four-way')
            self.assertEqual(before,sha256_file(base/'training/run.json'))

    def test_output_cannot_write_inside_or_above_inputs(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch).resolve()
            config = base/'config.json'
            atomic_json(config,dict(data_root=str(base/'data'),metadata_excel=str(base/'book.xlsx')))
            args = SimpleNamespace(prepared=base/'prepared',training=base/'training',
                                   config=config,policy=base/'policy.json',output=base/'output')
            validate_output(args)
            for out in (base,base/'data',base/'data/output',base/'training/sub',base/'prepared/sub'):
                args.output = out
                with self.subTest(out=out),self.assertRaisesRegex(ValueError,'separate'):
                    validate_output(args)
                self.assertFalse(out.exists() and (out/'.runtime').exists())

    def test_memory_admission_accounts_for_finer_target_spacing(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            plan = base/'preprocessed/Dataset282_CHDSource/nnUNetPlans.json'
            atomic_json(plan,dict(transpose_forward=[2,0,1],
                configurations={'3d_fullres':dict(spacing=[.5,.5,.5])}))
            job = dict(prepared=str(base),cases={'held':dict(shape=[10,20,30],spacing=[1,1,1])})
            self.assertEqual(prepare_memory(job),6000*32+48000*48+2*1024**3)


if __name__ == '__main__':
    unittest.main()
