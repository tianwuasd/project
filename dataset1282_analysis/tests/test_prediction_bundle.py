from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
import unittest

import torch

from dataset1282_analysis.contracts import atomic_json, sha256_file
from dataset1282_analysis.prediction import PROTOCOL
from dataset1282_analysis.prediction.source import _freeze_checkpoints, verify_frozen, _receipt
from dataset1282_analysis.prediction.compatibility import receipt_specs
from dataset1282_analysis.artifact_retention import release_branch_arrays
from dataset1282_analysis.xu_blood_checkpoint import FORMAT, selected_components


BASE = Path('D:/projects/back/t17/d1282-prediction-bundle') if os.name == 'nt' else Path(__file__).resolve().parents[1]/'local-validation/prediction-bundle'
BASE.mkdir(parents=True, exist_ok=True)


class PredictionBundleTests(unittest.TestCase):
    def freeze(self, original, output):
        destination = output/'source'
        destination.mkdir(parents=True)
        identity = dict(protocol=PROTOCOL,checkpoints={'blood':sha256_file(original)})
        _freeze_checkpoints({'blood':original},identity,destination)
        atomic_json(destination/'manifest.json',identity)
        self.assertEqual(verify_frozen(output),identity)
        return destination

    def test_reference_bundle_freezes_components_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            base = Path(temporary); original = base/'training'; original.mkdir()
            components = {}
            for index, stage in enumerate(('encoder','sequence'),1):
                path = original/(stage+'-selected.pt')
                torch.save(dict(contract='contract',stage=stage,epoch=index,
                                model={'weight':torch.arange(8)*index}),path)
                components[stage] = dict(file=path.name,sha256=sha256_file(path),epoch=index)
            checkpoint = original/'best.pt'
            torch.save(dict(format=FORMAT,kind='blood',contract='contract',components=components),checkpoint)
            destination = self.freeze(checkpoint,base/'output')
            frozen = torch.load(destination/'blood.pt',map_location='cpu',weights_only=True)
            self.assertNotIn('encoder',frozen)
            self.assertNotIn('sequence',frozen)
            self.assertEqual(sha256_file(destination/'blood.pt'),sha256_file(checkpoint))
            for stage in components:
                name = components[stage]['file']
                self.assertEqual(sha256_file(destination/name),sha256_file(original/name))
            state = selected_components(destination/'blood.pt')
            self.assertTrue(torch.equal(state['encoder']['weight'],torch.arange(8)))
            torch.save(dict(model={'weight':torch.zeros(8)}),destination/'encoder-selected.pt')
            with self.assertRaises(ValueError):
                verify_frozen(base/'output')

    def test_inline_blood_checkpoint_remains_supported(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            base = Path(temporary); checkpoint = base/'best.pt'
            torch.save(dict(kind='blood',contract='contract',encoder={'weight':torch.ones(3)},
                            sequence={'weight':torch.arange(4)}),checkpoint)
            destination = self.freeze(checkpoint,base/'output')
            self.assertEqual({p.name for p in destination.iterdir()},{'blood.pt','manifest.json'})
            self.assertTrue(torch.equal(selected_components(destination/'blood.pt')['encoder']['weight'],torch.ones(3)))

    def test_reference_epoch_identity_is_validated(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            base = Path(temporary); components = {}
            for stage in ('encoder','sequence'):
                path = base/(stage+'-selected.pt')
                torch.save(dict(contract='contract',stage=stage,epoch=1,model={'weight':torch.ones(1)}),path)
                components[stage] = dict(file=path.name,sha256=sha256_file(path),epoch=2)
            checkpoint = base/'best.pt'
            torch.save(dict(format=FORMAT,kind='blood',contract='contract',components=components),checkpoint)
            with self.assertRaises(ValueError):
                self.freeze(checkpoint,base/'output')

    def test_source_receipt_accepts_only_registered_branch_cleanup(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            training = Path(temporary); output = training/'all64'
            (output/'branches/case').mkdir(parents=True)
            torch.save({'weight':torch.ones(1)},output/'best.pt')
            atomic_json(output/'training.json',{'completed':True})
            np.save(output/'branches/case/all64.npy',np.ones((2,2,2),dtype=np.uint8))
            files = {p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob('*') if p.is_file()}
            stage = SimpleNamespace(name='all64',argv=('train','--output',str(output)),
                                    required=('best.pt','training.json'),skip_reason=None)
            receipt = dict(spec=next(iter(receipt_specs(stage,'contract','xu-stage-1'))),state='complete',files=files)
            atomic_json(training/'receipts/all64.json',receipt)
            release_branch_arrays(output,files,stage.required)
            self.assertEqual(_receipt(training,stage,'contract','xu-stage-1'),receipt)
            atomic_json(output/'training.json',{'completed':False})
            with self.assertRaises(ValueError):
                _receipt(training,stage,'contract','xu-stage-1')


if __name__ == '__main__':
    unittest.main()
