import copy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dataset1282_analysis.contracts import atomic_json, digest, read_json, sha256_file
from dataset1282_analysis.storage_migration import apply_migration, migration_plan
from dataset1282_analysis.xu_geometry import PreparedCache
from dataset1282_analysis.xu_pipeline import _contract, _run_stage
from dataset1282_analysis.xu_pipeline_stages import Stage
from test_xu_targets import on_disk_phantom


BASE = Path(__file__).resolve().parents[1]/'local-validation/storage-resume'
BASE.mkdir(parents=True,exist_ok=True)


class StorageResumeTests(unittest.TestCase):
    def test_migrated_prepared_and_stage_retain_original_identity(self):
        with TemporaryDirectory(dir=BASE) as temporary:
            root=Path(temporary).resolve()
            source=root/'old'
            out=source/'experiments/run'
            out.mkdir(parents=True)
            cache=on_disk_phantom(source/'prepared')
            args=SimpleNamespace(profile='smoke',technical_subset=True,device='cpu',devices=None,
                resource_policy='legacy',supervision='marginal',seed=2023,prefetch=False,
                prefetch_mib=4096,prefetch_workers=4,threads=2,crop64=None,crop128=None,
                reviews=None,resume=False,output=out)
            contract=_contract(cache,args)
            atomic_json(out/'run.json',contract)
            stage=Stage('crop64',('train','--prepared',str(cache.run_dir),'--output',str(out/'crop64')),
                        (),skip_reason=dict(reason='supplied_frozen_locator'))
            receipt=_run_stage(stage,args,out,{},contract,digest(contract),{},{},1)
            protected=sha256_file(out/'run.json')
            plan=migration_plan(source,root/'new')
            atomic_json(root/'plan.json',plan)
            apply_migration(root/'plan.json',root/'home')
            moved=PreparedCache(root/'new/prepared',allow_technical_subset=True)
            image_box=((0,20),)*3
            np.testing.assert_array_equal(moved.read_image('train_a',(20,20,20),image_box),
                                          cache.read_image('train_a',(20,20,20),image_box))
            resumed=copy.copy(args)
            resumed.resume=True
            resumed.output=root/'new/experiments/run'
            new_contract=_contract(moved,resumed)
            self.assertEqual(new_contract,contract)
            self.assertIsNotNone(resumed.execution_relocation)
            new_stage=replace(stage,argv=tuple(value.replace(str(source),str(root/'new'))
                                             for value in stage.argv))
            reused=_run_stage(new_stage,resumed,resumed.output,{},contract,digest(contract),{},{},1)
            self.assertEqual(reused,receipt)
            self.assertEqual(sha256_file(resumed.output/'run.json'),protected)
            self.assertEqual(read_json(resumed.output/'run.json')['prepared'],str(cache.run_dir))

    def test_new_blood_bundle_matches_legacy_and_resume_keeps_latest_state(self):
        from dataset1282_analysis.xu_blood import train, load_blood_predictor
        from dataset1282_analysis.xu_blood_checkpoint import FORMAT, selected_components
        from dataset1282_analysis.xu_targets import build_targets
        old_threads=torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,old_threads)
        with TemporaryDirectory(dir=BASE) as temporary:
            root=Path(temporary).resolve()
            cache=on_disk_phantom(root/'prepared')
            targets=root/'targets'
            build_targets(cache,SimpleNamespace(width_mm=1.,reviews=None,resume=False),targets)
            args=SimpleNamespace(targets=targets,profile='smoke',technical_unvalidated=False,
                                 supervision='marginal',seed=17,device='cpu',resume=False)
            out=root/'blood'
            out.mkdir()
            train(cache,args,out)
            minimal=torch.load(out/'best.pt',weights_only=True)
            self.assertEqual(minimal['format'],FORMAT)
            self.assertNotIn('encoder',minimal)
            self.assertNotIn('sequence',minimal)
            components=selected_components(out/'best.pt')
            legacy={name:components[name] for name in ('encoder','sequence','kind','contract')}
            torch.save(legacy,out/'legacy.pt')
            new_pred=load_blood_predictor(cache,out/'best.pt','cpu','smoke')
            old_pred=load_blood_predictor(cache,out/'legacy.pt','cpu','smoke')
            np.testing.assert_array_equal(new_pred('dev_b',None,(3,2,4)),old_pred('dev_b',None,(3,2,4)))
            names=('blood-contract.json','encoder-last.pt','sequence-last.pt',
                   'encoder-selected.pt','sequence-selected.pt','best.pt')
            before={name:sha256_file(out/name) for name in names}
            for stage in ('encoder','sequence'):
                state=torch.load(out/(stage+'-last.pt'),weights_only=True)
                self.assertEqual(state['epoch'],1)
                self.assertTrue(state['optimizer']['state'])
            args.resume=True
            train(cache,args,out)
            self.assertEqual({name:sha256_file(out/name) for name in names},before)
            with (out/'sequence-selected.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'变化'):
                selected_components(out/'best.pt')
