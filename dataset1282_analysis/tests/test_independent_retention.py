import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from dataset1282_analysis.contracts import atomic_json, digest, read_json, sha256_file
from dataset1282_analysis.independent.stunet import _prune_best_checkpoints
from dataset1282_analysis.independent.cli import _resume_training_job, _training_execution_paths
from dataset1282_analysis.storage_migration import migration_plan, apply_migration


BASE = Path('D:/projects/back/t17/d1282-independent-retention') if os.name == 'nt' else Path(__file__).resolve().parents[1]/'local-validation/independent-retention'
BASE.mkdir(parents=True,exist_ok=True)


class IndependentRetentionTests(unittest.TestCase):
    def committed_training(self, output):
        output.mkdir()
        job = dict(model='stunet',command='train',settings={'fixture':'real-sgd'})
        atomic_json(output/'run.json',job)
        identity = digest(job)
        model = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():
            model.weight.fill_(0.)
        optimizer = torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        history = []; best = float('inf'); best_epoch = None
        for epoch in range(1,4):
            if epoch == 3:
                optimizer.param_groups[0]['lr'] = 10.
            optimizer.zero_grad()
            loss = (model(torch.ones(1,1))-1).square().mean()
            loss.backward(); optimizer.step()
            validation = float((model(torch.ones(1,1))-1).square().mean().detach())
            history.append(dict(epoch=epoch,development_loss=validation,training_loss=float(loss.detach())))
            state = dict(job=identity,epoch=epoch,model=model.state_dict())
            if validation < best:
                best = validation; best_epoch = epoch
                torch.save(state,output/f'best-{epoch}.pt')
        best_file = output/f'best-{best_epoch}.pt'
        torch.save(dict(state,optimizer=optimizer.state_dict(),history=history,
                        best_epoch=best_epoch,best_sha256=sha256_file(best_file)),output/'last.pt')
        atomic_json(output/'training.json',dict(job=identity,epoch=3,history=history,
                    best_file=best_file.name,best_sha256=sha256_file(best_file)))
        return identity

    def test_selected_best_and_latest_optimizer_survive_pruning_and_restore(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            output = Path(temporary)/'training'; identity = self.committed_training(output)
            before = sha256_file(output/'last.pt')
            selected = read_json(output/'training.json')['best_file']
            unknown = output/'best-manual.pt'; unknown.write_bytes(b'owned by another workflow')
            self.assertEqual(_prune_best_checkpoints(output,identity),['best-1.pt'])
            self.assertEqual(sorted(p.name for p in output.glob('best-*.pt')),[selected,'best-manual.pt'])
            self.assertEqual(sha256_file(output/'last.pt'),before)
            restored = []
            for _ in range(2):
                state = torch.load(output/'last.pt',weights_only=True,map_location='cpu')
                model = torch.nn.Linear(1,1,bias=False); model.load_state_dict(state['model'])
                optimizer = torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
                optimizer.load_state_dict(state['optimizer'])
                self.assertTrue(optimizer.state_dict()['state'])
                optimizer.zero_grad(); (model(torch.ones(1,1))-1).square().mean().backward(); optimizer.step()
                restored.append(model.weight.detach().clone())
            self.assertTrue(torch.equal(*restored))
            self.assertEqual(_prune_best_checkpoints(output,identity),[])

    def test_foreign_numeric_checkpoint_prevents_any_deletion(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            output = Path(temporary)/'training'; identity = self.committed_training(output)
            torch.save(dict(job='foreign-job',epoch=3,model={'weight':torch.ones(1)}),output/'best-3.pt')
            with self.assertRaises(ValueError):
                _prune_best_checkpoints(output,identity)
            self.assertTrue((output/'best-1.pt').is_file())
            self.assertTrue((output/'best-3.pt').is_file())

    def test_epoch_mismatch_and_uncommitted_summary_preserve_prior_best(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            output = Path(temporary)/'training'; identity = self.committed_training(output)
            original = (output/'training.json').read_bytes()
            summary = read_json(output/'training.json'); summary['epoch'] = 2
            atomic_json(output/'training.json',summary)
            with self.assertRaises(ValueError):
                _prune_best_checkpoints(output,identity)
            self.assertTrue((output/'best-1.pt').is_file())
            (output/'training.json').write_bytes(original)
            torch.save(dict(job=identity,epoch=9,model={'weight':torch.ones(1)}),output/'best-1.pt')
            with self.assertRaises(ValueError):
                _prune_best_checkpoints(output,identity)
            self.assertTrue((output/'best-1.pt').is_file())

    def test_registered_migration_preserves_job_and_resolves_execution_inputs(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            base = Path(temporary); source = base/'old'; target = base/'new'
            (source/'prepared').mkdir(parents=True); (source/'experiment').mkdir()
            (source/'prepared/data.bin').write_bytes(b'prepared-content')
            torch.save({'weight':torch.arange(4)},source/'weights.pt')
            job = dict(model='stunet',command='train',prepared=str(source/'prepared'),
                       weights=str(source/'weights.pt'),weights_sha256=sha256_file(source/'weights.pt'))
            atomic_json(source/'experiment/run.json',job)
            original = (source/'experiment/run.json').read_bytes()
            atomic_json(base/'plan.json',migration_plan(source,target))
            apply_migration(base/'plan.json',base/'home')
            output = target/'experiment'
            candidate = dict(job,prepared=str(target/'prepared'),weights=str(target/'weights.pt'))
            resumed = _resume_training_job(output,candidate)
            self.assertEqual(resumed,job)
            self.assertEqual(digest(resumed),digest(job))
            self.assertEqual((output/'run.json').read_bytes(),original)
            self.assertEqual(_training_execution_paths(output,resumed),
                             {'prepared':target/'prepared','weights':target/'weights.pt'})
            candidate['weights_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                _resume_training_job(output,candidate)
            (target/'weights.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                _training_execution_paths(output,resumed)


if __name__ == '__main__':
    unittest.main()
