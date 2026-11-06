"""Audited checkpoint import must not alter its source or reset progress."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        from chd_stunet.training.contracts import digest,scientific_sources
        from chd_stunet.training.code_snapshot import freeze_code
        from chd_stunet.training.checkpoints import save_checkpoint
        from chd_stunet.training.engine import initial_state
        scratch=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.tmp.name)
        self.source=self.root/'old';self.source.mkdir();(self.source/'.training.lock').write_bytes(b'0')
        self.settings=dict(epochs=2,train_steps_per_epoch=4,validation_steps_per_epoch=2,batch_size=4,
                           membership_policy='step-boundary-v1')
        body=dict(recipe=self.settings,data={'fold':0},versions={'torch':'test'},weight_sha256='fixed',
                  producer_ast_sha256=scientific_sources())
        self.signature=dict(identity=digest(body),**body)
        freeze_code(self.source,self.signature,True)
        (self.source/'signature.json').write_text(json.dumps(self.signature))
        self.model=torch.nn.Linear(2,1);self.opt=torch.optim.SGD(self.model.parameters(),lr=.01,momentum=.99)
        self.scaler=torch.amp.GradScaler('cuda',enabled=False)
        self.state=initial_state()
        distributed=dict(members=list('abcd'),rng_by_rank=[{}]*4)
        pointer=save_checkpoint(self.source/'checkpoints',self.model,self.opt,self.scaler,self.state,
                                self.signature,torch.device('cpu'),distributed)
        (self.source/'checkpoints/epoch.json').write_text(json.dumps(pointer))
        self.state.update(step=1,attempts=1,updates=1,train_batches=1,updates_by_cohort=dict(main=4,auxiliary=0))
        save_checkpoint(self.source/'checkpoints',self.model,self.opt,self.scaler,self.state,
                        self.signature,torch.device('cpu'),distributed)

    def tearDown(self):self.tmp.cleanup()
    def hashes(self):return {str(p.relative_to(self.source)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in self.source.rglob('*') if p.is_file()}

    def test_import_preserves_full_payload_and_source_bytes(self):
        from chd_stunet.training.continuation import read_continuation,install_continuation
        from chd_stunet.training.checkpoints import read_checkpoint
        before=self.hashes();bundle=read_continuation(self.source,self.signature,self.settings)
        output=self.root/'new';output.mkdir()
        install_continuation(output,bundle,self.signature)
        actual=read_checkpoint(output/'checkpoints',self.signature)
        self.assertEqual(actual['state'],self.state)
        for key,value in self.model.state_dict().items():torch.testing.assert_close(actual['model'][key],value,rtol=0,atol=0)
        self.assertEqual(actual['optimizer'],self.opt.state_dict());self.assertEqual(before,self.hashes())
        self.assertTrue((output/'continuation.json').exists())
        self.assertFalse((output/'continuation.pending.json').exists())
        self.assertEqual(read_checkpoint(output/'checkpoints',self.signature,'epoch.json')['state']['step'],0)

    def test_data_recipe_and_environment_mismatch_refused(self):
        from chd_stunet.training.continuation import read_continuation
        from chd_stunet.training.contracts import digest
        for field,value in (('data',{'fold':1}),('versions',{'torch':'changed'}),('weight_sha256','changed'),
                            ('recipe',dict(self.settings,batch_size=8))):
            new=copy.deepcopy(self.signature);new[field]=value
            new['identity']=digest({k:v for k,v in new.items() if k!='identity'})
            with self.subTest(field=field),self.assertRaises(ValueError):read_continuation(self.source,new,self.settings)

    def test_active_source_and_tampered_checkpoint_refused(self):
        from chd_stunet.training.continuation import read_continuation
        from chd_stunet.observability.locking import run_lock
        with run_lock(self.source/'.training.lock'):
            with self.assertRaisesRegex(RuntimeError,'active'):read_continuation(self.source,self.signature,self.settings)
        pointer=json.loads((self.source/'checkpoints/latest.json').read_text())
        target=self.source/'checkpoints'/pointer['file'];target.write_bytes(target.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'checksum'):read_continuation(self.source,self.signature,self.settings)

    def test_incomplete_import_cannot_resume_from_author_weights(self):
        from chd_stunet.training.supervisor import _existing_checkpoint
        output=self.root/'incomplete';output.mkdir();(output/'continuation.pending.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Incomplete checkpoint import'):_existing_checkpoint(output,self.signature,self.settings)

    def test_legacy_import_requires_exact_allowlist_and_only_membership_policy_difference(self):
        from chd_stunet.training.continuation import read_continuation
        from chd_stunet.training.contracts import scientific_sources,digest
        from chd_stunet.training.checkpoints import read_checkpoint,commit_payload
        payloads={role:read_checkpoint(self.source/'checkpoints',self.signature,role) for role in ('epoch.json','latest.json')}
        path=self.source/'code/src/chd_stunet/training/rank_runtime.py'
        path.write_bytes(path.read_bytes()+b'\nSYNTHETIC_OLD_PRODUCER = True\n')
        record=json.loads((self.source/'code_sources.json').read_text());record['files']['src/chd_stunet/training/rank_runtime.py']=hashlib.sha256(path.read_bytes()).hexdigest()
        (self.source/'code_sources.json').write_text(json.dumps(record))
        old=copy.deepcopy(self.signature);old['recipe'].pop('membership_policy');old['producer_ast_sha256']=scientific_sources(self.source/'code')
        old['identity']=digest({k:v for k,v in old.items() if k!='identity'})
        (self.source/'signature.json').write_text(json.dumps(old))
        for role,payload in payloads.items():commit_payload(self.source/'checkpoints',dict(payload,signature=old),role)
        with self.assertRaisesRegex(ValueError,'Unknown continuation producer'):read_continuation(self.source,self.signature,self.settings)
        with patch('chd_stunet.training.continuation.LEGACY_052_SOURCES',{digest(old['producer_ast_sha256'])}):
            bundle=read_continuation(self.source,self.signature,self.settings)
            self.assertEqual(bundle['payloads']['latest.json']['state'],self.state)

    def test_partial_import_failure_leaves_recovery_guard(self):
        from chd_stunet.training.continuation import read_continuation,install_continuation
        from chd_stunet.training.supervisor import _existing_checkpoint
        bundle=read_continuation(self.source,self.signature,self.settings)
        output=self.root/'crashed';output.mkdir()
        with patch('chd_stunet.training.continuation.commit_payload',side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):install_continuation(output,bundle,self.signature)
        with self.assertRaisesRegex(ValueError,'Incomplete checkpoint import'):_existing_checkpoint(output,self.signature,self.settings)

    def test_053_retains_exact_recipe_and_requires_known_producer(self):
        from chd_stunet.training.continuation import _check_contract
        from chd_stunet.training.contracts import digest
        old = copy.deepcopy(self.signature)
        old['producer_ast_sha256'] = {'synthetic.py': 'old'}
        old['identity'] = digest({k:v for k,v in old.items() if k != 'identity'})
        with self.assertRaisesRegex(ValueError, 'Unknown continuation producer'):
            _check_contract(old, self.signature)
        with patch('chd_stunet.training.continuation.LEGACY_053_SOURCES', {digest(old['producer_ast_sha256'])}):
            _check_contract(old, self.signature)
            old['recipe']['batch_size'] = 8
            old['identity'] = digest({k:v for k,v in old.items() if k != 'identity'})
            with self.assertRaisesRegex(ValueError, 'recipe differs'):
                _check_contract(old, self.signature)

    def test_legacy_user_limit_survives_even_after_pressure(self):
        from chd_stunet.training.continuation import read_continuation
        control = self.source / 'control'
        control.mkdir()
        (control/'controller.json').write_text(json.dumps({'pressure_recoveries': 2, 'worker_cap': 1}))
        for value in (None, 0, 7):
            with self.subTest(value=value):
                (self.source/'effective_config.json').write_text(json.dumps({'requested_workers': value}))
                controller = read_continuation(self.source, self.signature, self.settings)['controller']
                self.assertEqual(controller['worker_limit'], 1 if value is None else value)
                self.assertEqual(controller['worker_cap'], 1)

    def test_legacy_resume_without_flag_recovers_provable_operator_limit(self):
        from chd_stunet.training.continuation import read_continuation
        control = self.source / 'control'
        control.mkdir()
        (self.source/'effective_config.json').write_text(json.dumps({'requested_workers': None}))
        for recoveries, expected in ((0, 2), (1, 2)):
            with self.subTest(recoveries=recoveries):
                (control/'controller.json').write_text(json.dumps({'pressure_recoveries': recoveries, 'worker_cap': 2}))
                result = read_continuation(self.source, self.signature, self.settings)
                self.assertEqual(result['controller']['worker_limit'], expected)

    def test_supervisor_history_separates_automatic_pressure_cuts_from_operator_caps(self):
        from chd_stunet.training.continuation import _legacy_worker_limit
        folder = self.source / 'control/logs'
        folder.mkdir(parents=True)

        def launch(seq, requested, applied):
            return dict(seq=seq, event='parallel.launch',
                        supply=dict(requested_total_workers=requested, applied_total_workers=applied))

        def pressure(seq, count, cap):
            return dict(seq=seq, event='parallel.pressure',
                        controller=dict(pressure_recoveries=count, worker_cap=cap))

        for first in (None, 32):
            events = [launch(1, first, 28), pressure(2, 1, 14), launch(3, 14, 14),
                      pressure(4, 2, 7), launch(5, 7, 6), launch(1, 7, 7)]
            for extra, cap, limit, source in [
                    ([], 7, first, 'supervisor_history'),
                    ([launch(1, 5, 5)], 5, 5, 'supervisor_history'),
                    ([launch(13, 5, 5)], 5, 5, 'incomplete_history_kept_cap')]:
                with self.subTest(first=first, extra=extra):
                    (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events+extra))
                    self.assertEqual(_legacy_worker_limit(self.source, dict(pressure_recoveries=2, worker_cap=cap)),
                                     (limit, source))

    def test_raw_snapshot_tamper_is_rejected_even_when_ast_is_unchanged(self):
        from chd_stunet.training.continuation import read_continuation
        path=self.source/'code/run.py';path.write_bytes(path.read_bytes()+b'\n# changed frozen bytes\n')
        with self.assertRaisesRegex(ValueError,'checksum differs'):read_continuation(self.source,self.signature,self.settings)

    def test_pressure_worker_cap_is_not_reset_by_import(self):
        from chd_stunet.training.continuation import read_continuation,install_continuation
        control=self.source/'control';control.mkdir()
        (control/'controller.json').write_text(json.dumps(dict(pressure_recoveries=2,worker_cap=0,reserve_at_entry=123)))
        bundle=read_continuation(self.source,self.signature,self.settings)
        output=self.root/'new';output.mkdir();install_continuation(output,bundle,self.signature)
        controller=json.loads((output/'control/controller.json').read_text())
        self.assertEqual((controller['pressure_recoveries'],controller['worker_cap'],controller['reserve_at_entry']),(2,0,123))


if __name__=='__main__':unittest.main()
