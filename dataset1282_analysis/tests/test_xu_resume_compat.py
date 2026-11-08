import copy
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def legacy_encoder(prepared, targets, output):
    import torch
    from dataset1282_analysis.contracts import atomic_json
    from dataset1282_analysis.xu_blood import _contract, _train_stage, blood_cohort
    from dataset1282_analysis.xu_blood_models import SliceUNet, BloodSequence
    from dataset1282_analysis.xu_geometry import PreparedCache
    from dataset1282_analysis.xu_targets import TargetStore, blood_ids

    torch.set_num_threads(2)
    torch.manual_seed(27)
    cache = PreparedCache(prepared, allow_technical_subset=True)
    store = TargetStore(Path(targets), cache)
    eligible = blood_cohort(cache, blood_ids(cache), 'marginal', store)
    contract = _contract(cache, store.identity, 'marginal', 'smoke', 27, eligible)
    out = Path(output)
    out.mkdir()
    atomic_json(out/'blood-contract.json', contract)
    encoder = SliceUNet('smoke')
    sequence = BloodSequence(encoder.width)
    _train_stage(cache, store, contract, out, encoder, sequence, 'encoder', 'cpu')


class ExactSourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        from dataset1282_analysis.contracts import atomic_json, sha256_file

        parent = ROOT/'local-validation/resume-compat-tests'
        parent.mkdir(parents=True, exist_ok=True)
        temporary = TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.audit = self.out/'audit.json'
        source = ROOT/'src/dataset1282_analysis'
        self.stored = {'contracts.py': sha256_file(source/'contracts.py')}
        self.current = dict(self.stored, **{'xu_resume_compat.py': sha256_file(source/'xu_resume_compat.py')})
        self.prefix = 'src/dataset1282_analysis/'
        self.manifest = dict(schema='xu-resume-compat-1', transitions=[dict(
            id='real-source-fixture', review='Exact real file digests for compatibility boundary testing',
            sources=[dict(path_prefix=self.prefix, stored=self.stored, current=self.current)])])
        atomic_json(self.audit, self.manifest)

    def compatible(self, stored, current, prefix=None):
        from dataset1282_analysis.xu_resume_compat import source_compatible
        return source_compatible(stored, current, self.prefix if prefix is None else prefix,
                                 audit_path=self.audit)

    def test_exact_full_mapping_allows_only_the_audited_addition(self):
        self.assertTrue(self.compatible(self.stored, self.current))
        self.assertTrue(self.compatible(self.current, self.current))
        self.assertFalse(self.compatible(self.stored, self.current, ''))
        self.assertFalse(self.compatible(self.current, self.stored))
        self.assertFalse(self.compatible(self.stored, dict(self.current, extra=self.stored['contracts.py'])))
        altered = dict(self.current, **{'contracts.py': self.current['xu_resume_compat.py']})
        self.assertFalse(self.compatible(self.stored, altered))
        self.assertFalse(self.compatible(self.stored, {'xu_resume_compat.py': self.current['xu_resume_compat.py']}))

    def test_transition_components_cannot_be_combined(self):
        from dataset1282_analysis.contracts import atomic_json
        other = {'xu_resume_compat.py': self.current['xu_resume_compat.py']}
        self.manifest['transitions'].append(dict(id='other', review='Independent complete file map',
            sources=[dict(path_prefix=self.prefix, stored=other, current=dict(other, second=other['xu_resume_compat.py']))]))
        atomic_json(self.audit, self.manifest)
        self.assertFalse(self.compatible(self.stored, dict(self.current, second=other['xu_resume_compat.py'])))

    def test_contract_and_receipt_identity_remain_unchanged(self):
        from dataset1282_analysis.contracts import atomic_json, digest, read_json, sha256_file
        from dataset1282_analysis.xu_pipeline import _run_stage, _stage_spec
        from dataset1282_analysis.xu_pipeline_stages import Stage
        from dataset1282_analysis.xu_resume_compat import preserve_contract

        stored = dict(source=self.stored, seed=27, targets='real-input-identity')
        current = dict(stored, source=self.current)
        marker = self.out/'run.json'
        atomic_json(marker, stored)
        before = sha256_file(marker)
        retained = preserve_contract(read_json(marker), current, self.out, field='source',
                                     path_prefix=self.prefix, audit_path=self.audit)
        self.assertEqual(retained, stored)
        self.assertEqual(sha256_file(marker), before)
        sidecars = list((self.out/'.runtime/resume-compat').glob('*.json'))
        self.assertEqual(len(sidecars), 1)
        evidence = read_json(sidecars[0])
        self.assertEqual(evidence['execution_contract'], digest(current))
        self.assertEqual(evidence['frozen_contract'], digest(stored))
        self.assertEqual(evidence['audit_sha256'], sha256_file(self.audit))
        from dataset1282_analysis.xu_pipeline import _sources
        args = SimpleNamespace(technical_subset=True, execution_implementation=_sources(True))
        stage = Stage('crop64', ('train',), (), skip_reason=dict(reason='supplied_frozen_locator'))
        receipt = _run_stage(stage, args, self.out, {}, retained, digest(retained), {}, {}, 1)
        self.assertEqual(receipt['spec'], _stage_spec(stage, digest(stored), None))
        receipt_path = self.out/'receipts/crop64.json'
        receipt_before = sha256_file(receipt_path)
        self.assertEqual(_run_stage(stage, args, self.out, {}, retained, digest(retained), {}, {}, 1), receipt)
        self.assertEqual(sha256_file(receipt_path), receipt_before)
        preserve_contract(stored, current, self.out, field='source', path_prefix=self.prefix, audit_path=self.audit)
        self.assertEqual(len(list((self.out/'.runtime/resume-compat').glob('*.json'))), 1)

    def test_all_non_source_changes_are_refused(self):
        from dataset1282_analysis.xu_resume_compat import preserve_contract
        stored = dict(source=self.stored, seed=27, targets='input', device=['cuda:2'])
        current = dict(stored, source=self.current)
        for changed in (dict(current, seed=28), dict(current, targets='changed'),
                        dict(current, device=['cuda:3']), dict(current, new_flag=True),
                        {k: v for k, v in current.items() if k != 'seed'}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'identical inputs'):
                preserve_contract(stored, changed, self.out, field='source',
                                  path_prefix=self.prefix, audit_path=self.audit)
        self.assertFalse((self.out/'.runtime/resume-compat').exists())

    def test_missing_malformed_or_unreviewed_audit_is_refused(self):
        from dataset1282_analysis.contracts import atomic_json
        self.audit.unlink()
        self.assertFalse(self.compatible(self.stored, self.current))
        invalid = copy.deepcopy(self.manifest)
        invalid['transitions'][0]['sources'][0]['current']['contracts.py'] = 'invalid'
        atomic_json(self.audit, invalid)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.compatible(self.stored, self.current)
        invalid = copy.deepcopy(self.manifest)
        invalid['transitions'][0]['review'] = ''
        atomic_json(self.audit, invalid)
        with self.assertRaisesRegex(ValueError, 'transition'):
            self.compatible(self.stored, self.current)

    def test_execution_rejects_new_source_or_audit_changes(self):
        from dataset1282_analysis.contracts import atomic_json, sha256_file
        from dataset1282_analysis.xu_resume_compat import assert_execution_sources
        audit_sha = sha256_file(self.audit)
        assert_execution_sources(self.current, self.current, audit_sha, audit_path=self.audit)
        with self.assertRaisesRegex(ValueError, 'Implementation changed'):
            assert_execution_sources(self.stored, self.current, audit_sha, audit_path=self.audit)
        atomic_json(self.audit, dict(schema='xu-resume-compat-1', transitions=[]))
        with self.assertRaisesRegex(ValueError, 'audit changed'):
            assert_execution_sources(self.current, self.current, audit_sha, audit_path=self.audit)

    def test_legacy_blood_contract_cannot_bypass_current_source_audit(self):
        from dataset1282_analysis.xu_blood import _matches, _source, _V1_SOURCE
        current = dict(method='xu-blood-v2',supervision='marginal',seed=27,source=_source())
        legacy = dict(current,method='xu-blood-v1',source=_V1_SOURCE)
        del legacy['supervision']
        self.assertTrue(_matches(legacy,current))
        changed = copy.deepcopy(current)
        changed['source']['xu_blood_supply.py'] = self.current['xu_resume_compat.py']
        self.assertFalse(_matches(legacy,changed))
        changed = copy.deepcopy(current)
        changed['source'].pop('xu_supply.py')
        self.assertFalse(_matches(legacy,changed))


class LegacyEncoderResumeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('XU_LEGACY_CODE'), 'Set XU_LEGACY_CODE to the real c5ced51 checkout')
    def test_real_legacy_encoder_finishes_sequence_without_retraining_encoder(self):
        import torch
        from dataset1282_analysis.contracts import atomic_json, read_json, sha256_file
        from dataset1282_analysis.xu_blood import train
        from dataset1282_analysis.xu_targets import build_targets
        from test_xu_targets import on_disk_phantom

        legacy = Path(os.environ['XU_LEGACY_CODE']).resolve()
        parent = ROOT/'local-validation/resume-compat-tests'
        parent.mkdir(parents=True, exist_ok=True)
        out = Path(mkdtemp(prefix='legacy-encoder-', dir=parent))
        cache = on_disk_phantom(out/'prepared')
        targets = out/'targets'
        build_targets(cache, SimpleNamespace(reviews=None, width_mm=1., resume=False), targets)
        blood = out/'blood'
        env = dict(os.environ, PYTHONPATH=str(legacy/'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--legacy-encoder',
                                 str(cache.run_dir), str(targets), str(blood)], env=env,
                                cwd=legacy, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        protected = ('blood-contract.json', 'encoder-last.pt', 'encoder-selected.pt')
        before = {name: sha256_file(blood/name) for name in protected}
        original = torch.load(blood/'encoder-last.pt', map_location='cpu', weights_only=True)
        self.assertEqual(original['epoch'], 1)
        self.assertTrue(original['optimizer']['state'])
        self.assertFalse((blood/'sequence-last.pt').exists())
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        args = SimpleNamespace(targets=targets, profile='smoke', technical_unvalidated=False,
                               seed=27, device='cpu', resume=True, supervision='marginal')
        result = train(cache, args, blood)
        self.assertEqual(result['history']['encoder'], original['history'])
        self.assertEqual(len(result['history']['sequence']), 1)
        self.assertEqual({name: sha256_file(blood/name) for name in protected}, before)
        self.assertEqual(read_json(blood/'blood-contract.json'), original['contract'])
        self.assertTrue(list((blood/'.runtime/resume-compat').glob('*.json')))
        atomic_json(out/'evidence.json', dict(legacy=str(legacy), encoder_epochs=1, sequence_epochs=1,
                                            protected_sha256=before, history=result['history']))
        print('Real legacy encoder resume evidence:', out/'evidence.json', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--legacy-encoder':
        legacy_encoder(*sys.argv[2:])
    else:
        unittest.main()
