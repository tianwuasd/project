from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dataset1282_analysis.contracts import digest, read_json, sha256_file
from dataset1282_analysis.observability.status import operation_lock
from dataset1282_analysis.xu_auto_init import build_auto_targets
from dataset1282_analysis.xu_dispatch import waves
from dataset1282_analysis.xu_fleet import preflight_sizes, work_limits
from dataset1282_analysis.xu_init_extension import bind_extension, eligible_counts, extension_identity
from dataset1282_analysis.xu_pipeline import _check_manifests, _run_stage, _sources, _stage_spec
from dataset1282_analysis.xu_pipeline_stages import Stage, iter_stages
from dataset1282_analysis.xu_supply_budget import stage_minimum
from dataset1282_analysis.xu_targets import build_targets
from dataset1282_analysis.xu_workflow import artifact_identity
from test_xu_targets import on_disk_phantom


TEST_ROOT = Path('D:/projects/back/t17/init-extension')


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.cache = on_disk_phantom(self.base/'prepared')
        self.out = self.base/'run'
        self.out.mkdir()
        self.args = SimpleNamespace(prepared=self.cache.run_dir, profile='smoke', device='cpu',
            supervision='marginal', technical_subset=True, seed=17, crop64=None, crop128=None,
            reviews=None, auto_init=True, init_radius_mm=30., init_margin_percent=20., stop_after=None,
            prefetch=True, max_source_voxels=8000)
        self.contract = dict(request=self.cache.request_digest, artifacts=artifact_identity(self.cache))

    def targets(self):
        build_targets(self.cache, SimpleNamespace(width_mm=1., reviews=None, resume=False), self.out/'targets')
        build_auto_targets(self.cache, SimpleNamespace(radius_mm=30., margin_percent=20., resume=False),
                           self.out/'auto-targets')

    def test_explicit_binding_resume_settings_and_downstream_protection(self):
        with operation_lock(self.out, 'extension-test', self.out):
            binding = bind_extension(self.args, self.out, self.contract)
            self.assertEqual(bind_extension(self.args, self.out, self.contract), binding)
            self.assertEqual(read_json(self.out/'.runtime/auto-init-extension.json'), binding)
            self.args.auto_init = False
            with self.assertRaisesRegex(ValueError, 'explicit'):
                bind_extension(self.args, self.out, self.contract)
            self.args.auto_init = True
            self.args.init_radius_mm = 31.
            with self.assertRaisesRegex(ValueError, 'settings'):
                bind_extension(self.args, self.out, self.contract)
        other = self.base/'occupied'
        (other/'vessels').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'untouched'):
            bind_extension(self.args, other, self.contract)
        self.assertFalse((other/'.runtime/auto-init-extension.json').exists())

    def test_real_books_keep_original_specs_and_route_new_weights(self):
        binding = bind_extension(self.args, self.out, self.contract)
        self.targets()
        before = {p.relative_to(self.out/'targets').as_posix(): sha256_file(p)
                  for p in (self.out/'targets').rglob('*') if p.is_file()}
        self.args.auto_init = False
        original = {s.name: s for s in iter_stages(self.cache, self.args, self.out)}
        self.args.auto_init = True
        extended = {s.name: s for s in iter_stages(self.cache, self.args, self.out)}
        base_id = digest(self.contract)
        for name in original.keys() - {'with-blood', 'vessels'}:
            self.assertEqual(_stage_spec(original[name], base_id), _stage_spec(extended[name], base_id))
        for name in ('init64', 'init128', 'six-way'):
            self.assertIsNotNone(extended[name].skip_reason)
        for name in ('auto-init64', 'auto-init128'):
            self.assertIsNone(extended[name].skip_reason)
            self.assertIn(str(self.out/'auto-targets'), extended[name].argv)
        self.assertIn(str(self.out/'auto-init128/best.pt'), extended['with-blood'].argv)
        self.assertIn(str(self.out/'targets'), extended['blood'].argv)
        self.assertEqual(eligible_counts(self.cache, self.out/'auto-targets', 'marginal'),
                         dict(train=1, development=1))
        self.assertEqual(before, {p.relative_to(self.out/'targets').as_posix(): sha256_file(p)
                                 for p in (self.out/'targets').rglob('*') if p.is_file()})
        self.assertEqual(bind_extension(self.args, self.out, self.contract), binding)
        self.assertEqual(extension_identity(extended['blood'], base_id, binding), base_id)
        self.assertNotEqual(extension_identity(extended['auto-init64'], base_id, binding), base_id)
        records = {'auto-targets': dict(files={'targets.json': sha256_file(self.out/'auto-targets/targets.json')})}
        dependency = Stage('auto-init64', ('train-init', '--targets', str(self.out/'auto-targets')), ())
        _check_manifests(dependency, self.out, records)
        records['auto-targets']['files']['targets.json'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'input changed'):
            _check_manifests(dependency, self.out, records)

    def test_lazy_targets_barrier_required_formal_counts_and_resource_aliases(self):
        build_targets(self.cache, SimpleNamespace(width_mm=1., reviews=None, resume=False), self.out/'targets')
        self.args.profile = 'paper'
        stages = iter_stages(self.cache, self.args, self.out)
        for stage in stages:
            if stage.name == 'auto-targets':
                break
        self.assertFalse((self.out/'auto-targets').exists())
        build_auto_targets(self.cache, SimpleNamespace(radius_mm=30., margin_percent=20., resume=False),
                           self.out/'auto-targets')
        with self.assertRaisesRegex(ValueError, 'requirement failed'):
            next(stages)
        self.args.profile = 'smoke'
        plan = list(iter_stages(self.cache, self.args, self.out))
        groups = list(waves(plan, ['cuda:0', 'cuda:1'], None))
        pair = next(g for g in groups if g[0].name == 'auto-init64')
        self.assertEqual([s.name for s in pair], ['auto-init64', 'auto-init128'])
        for name in ('auto-init64', 'auto-init128'):
            original = name.removeprefix('auto-')
            self.assertEqual(preflight_sizes(name), preflight_sizes(original))
            self.assertEqual(work_limits(name, self.args, self.cache, 2),
                             work_limits(original, self.args, self.cache, 2))
            self.assertEqual(stage_minimum(self.args, name), stage_minimum(self.args, original))
        self.assertEqual(work_limits('auto-six-way', self.args, self.cache, 1), (1, 0))

    def test_original_skipped_receipts_remain_byte_identical(self):
        self.targets()
        self.args.auto_init = False
        self.args.execution_implementation = _sources(True)
        original = {s.name: s for s in iter_stages(self.cache, self.args, self.out)}
        base_id = digest(self.contract)
        before = {}
        for name in ('init64', 'init128', 'six-way'):
            _run_stage(original[name], self.args, self.out, {}, self.contract, base_id, {}, {}, 1)
            before[name] = sha256_file(self.out/'receipts'/(name+'.json'))
        self.args.auto_init = True
        extended = {s.name: s for s in iter_stages(self.cache, self.args, self.out)}
        for name in before:
            _run_stage(extended[name], self.args, self.out, {}, self.contract, base_id, {}, {}, 1)
            self.assertEqual(sha256_file(self.out/'receipts'/(name+'.json')), before[name])


if __name__ == '__main__':
    unittest.main()
