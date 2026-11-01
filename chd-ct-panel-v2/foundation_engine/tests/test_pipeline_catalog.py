"""Fresh catalog tests: real files only, with no CT, network or GPU work."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import subprocess
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import pipeline_catalog as catalog


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = self.root / 'runtime'
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        (self.raw / 'original.txt').write_text('keep raw', encoding='utf-8')
        self.index = self.runtime / 'pipeline_v2' / 'index.json'
        self.counter = 0

    def job(self, stage='prepare', *, upstream=None, input_job=None, **changes):
        self.counter += 1
        job = self.root / 'results' / 'pipeline_v2' / stage / ('job' + str(self.counter))
        job.mkdir(parents=True)
        plan = dict(catalog_schema=2, tasks=[stage], runtime=str(self.runtime),
                    directory=str(job), results=str(job.parent), workspace=str(job / 'workspace'),
                    data=str(self.raw), name='任务' + str(self.counter),
                    created=f'2026-10-27T12:00:{self.counter:02d}+00:00',
                    upstream_jobs=[str(p) for p in (upstream or [])])
        if input_job:
            plan['input_job'] = str(input_job)
        plan.update(changes)
        self.plan(job, plan)
        return job

    def plan(self, job, plan):
        write(job / 'plan.json', plan)
        write(job / 'state.json', dict(status='completed', plan_sha256=hashlib.sha256((job / 'plan.json').read_bytes()).hexdigest()))

    def read_index(self):
        return json.loads(self.index.read_text(encoding='utf-8'))

    def test_empty_history_is_read_only_and_never_reads_old_registries(self):
        self.assertEqual(catalog.history(self.runtime), [])
        self.assertIsNone(catalog.latest(self.runtime))
        self.assertFalse(self.runtime.exists())
        old = self.root / 'old-job'
        write(self.runtime / 'history.json', dict(jobs=[str(old)]))
        write(self.runtime / 'last-job.json', dict(directory=str(old)))
        write(self.runtime / 'logs' / 'legacy.json', dict(work=str(old)))
        before = {str(p): p.read_bytes() for p in self.runtime.rglob('*') if p.is_file()}
        original = Path.read_text
        reads = []

        def read(path, *args, **kwargs):
            reads.append(path)
            if path in {self.runtime / 'history.json', self.runtime / 'last-job.json', self.runtime / 'logs' / 'legacy.json'}:
                self.fail('legacy registry was read')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', read), patch.object(Path, 'glob', side_effect=AssertionError('no scanning')), \
                patch.object(Path, 'rglob', side_effect=AssertionError('no recursive scanning')), \
                patch('task_runtime.legacy_configs', side_effect=AssertionError('no legacy fallback')):
            self.assertEqual(catalog.history(self.runtime), [])
            self.assertIsNone(catalog.latest(self.runtime))
        self.assertTrue(reads)
        self.assertFalse((self.runtime / 'pipeline_v2').exists())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.runtime.rglob('*') if p.is_file()})

    def test_registration_lists_only_indexed_stages_and_selection(self):
        old = self.job(name='not indexed')
        jobs = [self.job(stage) for stage in ['prepare', 'train', 'predict', 'assessment']]
        for job in jobs:
            catalog.register(self.runtime, job)
        with patch.object(catalog, 'active_processes', return_value=[]) as probe:
            rows = catalog.history(self.runtime)
        self.assertEqual({row['directory'] for row in rows}, {str(p) for p in jobs})
        self.assertNotIn(str(old), {row['directory'] for row in rows})
        self.assertEqual({row['stage'] for row in rows}, catalog.STAGES)
        self.assertTrue(all(set(row) == {'directory', 'name', 'status', 'workspace', 'created', 'stage', 'upstream_jobs'} for row in rows))
        self.assertEqual({call.args[0] for call in probe.call_args_list}, set(jobs))
        self.assertEqual(catalog.latest(self.runtime), jobs[-1])
        catalog.select(self.runtime, jobs[0])
        self.assertEqual(catalog.latest(self.runtime), jobs[0])
        with self.assertRaises(ValueError):
            catalog.select(self.runtime, old)
        self.assertEqual(catalog.latest(self.runtime), jobs[0])

    def test_status_and_corrupt_neighbours_are_isolated(self):
        good, broken = self.job(), self.job('train')
        catalog.register(self.runtime, good)
        catalog.register(self.runtime, broken)
        index = self.read_index()
        index['jobs']['damaged entry'] = ['not a record']
        malformed = dict(index['jobs'][str(good)], created=['bad type'])
        index['jobs'][str(self.root / 'bad')] = malformed
        write(self.index, index)
        (broken / 'plan.json').write_text('{', encoding='utf-8')
        with patch.object(catalog, 'active_processes', return_value=[dict(pid=1)]) as probe:
            rows = catalog.history(self.runtime)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['directory']: r['status'] for r in rows},
                         {str(good): 'running', str(broken): '记录缺失或损坏'})
        probe.assert_called_once_with(good)
        self.assertEqual(catalog.latest(self.runtime), good)
        write(good / 'state.json', dict(status='queued', plan_sha256=hashlib.sha256((good / 'plan.json').read_bytes()).hexdigest()))
        with patch.object(catalog, 'active_processes', return_value=[]):
            self.assertEqual(next(r for r in catalog.history(self.runtime) if r['directory'] == str(good))['status'], 'stale')
        write(good / 'state.json', ['invalid state'])
        self.assertEqual(next(r for r in catalog.history(self.runtime) if r['directory'] == str(good))['status'], '记录缺失或损坏')

    def test_upstream_must_be_indexed_new_job(self):
        old = self.job()
        downstream = self.job('train', upstream=[old])
        with self.assertRaisesRegex(ValueError, '新版索引'):
            catalog.register(self.runtime, downstream)
        self.assertEqual(catalog.history(self.runtime), [])

    def test_concurrent_registration_keeps_every_indexed_job(self):
        jobs = [self.job() for _ in range(6)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda job: catalog.register(self.runtime, job), jobs))
        self.assertEqual(set(self.read_index()['jobs']), {str(job) for job in jobs})
        self.assertEqual(len(catalog.history(self.runtime)), len(jobs))

    def nested_job(self, outer):
        job = self.job('assessment')
        plan = json.loads((job / 'plan.json').read_text(encoding='utf-8'))
        nested = outer / 'workspace' / 'pipeline_v2' / 'assessment' / 'inner'
        nested.parent.mkdir(parents=True)
        job.rename(nested)
        plan.update(directory=str(nested), results=str(nested.parent), workspace=str(nested / 'workspace'))
        self.plan(nested, plan)
        return nested

    def test_overlapping_ownership_is_rejected_in_both_registration_orders(self):
        for inner_first in [False, True]:
            with self.subTest(inner_first=inner_first):
                outer = self.job()
                inner = self.nested_job(outer)
                first, rejected = (inner, outer) if inner_first else (outer, inner)
                catalog.register(self.runtime, first)
                with self.assertRaisesRegex(ValueError, '重叠'):
                    catalog.register(self.runtime, rejected)
                self.assertTrue(outer.is_dir())
                self.assertTrue(inner.is_dir())

    def test_trash_fails_closed_for_preexisting_nested_registered_job(self):
        outer = self.job()
        inner = self.nested_job(outer)
        catalog.register(self.runtime, outer)
        index = self.read_index()
        index['jobs'][str(inner)] = catalog._plan_record(self.runtime, inner)
        write(self.index, index)
        with self.assertRaisesRegex(ValueError, '重叠'):
            catalog.trash(self.runtime, outer)
        self.assertTrue(outer.is_dir())
        self.assertTrue(inner.is_dir())

    def test_trash_protects_another_recorded_recovery_directory(self):
        outer, other = self.job(), self.job('train')
        catalog.register(self.runtime, outer)
        catalog.register(self.runtime, other)
        destination = catalog.trash(self.runtime, other)
        recovery = outer / 'workspace' / 'saved-recovery'
        recovery.parent.mkdir()
        destination.rename(recovery)
        index = self.read_index()
        index['jobs'][str(other)]['trash_directory'] = str(recovery)
        write(self.index, index)
        with self.assertRaisesRegex(ValueError, '重叠'):
            catalog.trash(self.runtime, outer)
        self.assertTrue(recovery.is_dir())

    def test_corrupt_top_level_index_does_not_fallback_or_get_overwritten(self):
        job = self.job()
        self.index.parent.mkdir(parents=True)
        self.index.write_text('{broken', encoding='utf-8')
        self.assertEqual(catalog.history(self.runtime), [])
        self.assertIsNone(catalog.latest(self.runtime))
        with self.assertRaises(ValueError):
            catalog.register(self.runtime, job)
        self.assertEqual(self.index.read_text(encoding='utf-8'), '{broken')

    def test_frozen_plan_schema_stage_runtime_and_layout_are_required(self):
        cases = [dict(catalog_schema=1), dict(tasks=['prepare', 'train']), dict(tasks=['environment']),
                 dict(runtime=str(self.root / 'other-runtime')), dict(results=str(self.root / 'raw')),
                 dict(workspace=str(self.root / 'raw')), dict(results_root=str(self.root / 'elsewhere'))]
        for changes in cases:
            with self.subTest(changes=changes):
                job = self.job(**changes)
                with self.assertRaises((ValueError, KeyError)):
                    catalog.register(self.runtime, job)
        job = self.job()
        (job / 'plan.json').write_text((job / 'plan.json').read_text(encoding='utf-8') + ' ', encoding='utf-8')
        with self.assertRaises(ValueError):
            catalog.register(self.runtime, job)
        self.assertEqual(catalog.history(self.runtime), [])

    def test_registration_is_idempotent_but_cannot_rebind_frozen_identity(self):
        job = self.job()
        catalog.register(self.runtime, job)
        catalog.register(self.runtime, job)
        self.assertEqual(len(self.read_index()['jobs']), 1)
        plan = json.loads((job / 'plan.json').read_text(encoding='utf-8'))
        plan['name'] = 'changed'
        self.plan(job, plan)
        with self.assertRaises(ValueError):
            catalog.register(self.runtime, job)
        with self.assertRaises(ValueError):
            catalog.select(self.runtime, job)
        with self.assertRaises(ValueError):
            catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())

    def test_dependency_chain_blocks_upstream_even_if_downstream_plan_corrupt(self):
        prepare = self.job()
        train = self.job('train', upstream=[prepare])
        predict = self.job('predict', input_job=train)
        for job in [prepare, train, predict]:
            catalog.register(self.runtime, job)
        self.assertEqual([r['directory'] for r in catalog.dependents(self.runtime, prepare)], [str(train)])
        self.assertEqual([r['directory'] for r in catalog.dependents(self.runtime, train)], [str(predict)])
        for job in [prepare, train]:
            with self.assertRaisesRegex(ValueError, '下游'):
                catalog.trash(self.runtime, job)
        (train / 'plan.json').write_text('broken', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '下游'):
            catalog.trash(self.runtime, prepare)
        self.assertTrue(prepare.is_dir())

    def test_missing_or_replaced_downstream_root_keeps_its_upstream_blocker(self):
        prepare = self.job()
        train = self.job('train', upstream=[prepare])
        catalog.register(self.runtime, prepare)
        catalog.register(self.runtime, train)
        moved = train.parent.with_name('moved-train')
        train.parent.rename(moved)
        for replaced in [False, True]:
            with self.subTest(replaced=replaced):
                if replaced:
                    train.parent.mkdir()
                self.assertEqual([r['directory'] for r in catalog.dependents(self.runtime, prepare)], [str(train)])
                with self.assertRaisesRegex(ValueError, '下游'):
                    catalog.trash(self.runtime, prepare)
        self.assertTrue(prepare.is_dir())
        self.assertTrue((moved / train.name).is_dir())

    def test_trash_is_recoverable_and_leaves_raw_runtime_and_unindexed_jobs(self):
        code = self.root / 'original-code'
        code.mkdir()
        (code / 'keep.py').write_text('# unchanged', encoding='utf-8')
        write(self.runtime / 'history.json', {'legacy': 'keep'})
        old = self.job(name='old unindexed')
        prepare = self.job()
        train = self.job('train', upstream=[prepare])
        for job in [prepare, train]:
            catalog.register(self.runtime, job)
        (train / 'workspace').mkdir()
        (train / 'workspace' / 'checkpoint.bin').write_bytes(b'checkpoint')
        destination = catalog.trash(self.runtime, train)
        self.assertEqual(destination.parent, train.parent / '.trash')
        self.assertEqual((destination / 'workspace' / 'checkpoint.bin').read_bytes(), b'checkpoint')
        self.assertFalse(train.exists())
        self.assertEqual([r['directory'] for r in catalog.history(self.runtime)], [str(prepare)])
        self.assertEqual(catalog.latest(self.runtime), prepare)
        self.assertEqual(catalog.dependents(self.runtime, prepare), [])
        record = self.read_index()['jobs'][str(train)]
        self.assertEqual(record['state'], 'trashed')
        self.assertEqual(record['trash_directory'], str(destination))
        with self.assertRaises(ValueError):
            catalog.trash(self.runtime, train)
        catalog.trash(self.runtime, prepare)
        self.assertIsNone(catalog.latest(self.runtime))
        self.assertTrue(old.is_dir())
        self.assertEqual((self.raw / 'original.txt').read_text(encoding='utf-8'), 'keep raw')
        self.assertEqual((code / 'keep.py').read_text(encoding='utf-8'), '# unchanged')
        self.assertEqual(json.loads((self.runtime / 'history.json').read_text(encoding='utf-8')), {'legacy': 'keep'})

    def test_active_process_blocks_without_signalling_or_moving(self):
        job = self.job()
        catalog.register(self.runtime, job)
        with patch.object(catalog, 'active_processes', return_value=[dict(pid=321)]), \
                patch.object(os, 'kill', side_effect=AssertionError('must not signal')):
            with self.assertRaisesRegex(ValueError, '活动进程'):
                catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())
        self.assertEqual(self.read_index()['jobs'][str(job)]['state'], 'live')

    def test_execution_lock_blocks_trash_before_launch_can_race(self):
        job = self.job()
        catalog.register(self.runtime, job)
        with catalog._file_lock(self.runtime / 'pipeline_v2' / '.baseline_job.lock'):
            with self.assertRaises(ValueError):
                catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())

    def test_unregistered_and_tampered_out_of_root_paths_are_rejected(self):
        job = self.job()
        catalog.register(self.runtime, job)
        for outsider in [self.raw, self.runtime, job.parent, job / '..' / job.name]:
            with self.subTest(outsider=outsider), self.assertRaises(ValueError):
                catalog.trash(self.runtime, outsider)
        index = self.read_index()
        index['jobs'][str(job)]['results_root'] = str(self.raw)
        write(self.index, index)
        with self.assertRaises(ValueError):
            catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())
        self.assertTrue(self.raw.is_dir())

    def test_replaced_directory_identity_is_not_trusted(self):
        job = self.job()
        catalog.register(self.runtime, job)
        backup = job.with_name(job.name + '-moved')
        job.rename(backup)
        job.mkdir()
        for name in ['plan.json', 'state.json']:
            (job / name).write_bytes((backup / name).read_bytes())
        with self.assertRaises(ValueError):
            catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())
        self.assertTrue(backup.is_dir())

    def test_symlink_job_is_never_moved(self):
        job = self.job()
        catalog.register(self.runtime, job)
        backup = job.with_name(job.name + '-source')
        job.rename(backup)
        try:
            job.symlink_to(backup, target_is_directory=True)
        except OSError as error:
            backup.rename(job)
            self.skipTest('OS does not permit directory symlinks: ' + str(error))
        with self.assertRaises(ValueError):
            catalog.trash(self.runtime, job)
        self.assertTrue(backup.is_dir())
        self.assertTrue(job.is_symlink())

    def test_symlink_recycle_directory_is_rejected(self):
        job = self.job()
        catalog.register(self.runtime, job)
        try:
            (job.parent / '.trash').symlink_to(self.raw, target_is_directory=True)
        except OSError as error:
            self.skipTest('OS does not permit directory symlinks: ' + str(error))
        with self.assertRaises(ValueError):
            catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())
        self.assertEqual(list(self.raw.iterdir()), [self.raw / 'original.txt'])

    @unittest.skipUnless(os.name == 'nt', 'Windows directory junction safety')
    def test_windows_junction_recycle_directory_is_rejected(self):
        job = self.job()
        catalog.register(self.runtime, job)
        link = job.parent / '.trash'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(self.raw)],
                                capture_output=True, text=True)
        if result.returncode:
            self.skipTest('Windows junction creation unavailable: ' + result.stderr)
        try:
            with self.assertRaises(ValueError):
                catalog.trash(self.runtime, job)
            self.assertTrue(job.is_dir())
            self.assertEqual(list(self.raw.iterdir()), [self.raw / 'original.txt'])
        finally:
            link.rmdir()

    def assert_linked_process_metadata_is_isolated(self, job):
        # The real unsafe helper is never invoked; the test itself must not read
        # any old/unindexed content even when run against a regressed guard.
        with patch.object(catalog, 'active_processes', side_effect=AssertionError('redirected metadata reached probe')):
            self.assertEqual(catalog.history(self.runtime)[0]['status'], '记录缺失或损坏')
            self.assertIsNone(catalog.latest(self.runtime))
            with self.assertRaises(ValueError):
                catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())

    @unittest.skipUnless(os.name == 'nt', 'Windows process metadata junction safety')
    def test_windows_junction_process_directory_is_never_probed(self):
        job = self.job()
        catalog.register(self.runtime, job)
        unindexed = self.root / 'unindexed-processes'
        unindexed.mkdir()
        (unindexed / 'foreign.json').write_text('must not read', encoding='utf-8')
        link = job / 'processes'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(unindexed)], capture_output=True, text=True)
        if result.returncode:
            self.skipTest('Windows junction creation unavailable: ' + result.stderr)
        try:
            self.assert_linked_process_metadata_is_isolated(job)
        finally:
            link.rmdir()

    @unittest.skipUnless(os.name == 'nt', 'Windows process metadata junction safety')
    def test_windows_junction_process_entry_is_never_probed(self):
        job = self.job()
        catalog.register(self.runtime, job)
        unindexed = self.root / 'unindexed-process-entry'
        unindexed.mkdir()
        (job / 'processes').mkdir()
        link = job / 'processes' / 'foreign.json'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(unindexed)], capture_output=True, text=True)
        if result.returncode:
            self.skipTest('Windows junction creation unavailable: ' + result.stderr)
        try:
            self.assert_linked_process_metadata_is_isolated(job)
        finally:
            link.rmdir()

    def test_symlink_process_json_is_never_probed(self):
        job = self.job()
        catalog.register(self.runtime, job)
        foreign = self.root / 'unindexed-foreign.json'
        foreign.write_text('must not read', encoding='utf-8')
        (job / 'processes').mkdir()
        try:
            (job / 'processes' / 'linked.json').symlink_to(foreign)
        except OSError as error:
            self.skipTest('OS does not permit file symlinks: ' + str(error))
        self.assert_linked_process_metadata_is_isolated(job)

    def test_hardlinked_process_json_is_never_probed(self):
        job = self.job()
        catalog.register(self.runtime, job)
        foreign = self.root / 'unindexed-hardlink.json'
        foreign.write_text('must not read', encoding='utf-8')
        (job / 'processes').mkdir()
        os.link(foreign, job / 'processes' / 'linked.json')
        self.assert_linked_process_metadata_is_isolated(job)

    def test_intent_write_failure_never_moves_job(self):
        job = self.job()
        catalog.register(self.runtime, job)
        with patch.object(catalog, '_write_index', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())
        self.assertEqual(self.read_index()['jobs'][str(job)]['state'], 'live')

    def test_rename_failure_restores_live_registration(self):
        job = self.job()
        catalog.register(self.runtime, job)
        with patch.object(catalog.os, 'rename', side_effect=OSError('rename denied')):
            with self.assertRaises(OSError):
                catalog.trash(self.runtime, job)
        self.assertTrue(job.is_dir())
        self.assertEqual(self.read_index()['jobs'][str(job)]['state'], 'live')
        self.assertEqual(catalog.latest(self.runtime), job)

    def test_failure_after_rename_preserves_hidden_tombstone_and_destination(self):
        job = self.job()
        catalog.register(self.runtime, job)
        original = catalog._write_index
        calls = []

        def fail_final(runtime, index):
            calls.append(index['jobs'][str(job)]['state'])
            if len(calls) == 2:
                raise OSError('final write failed')
            return original(runtime, index)

        with patch.object(catalog, '_write_index', side_effect=fail_final):
            with self.assertRaisesRegex(OSError, '可恢复目录'):
                catalog.trash(self.runtime, job)
        record = self.read_index()['jobs'][str(job)]
        self.assertEqual(record['state'], 'trashing')
        self.assertFalse(job.exists())
        destination = Path(record['trash_directory'])
        self.assertTrue((destination / 'plan.json').is_file())
        self.assertEqual(catalog.history(self.runtime), [])
        self.assertIsNone(catalog.latest(self.runtime))
        self.assertEqual(calls, ['trashing', 'trashed'])

    def test_completed_rename_then_interrupt_retains_recovery_intent(self):
        job = self.job()
        catalog.register(self.runtime, job)
        rename = os.rename

        def move_then_interrupt(source, destination):
            rename(source, destination)
            raise KeyboardInterrupt('interrupted at successful rename boundary')

        with patch.object(catalog.os, 'rename', side_effect=move_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                catalog.trash(self.runtime, job)
        record = self.read_index()['jobs'][str(job)]
        self.assertEqual(record['state'], 'trashing')
        self.assertFalse(job.exists())
        destination = Path(record['trash_directory'])
        self.assertTrue((destination / 'plan.json').is_file())
        self.assertEqual(catalog._identity(destination), record['job_identity'])
        self.assertEqual(catalog.history(self.runtime), [])
        self.assertIsNone(catalog.latest(self.runtime))

    def test_failed_rename_and_failed_rollback_remain_recoverable_and_block_upstream(self):
        prepare = self.job()
        train = self.job('train', upstream=[prepare])
        catalog.register(self.runtime, prepare)
        catalog.register(self.runtime, train)
        original = catalog._write_index
        calls = []

        def fail_rollback(runtime, index):
            calls.append(1)
            if len(calls) == 2:
                raise OSError('rollback write failed')
            return original(runtime, index)

        with patch.object(catalog.os, 'rename', side_effect=OSError('rename denied')), \
                patch.object(catalog, '_write_index', side_effect=fail_rollback):
            with self.assertRaises(OSError):
                catalog.trash(self.runtime, train)
        record = self.read_index()['jobs'][str(train)]
        self.assertEqual(record['state'], 'trashing')
        self.assertTrue(train.is_dir())
        self.assertFalse(Path(record['trash_directory']).exists())
        self.assertEqual([row['directory'] for row in catalog.dependents(self.runtime, prepare)], [str(train)])
        with self.assertRaisesRegex(ValueError, '下游'):
            catalog.trash(self.runtime, prepare)

    def test_catalog_atomic_replace_failure_keeps_previous_index(self):
        first, second = self.job(), self.job('train')
        catalog.register(self.runtime, first)
        before = self.index.read_bytes()
        with patch.object(catalog.os, 'replace', side_effect=OSError('cannot replace index')):
            with self.assertRaises(OSError):
                catalog.register(self.runtime, second)
        self.assertEqual(self.index.read_bytes(), before)
        self.assertEqual(catalog.latest(self.runtime), first)
        self.assertEqual(list(self.index.parent.glob('.index-*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
