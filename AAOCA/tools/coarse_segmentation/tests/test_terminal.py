"""Real synthetic files and process lifecycle; never load models or patient data."""
import csv
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aaoca_seg import executor, resources, scheduler
from aaoca_seg.schema import LABELS
from aaoca_seg.service_state import CODE_ROOT, file_lock, read_json, write_json
from aaoca_seg.terminal import Back, Exit, Panel
from aaoca_seg.workbench import Workbench, status, validate_settings


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fake_lock = threading.Lock()
        self.base = Path(self.temp.name)
        self.work = Workbench(self.base / 'runtime')
        source = self.base / 'source'
        source.mkdir()
        self.cases = []
        for i in range(3):
            image = source / f'demo{i}.nii.gz'
            volume = nib.Nifti1Image(np.arange(512, dtype=np.float32).reshape(8, 8, 8) - 300,
                                    np.diag([.7, .8, 1., 1.]))
            volume.header.set_xyzt_units('mm')
            nib.save(volume, image)
            self.cases.append({'case_id': f'demo{i}', 'image': str(image)})
        self.manifest = self.base / 'manifest.csv'
        with self.manifest.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['case_id', 'image', 'hu_confirmed', 'spacing_confirmed'])
            writer.writeheader()
            writer.writerows(dict(r, hu_confirmed='true', spacing_confirmed='true') for r in self.cases)
        self.work.save(dict(self.work.settings, manifest=str(self.manifest)))
        self.alloc = {'cpu_limit': 8, 'case_slots': 2, 'threads': 2, 'memory_bytes': 20 * 2**30, 'io_slots': 2}

    def tearDown(self):
        for child in self.work.children:
            child.wait(timeout=10)
        self.temp.cleanup()

    def create(self, steps=None):
        return self.work.create('合成任务', steps or ['plan', 'infer', 'merge'])

    def fake_model(self, command, log, stop, registry):
        source = nib.load(command[command.index('-i')+1])
        root = Path(command[command.index('-o')+1])
        root.mkdir(parents=True, exist_ok=True)
        task = command[command.index('--task')+1]
        for label, name, _, model in LABELS:
            if task == model:
                data = np.zeros(source.shape, dtype=np.uint8)
                data[label % 7, 3, 4] = 1
                nib.save(nib.Nifti1Image(data, source.affine, source.header), root / (name + '.nii.gz'))

    def test_no_dependencies_for_help_or_readonly_empty_history(self):
        root = CODE_ROOT.parents[1] / 'start_segmentation.py'
        result = subprocess.run([sys.executable, '-S', str(root), '--runtime', str(self.base / 'absent'),
                                 '--query', 'history'], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.base / 'absent').exists())

    def test_cancel_form_and_zero_gpu_not_exit(self):
        panel = Panel(self.work.root, ask=Mock(side_effect=['0', 'b', 'q']), emit=lambda _: None)
        self.assertEqual(panel.prompt(''), '0')
        with self.assertRaises(Back):
            panel.prompt('')
        with self.assertRaises(Exit):
            panel.prompt('')
        panel.ask = Mock(side_effect=['任务', '0', '1,2,3', '', 'n'])
        panel.new()
        self.assertEqual(self.work.entries(), [])

    def test_failed_case_log_remains_visible_after_exit(self):
        job = self.create()
        request = read_json(job / 'request.json')
        log = Path(request['output']) / 'cases/demo0/attempt-failed/infer-worker.log'
        log.parent.mkdir(parents=True)
        log.write_text('synthetic detailed failure')
        write_json(Path(request['output']) / 'status.json', {'active': {}, 'cases': [
            {'case_id': 'demo0', 'status': 'failed', 'phase': 'infer', 'log': str(log)}]})
        self.assertIn('synthetic detailed failure', self.work.logs(job))

    def test_corrupt_neighbor_history_does_not_hide_valid_job(self):
        job = self.create(['plan'])
        write_json(self.work.root / 'jobs' / ('b' * 32) / 'request.json', [])
        self.assertEqual(len(self.work.entries()), 1)
        (job / 'state.json').unlink()
        self.assertEqual(self.work.entries()[0]['status'], 'unknown')

    def test_unique_jobs_raw_input_lineage_and_rerun(self):
        job = self.create()
        original = (job / 'request.json').read_bytes()
        write_json(job / 'state.json', {'status': 'completed'})
        other = self.work.rerun(job.name)
        first, second = read_json(job / 'request.json'), read_json(other / 'request.json')
        self.assertNotEqual(first['output'], second['output'])
        self.assertEqual(second['rerun_of'], job.name)
        self.assertEqual(second['cases'], first['cases'])
        self.assertEqual((job / 'request.json').read_bytes(), original)
        self.assertEqual(len(self.work.entries()), 2)

    def test_configuration_and_snapshot_mutation_prevent_resume(self):
        job = self.create(['plan'])
        with (job / 'code/aaoca_seg/schema.py').open('a') as stream:
            stream.write('\n# changed')
        with self.assertRaisesRegex(ValueError, '快照'):
            self.work.start(job)
        (job / 'request.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.work.start(job)

    def test_nan_or_oversized_resource_settings_rejected(self):
        for key, value in [('memory_gib', float('nan')), ('threads', 0), ('workers', 100), ('min_gpu_gib', -1)]:
            with self.assertRaises(ValueError):
                validate_settings(dict(self.work.settings, **{key: value}))

    def test_cgroup_ancestors_and_one_global_thread_budget(self):
        root = self.base / 'groups'
        leaf = root / 'parent/child'
        leaf.mkdir(parents=True)
        membership = self.base / 'cgroup'
        membership.write_text('0::/parent/child\n')
        (root / 'parent/cpu.max').write_text('400000 100000')
        (root / 'parent/memory.max').write_text(str(12 * 2**30))
        (root / 'parent/memory.current').write_text(str(2 * 2**30))
        proc = Mock(); proc.cpu_affinity.return_value = list(range(16))
        with patch('psutil.Process', return_value=proc), patch('psutil.virtual_memory', return_value=Mock(available=64 * 2**30)), patch('psutil.cpu_percent', return_value=[0] * 16):
            result = resources.budget(self.work.settings, root, membership)
        self.assertEqual(result['cpu_quota'], 4)
        self.assertLessEqual(result['threads'] * result['case_slots'] * 2, result['cpu_limit'])
        self.assertLessEqual(result['memory_bytes'], 10 * 2**30)

    def test_gpu_visibility_uuid_and_other_compute_users(self):
        devices = [{'index': 0, 'uuid': 'GPU-a'}, {'index': 1, 'uuid': 'GPU-b'}]
        selected = resources.select_devices({'devices': devices}, [1])
        with self.assertRaises(ValueError):
            resources.select_devices({'devices': devices}, [2])
        with self.assertRaises(ValueError):
            resources.available_devices(selected, {'devices': devices[:1]})
        query = Mock(stdout='GPU-a, 20000, 0\nGPU-b, 22000, 0\n')
        busy = Mock(stdout='GPU-b, 12345\n')
        with patch('aaoca_seg.resources.subprocess.run', side_effect=[query, busy]):
            self.assertEqual(resources.available_devices(selected, {'devices': devices}), [])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,3'}):
            self.assertEqual(resources.child_env(2, {'uuid': 'GPU-b'})['CUDA_VISIBLE_DEVICES'], 'GPU-b')
            self.assertEqual(resources.child_env(2)['CUDA_VISIBLE_DEVICES'], '')

    def infer(self, case=None, attempt=None):
        case = case or self.cases[0]
        attempt = attempt or self.base / 'output/cases/demo0/attempt-1'
        with self.fake_lock, patch.object(executor, 'execute_model', side_effect=self.fake_model):
            executor.infer(case, attempt, 1, 'cpu', self.base / 'stop', {'synthetic': True})
        return attempt / 'inference.json'

    def test_native_decompression_checkpoint_merge_and_source_unchanged(self):
        original = Path(self.cases[0]['image']).read_bytes()
        cp = self.infer()
        self.assertFalse((cp.parent / 'input.nii').exists())
        output = self.base / 'merged'
        executor.merge(self.cases[0], cp, output)
        self.assertTrue(executor.completed(output / 'result.json', self.cases[0], cp))
        self.assertEqual(Path(self.cases[0]['image']).read_bytes(), original)
        result = nib.load(output / 'aaoca_11_labels.nii.gz')
        self.assertTrue(np.allclose(result.affine, nib.load(self.cases[0]['image']).affine))
        self.assertEqual(result.shape, (8, 8, 8))

    def test_changed_source_or_missing_mask_cannot_be_reused(self):
        cp = self.infer()
        record = read_json(cp)
        first = next(iter(record['artifacts']))
        (cp.parent / first).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            executor.checkpoint(cp, self.cases[0])
        image = Path(self.cases[0]['image'])
        image.write_bytes(image.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, '冻结输入'):
            self.infer(attempt=cp.parent.parent / 'attempt-2')

    def test_merge_only_child_references_masks_without_overwriting(self):
        job = self.create()
        request = read_json(job / 'request.json')
        for case in self.cases:
            cp = self.infer(case, Path(request['output']) / 'cases' / case['case_id'] / 'attempt-1')
            write_json(cp.parent.parent / 'latest-inference.json', {'checkpoint': str(cp)})
        write_json(job / 'state.json', {'status': 'completed'})
        child = self.work.rerun(job.name, ['merge'])
        new = read_json(child / 'request.json')
        self.assertEqual(len(new['inputs']), 3)
        self.assertNotEqual(new['output'], request['output'])

    def test_plan_only_does_not_probe_or_wait_for_gpu(self):
        job = self.create(['plan'])
        with patch.object(scheduler, 'gpu_info', side_effect=AssertionError('no GPU')):
            result = scheduler.run(job, allocation=self.alloc, backend={'versions': {'synthetic': '1'}})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['finished'], 3)

    def test_oversized_case_fails_without_model_start(self):
        job = self.create()
        launch = Mock(side_effect=AssertionError('no model'))
        result = scheduler.run(job, allocation=dict(self.alloc, memory_bytes=1), popen=launch,
                               backend={'versions': {}})
        self.assertEqual(result['failed'], 3)
        launch.assert_not_called()

    def test_submit_then_modify_input_rejected_before_gpu(self):
        job = self.create()
        path = Path(self.cases[0]['image'])
        path.write_bytes(path.read_bytes() + b'changed')
        request = read_json(job / 'request.json')
        request['cases'] = request['cases'][:1]
        write_json(job / 'request.json', request)
        with patch.object(scheduler, 'gpu_info', side_effect=AssertionError('no GPU')):
            result = scheduler.run(job, allocation=self.alloc, backend={'versions': {}})
        self.assertEqual(result['failed'], 1)
        self.assertIn('提交后', result['cases'][0]['error'])

    def test_managed_release_reopens_real_launcher_and_setup(self):
        import shutil
        commit = 'a' * 40
        release = self.work.root / 'releases' / commit
        shutil.copytree(CODE_ROOT, release, ignore=shutil.ignore_patterns('__pycache__'))
        write_json(self.work.root / 'release.json', {'path': str(release), 'commit': commit})
        root = CODE_ROOT.parents[1] / 'start_segmentation.py'
        result = subprocess.run([sys.executable, '-S', str(root), '--runtime', str(self.work.root), '--query', 'history'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([sys.executable, '-S', '-c',
                                'import sys; sys.path.insert(0,sys.argv[1]); from aaoca_seg.managed import ManagedController; ManagedController(sys.argv[2])',
                                 str(release), str(self.work.root)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def fake_processes(self, events):
        class Child:
            next_pid = 900000
            def __init__(child, args, **kwargs):
                Child.next_pid += 1
                child.pid = Child.next_pid
                child.code = None
                payload = read_json(args[-1])
                events.append(('start', payload['phase'], payload['case']['case_id'], kwargs['env']['CUDA_VISIBLE_DEVICES']))
                def work():
                    try:
                        time.sleep(.25)
                        if payload['phase'] == 'infer':
                            self.infer(payload['case'], Path(payload['attempt']))
                        elif payload['phase'] == 'merge':
                            executor.merge(payload['case'], payload['source'], payload['attempt'])
                        else:
                            executor.checkpoint(payload['source'], payload['case'])
                            done = executor.completed(Path(payload['attempt']) / 'result.json', payload['case'], payload['source'])
                            write_json(payload['verification'], {'valid': True, 'completed': done})
                        child.code = 0
                    except Exception:
                        child.code = 2
                    events.append(('end', payload['phase'], payload['case']['case_id']))
                child.thread = threading.Thread(target=work)
                child.thread.start()
            def poll(child):
                return child.code
            def wait(child):
                child.thread.join()
                return child.code
        return Child

    def test_parallel_gpu_queue_cold_cache_and_cpu_tail(self):
        settings = dict(self.work.settings, mode='gpu', devices=[{'index': 0, 'uuid': 'GPU-a'}, {'index': 1, 'uuid': 'GPU-b'}], gpu_count=2)
        self.work.save(settings)
        job = self.create()
        events = []
        query_calls = []
        def gpu_query(selected, current, minimum):
            query_calls.append(1)
            return [] if len(query_calls) == 1 else selected
        result = scheduler.run(job, allocation=self.alloc, probe=lambda: {'devices': settings['devices']},
                               gpu_query=gpu_query, popen=self.fake_processes(events), backend={'versions': {}})
        self.assertEqual(result['status'], 'completed', events)
        self.assertEqual(result['max_parallel'], 2)
        first_end = next(i for i,e in enumerate(events) if e[:2] == ('end', 'infer'))
        self.assertEqual(sum(e[:2] == ('start', 'infer') for e in events[:first_end]), 1)
        for event in events:
            if event[:2] == ('start', 'merge'):
                self.assertEqual(event[3], '')
                self.assertIn(('end', 'infer', event[2]), events[:events.index(event)])
        self.assertGreaterEqual(len(query_calls), 2)
        running, maximum, used = set(), 0, set()
        for event in events:
            if event[:2] == ('start', 'infer'):
                running.add(event[2]); used.add(event[3])
                maximum = max(maximum, len(running))
            elif event[:2] == ('end', 'infer'):
                running.discard(event[2])
        self.assertEqual(maximum, 2)
        self.assertEqual(used, {'GPU-a', 'GPU-b'})

    def test_waiting_gpu_can_be_stopped(self):
        self.work.save(dict(self.work.settings, mode='gpu', devices=[{'index': 0, 'uuid': 'GPU-a'}]))
        job = self.create()
        def unavailable(*_):
            (job / 'stop').touch()
            return []
        result = scheduler.run(job, allocation=self.alloc, probe=lambda: {}, gpu_query=unavailable,
                               backend={'versions': {}})
        self.assertEqual(result['status'], 'stopped')

    def test_real_detached_frozen_plan_and_resume(self):
        import site

        from aaoca_seg.operations import python_path
        from aaoca_seg.workbench import status
        envroot = self.work.root / 'venv'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(envroot)], check=True, capture_output=True)
        result = subprocess.run([str(python_path(self.work.root)), '-c', 'import site; print(site.getsitepackages()[-1])'],
                                check=True, capture_output=True, text=True)
        target = Path(result.stdout.strip())
        target.mkdir(parents=True, exist_ok=True)
        # Some research environments expose packages through their own .pth files;
        # nested .pth files are not processed when this child adds their directory.
        dependency_paths = [*site.getsitepackages(), str(Path(nib.__file__).parents[1]), str(Path(np.__file__).parents[1])]
        # Python 3.11 reads .pth using the Windows locale even in UTF-8 mode.
        (target / 'synthetic-test-deps.pth').write_text(
            'import sys; sys.path[:0] = ' + ascii(dependency_paths) + '\n', encoding='ascii')
        job = self.create(['plan'])
        request = (job / 'request.json').read_bytes()
        for _ in range(2):
            self.work.start(job)
            # A different console can query and resume the same durable job.
            restored = Workbench(self.work.root)
            deadline = time.monotonic() + 30
            while status(job)['active'] and time.monotonic() < deadline:
                time.sleep(.1)
            self.assertEqual(status(job)['status'], 'completed', restored.logs(job))
            self.assertFalse(status(job)['active'])
        self.assertEqual(request, (job / 'request.json').read_bytes())
        self.assertEqual(len(list((job / 'runs').glob('*.json'))), 2)

    def test_update_archive_rejects_links_and_traversal(self):
        import io
        import tarfile

        from aaoca_seg.updater import extract_release
        for name, kind in [('AAOCA/tools/coarse_segmentation/../../escape', tarfile.REGTYPE),
                           ('AAOCA/tools/coarse_segmentation/link', tarfile.SYMTYPE)]:
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as archive:
                item = tarfile.TarInfo(name); item.type = kind
                archive.addfile(item)
            with self.assertRaises(ValueError):
                extract_release(stream.getvalue(), self.base / 'extract')

    def test_update_fallback_creates_isolated_release_and_backup(self):
        import io
        import tarfile

        from aaoca_seg.updater import OFFICIAL, update
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as archive:
            payload = b'# synthetic release\n'
            item = tarfile.TarInfo('AAOCA/tools/coarse_segmentation/aaoca_seg/terminal.py')
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
        calls, fetches = [], []
        def git(root, *args, **kwargs):
            calls.append(args)
            if args == ('rev-parse', '--show-toplevel'):
                return str(self.base).encode()
            if args[:2] == ('config', '--get'):
                return OFFICIAL.encode()
            if args[0] == 'status' or args[0] == 'merge-base':
                return b''
            if args[:2] == ('rev-parse', 'HEAD'):
                return b'0' * 40
            if args[0] == 'ls-remote':
                return b'1' * 40 + b' refs/heads/main'
            if args[0] == 'fetch':
                fetches.append(args)
                if len(fetches) == 1:
                    raise subprocess.TimeoutExpired('git', 45)
                return b''
            if args[0] == 'rev-parse':
                return b'1' * 40
            if args[0] == 'archive':
                return data.getvalue()
            raise AssertionError(args)
        with patch('aaoca_seg.updater.git', side_effect=git):
            message = update(self.work)
        release = read_json(self.work.root / 'release.json')
        self.assertEqual(len(fetches), 2)
        self.assertTrue(Path(release['backup']).is_dir())
        self.assertTrue((Path(release['path']) / 'aaoca_seg/terminal.py').exists())
        self.assertFalse(any(c[0] in {'merge', 'checkout', 'reset', 'stash'} for c in calls))
        self.assertIn('重新打开', message)

    @unittest.skipIf(os.name == 'nt', 'POSIX owned process group')
    def test_stop_reaps_only_registered_orphan_model(self):
        from aaoca_seg.service_state import process_identity
        job = self.create()
        request = read_json(job / 'request.json')
        attempt = Path(request['output']) / 'cases/demo0/attempt-orphan'
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            write_json(attempt / 'model-process.json', process_identity(child.pid))
            write_json(attempt / 'executor-process.json', None)
            self.assertTrue(status(job)['active'])
            self.work.stop(job.name)
            child.wait(timeout=5)
            self.assertIsNone(unrelated.poll())
        finally:
            for proc in (child, unrelated):
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=5)

    @unittest.skipIf(os.name == 'nt', 'POSIX descriptor inheritance')
    def test_descendant_keeps_os_lock_after_owner_exit(self):
        lock = self.base / 'inherited.lock'
        ready = self.base / 'ready'
        stop = self.base / 'child-stop'
        code = "from aaoca_seg.service_state import file_lock; import subprocess,sys,os;\nwith file_lock(sys.argv[1]) as fd:\n subprocess.Popen([sys.executable,'-c',\"import pathlib,time,sys; pathlib.Path(sys.argv[1]).touch();\\nwhile not pathlib.Path(sys.argv[2]).exists(): time.sleep(.05)\",sys.argv[2],sys.argv[3]],pass_fds=(fd,),start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
        subprocess.run([sys.executable, '-c', code, str(lock), str(ready), str(stop)], cwd=CODE_ROOT, check=True)
        try:
            for _ in range(100):
                if ready.exists(): break
                time.sleep(.05)
            self.assertTrue(ready.exists())
            with self.assertRaises(ValueError):
                with file_lock(lock): pass
        finally:
            stop.touch()
            time.sleep(.2)
        with file_lock(lock): pass


if __name__ == '__main__':
    unittest.main()
