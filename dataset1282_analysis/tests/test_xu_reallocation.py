import os
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from dataset1282_analysis import xu_gpu_probe
from dataset1282_analysis.xu_dispatch import Children
from dataset1282_analysis.xu_pipeline import _stage_spec
from dataset1282_analysis.xu_pipeline_stages import Stage
from dataset1282_analysis.observability.status import operation_lock
from dataset1282_analysis.contracts import read_json


class DeviceSelectionTests(unittest.TestCase):
    def test_physical_exclusion_survives_visible_index_reordering(self):
        self.assertTrue(callable(getattr(xu_gpu_probe, 'filter_physical_devices', None)))
        devices = [dict(device='cuda:0', uuid='gpu-3'), dict(device='cuda:1', uuid='gpu-0')]
        inventory = {'gpu-3': {'physical_index': 3}, 'gpu-0': {'physical_index': 0}}
        result = xu_gpu_probe.filter_physical_devices(devices, (0, 1), inventory)
        self.assertEqual(result, devices[:1])
        self.assertEqual(xu_gpu_probe.filter_physical_devices(devices, (), inventory), devices)
        with self.assertRaises(ValueError):
            xu_gpu_probe.filter_physical_devices(devices, (0, 1), {})


class SchedulingIdentityTests(unittest.TestCase):
    def test_reallocated_fusion_archives_partial_output_and_training_resumes(self):
        from dataset1282_analysis.xu_pipeline import _restore_attempt
        from dataset1282_analysis.contracts import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            fusion = Stage('four-way', ('fuse',), ('complete.json',), resume_marker='job.json')
            atomic_json(out/'four-way/job.json', {'inference_shards': ['old-device']})
            self.assertEqual(_restore_attempt(fusion,out/'four-way',out,reallocated=True),[])
            self.assertFalse((out/'four-way').exists())
            archived = list((out/'attempts').glob('four-way-*/job.json'))
            self.assertEqual(len(archived),1)
            self.assertEqual(read_json(archived[0]),{'inference_shards':['old-device']})
            training = Stage('crop64', ('train',), ('last.pt',), resume_marker='last.pt')
            (out/'crop64').mkdir()
            (out/'crop64/last.pt').write_bytes(b'checkpoint evidence')
            self.assertEqual(_restore_attempt(training,out/'crop64',out,reallocated=True),['--resume'])

    def test_stage_identity_preserves_research_across_device_changes(self):
        a = Stage('crop64', ('train', '--device', 'cuda:2', '--seed', '2023'), ('last.pt',))
        b = Stage('crop64', ('train', '--device', 'cuda:5', '--seed', '2023'), ('last.pt',))
        self.assertEqual(_stage_spec(a, 'research', ['cuda:2', 'cuda:3']),
                         _stage_spec(b, 'research', ['cuda:5']))
        self.assertNotEqual(_stage_spec(a, 'research'), _stage_spec(a, 'different-data'))

    def test_launch_registration_gate_is_available(self):
        self.assertTrue(callable(getattr(Children, 'configure_launch', None)))


class LaunchTests(unittest.TestCase):
    def test_cancelled_group_refuses_atomic_spawn(self):
        self.assertTrue(callable(getattr(Children, 'spawn', None)))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with operation_lock(out,'spawn-test',out) as owner:
                children = Children(out,owner)
                children.stop()
                with self.assertRaises(RuntimeError):
                    children.spawn([sys.executable,'-c','raise RuntimeError("must not run")'],
                                   'blocked','cpu',out/'blocked.log')
                self.assertFalse(children.active)

    def test_real_children_wait_until_both_are_registered(self):
        from dataset1282_analysis.xu_pipeline import _execute
        self.assertTrue(callable(getattr(Children, 'configure_launch', None)))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with operation_lock(out, 'launch-test', out) as owner:
                children = Children(out, owner)
                children.configure_launch(2)
                args = SimpleNamespace(threads=2, children=children, overlay={}, device='cpu',
                                       assigned_devices=['cpu'])
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(_execute, '../tests/fixtures/xu_launch_child.py',
                                        [str(out/'first.json'),'wait-peer',str(out/'second.json')], 'first', out, owner, args)
                    time.sleep(.25)
                    self.assertFalse((out/'first.json').exists())
                    second = pool.submit(_execute, '../tests/fixtures/xu_launch_child.py',
                                         [str(out/'second.json'),'wait-peer',str(out/'first.json')], 'second', out, owner, args)
                    first.result(timeout=30)
                    second.result(timeout=30)
                a, b = read_json(out/'first.json'), read_json(out/'second.json')
                self.assertEqual(a['parents'], b['parents'])
                self.assertGreaterEqual(len(a['parents']), 3)
                self.assertEqual(a['verified'],a['parents'])
                self.assertEqual(b['verified'],b['parents'])
                self.assertFalse(children.active)


class ResourceChangeTests(unittest.TestCase):
    def test_actual_worker_event_retries_and_data_error_stops(self):
        from dataset1282_analysis import xu_reallocation
        from dataset1282_analysis.xu_pipeline import _execute, StageError
        self.assertTrue(callable(getattr(xu_reallocation, 'run_with_reallocation', None)))
        for event in ('resource-change', 'data-error'):
            with self.subTest(event=event), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                calls = []
                args = SimpleNamespace(resume=False,max_reallocations=3)
                with operation_lock(out, 'retry-test', out) as owner:
                    def actual_stage(current):
                        calls.append(current.resume)
                        children = Children(out, owner)
                        worker_args = SimpleNamespace(threads=2,children=children,overlay={},device='cpu',
                                                      assigned_devices=['cpu'])
                        argv = [str(out/f'payload-{len(calls)}.json')]
                        if not current.resume:
                            argv.append(event)
                        try:
                            _execute('../tests/fixtures/xu_launch_child.py',argv,'worker',out,owner,worker_args)
                        finally:
                            children.stop()
                    if event == 'resource-change':
                        xu_reallocation.run_with_reallocation(actual_stage,args,out)
                        self.assertEqual(calls,[False,True])
                        self.assertTrue((out/'payload-2.json').exists())
                        self.assertEqual(len(list((out/'reallocations').glob('*.json'))),1)
                    else:
                        with self.assertRaises(StageError):
                            xu_reallocation.run_with_reallocation(actual_stage,args,out)
                        self.assertEqual(calls,[False])
                        self.assertFalse((out/'reallocations').exists())

    def test_gpu_pid_classification_uses_actual_process_birth(self):
        from dataset1282_analysis import xu_runtime_guard
        import psutil
        self.assertTrue(callable(getattr(xu_runtime_guard, 'external_processes', None)))
        current = os.getpid()
        birth = psutil.Process(current).create_time()
        self.assertEqual(xu_runtime_guard.external_processes([current], {}), [current])
        self.assertEqual(xu_runtime_guard.external_processes([current], {str(current): birth}), [])
        self.assertEqual(xu_runtime_guard.external_processes([current], {str(current): birth-1}), [current])

    def test_execution_contract_allows_new_counts_but_keeps_fixed_batch(self):
        from dataset1282_analysis import xu_training
        self.assertTrue(callable(getattr(xu_training, 'execution_contract', None)))
        self.assertEqual(xu_training.execution_contract(1, 4, True),
                         xu_training.execution_contract(3, 4, True))
        self.assertNotEqual(xu_training.execution_contract(1, 4, True),
                            xu_training.execution_contract(1, 2, True))
        self.assertEqual(xu_training.execution_contract(1, 4, False), {})
        self.assertEqual(xu_training.execution_contract(3, 4, False), {'batch_replicas': 3})

    def test_reallocation_is_bounded_and_keeps_completed_artifacts(self):
        from dataset1282_analysis import xu_pipeline
        self.assertTrue(callable(getattr(xu_pipeline, 'record_reallocation', None)))
        from dataset1282_analysis.xu_resources import GpuResourcesChanged
        from dataset1282_analysis.contracts import atomic_json, digest, sha256_file
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            assignment = dict(stage='crop64', devices=['cuda:2'], signature=digest(['id','crop64',['cuda:2']]))
            atomic_json(out/'assignments/crop64.json', assignment)
            atomic_json(out/'receipts/crop64.json', {'state':'complete'})
            before = sha256_file(out/'receipts/crop64.json')
            error = GpuResourcesChanged('actual occupancy changed', ['cuda:3'])
            for attempt in range(1,4):
                xu_pipeline.record_reallocation(out, error, attempt, 3)
            self.assertEqual(sha256_file(out/'receipts/crop64.json'), before)
            self.assertEqual(read_json(out/'assignments/crop64.json'), assignment)
            self.assertEqual(len(list((out/'reallocations').glob('*.json'))), 3)
            with self.assertRaises(GpuResourcesChanged):
                xu_pipeline.record_reallocation(out, error, 4, 3)

    def test_cancel_wakes_waiting_child_and_preserves_unowned_process(self):
        from dataset1282_analysis.xu_pipeline import _execute
        self.assertTrue(callable(getattr(Children, 'configure_launch', None)))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            self.addCleanup(foreign.wait)
            self.addCleanup(foreign.terminate)
            with operation_lock(out, 'cancel-test', out) as owner:
                children = Children(out, owner)
                children.configure_launch(2)
                args = SimpleNamespace(threads=2, children=children, overlay={}, device='cpu',
                                       assigned_devices=['cpu'])
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_execute, '../tests/fixtures/xu_launch_child.py',
                                         [str(out/'unexpected.json')], 'blocked', out, owner, args)
                    deadline = time.monotonic()+10
                    while len(children.pids()) < 2 and time.monotonic() < deadline:
                        time.sleep(.02)
                    children.stop()
                    with self.assertRaises((RuntimeError, threading.BrokenBarrierError)):
                        future.result(timeout=15)
                self.assertFalse((out/'unexpected.json').exists())
                self.assertIsNone(foreign.poll())
                self.assertFalse(children.active)
