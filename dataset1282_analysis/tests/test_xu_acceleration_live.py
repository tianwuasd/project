import copy
import importlib.util
from contextlib import contextmanager
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.ndimage import zoom
import torch

from dataset1282_analysis.xu_supply import BatchSupply
from dataset1282_analysis.xu_training import _epoch
from dataset1282_analysis.contracts import sha256_file


@contextmanager
def environment(**values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class SupplyLiveTests(unittest.TestCase):
    def test_multiple_workers_keep_original_order_and_actual_values(self):
        threads = set()
        lock = threading.Lock()
        source = np.arange(32**3, dtype='float32').reshape((32,)*3)
        def prepare(batch):
            with lock:
                threads.add(threading.current_thread().name)
            return torch.from_numpy(zoom(source + batch[0], .5, order=1, prefilter=False))
        with environment(XU_PREFETCH=1, XU_PREFETCH_BYTES=1024**3, XU_PREFETCH_WORKERS=2):
            supply = BatchSupply(list(range(12)), 1, 16, prepare, lambda _:32**3)
            with supply.batches() as stream:
                rows = [(names, values.clone()) for names, values in stream]
        self.assertEqual([names[0] for names, _ in rows], list(range(12)))
        for names, value in rows:
            np.testing.assert_array_equal(value.numpy(), zoom(source+names[0], .5, order=1, prefilter=False))
        self.assertGreaterEqual(len([name for name in threads if name.startswith('xu-cpu')]), 2)
        self.assertGreaterEqual(supply.stats.get('workers_applied', 0), 2)
        self.assertFalse(any(t.name.startswith('xu-cpu') for t in threading.enumerate()))

    def test_parallel_supply_keeps_loss_weights_optimizer_and_rng(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        raw = np.arange(4**3).reshape((4,)*3)
        names = ['a','b','c','d','e']
        def load(name, branch):
            return (raw+names.index(name)).astype('float32')/32, (raw%11).astype('int8')
        load.source_voxels = lambda _:64
        net = torch.nn.Conv3d(1,11,1)
        net.spec = SimpleNamespace(side=4, batch=2)
        results = []
        for workers in (0,2):
            model = copy.deepcopy(net)
            optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
            rng = torch.Generator().manual_seed(31)
            stats = {}
            with environment(XU_PREFETCH=int(workers>0), XU_PREFETCH_WORKERS=max(1,workers),
                             XU_PREFETCH_BYTES=1024**3):
                loss = _epoch(model,names,{n:[] for n in names},load,'crop64','cpu',rng,optimizer,stats)
            results.append((loss,copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict()),rng.get_state()))
        self.assertEqual(results[0][0],results[1][0])
        for key in results[0][1]:
            torch.testing.assert_close(results[0][1][key],results[1][1][key],rtol=0,atol=0)
        for index, state in results[0][2]['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value,results[1][2]['state'][index][key],rtol=0,atol=0)
        self.assertTrue(torch.equal(results[0][3],results[1][3]))

    def test_invalid_memory_budget_is_reported(self):
        with environment(XU_PREFETCH=1, XU_PREFETCH_BYTES='invalid'):
            with self.assertRaises(ValueError):
                BatchSupply([0],1,4,lambda batch:torch.ones(1),lambda _:64)


class ArrayCacheLiveTests(unittest.TestCase):
    def test_actual_numpy_cache_reuses_exact_values_and_detects_corruption(self):
        self.assertIsNotNone(importlib.util.find_spec('dataset1282_analysis.xu_sample_cache'))
        from dataset1282_analysis.xu_sample_cache import ArrayCache
        root = Path(__file__).resolve().parents[1]/'local-validation'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            base = Path(directory)
            source = base/'source.npy'
            np.save(source,np.arange(32**3,dtype='float32').reshape((32,)*3),allow_pickle=False)
            identity = dict(source=sha256_file(source),shape=[16]*3,order=1)
            def build():
                return zoom(np.load(source,allow_pickle=False),.5,order=1,prefilter=False)
            cache = ArrayCache(base/'derived')
            first = cache.get(identity,build)
            second = cache.get(identity,build)
            np.testing.assert_array_equal(first,second)
            self.assertEqual(cache.stats['hits'],1)
            self.assertEqual(cache.stats['created'],1)
            entry = next((base/'derived').glob('*.npy'))
            values = np.load(entry,allow_pickle=False)
            values.flat[0] += 1
            np.save(entry,values,allow_pickle=False)
            with self.assertRaisesRegex(ValueError,'缓存'):
                cache.get(identity,build)


class GpuQueryLiveTests(unittest.TestCase):
    def test_query_preserves_absent_cuda_context_and_reports_real_uuids(self):
        self.assertIsNotNone(importlib.util.find_spec('dataset1282_analysis.xu_gpu_probe'))
        from cuda.bindings import driver
        from dataset1282_analysis.xu_gpu_probe import visible_devices, gpu_snapshot
        devices = visible_devices()
        rows = gpu_snapshot()
        self.assertTrue(devices)
        self.assertTrue(all(row['uuid'] in rows for row in devices))
        self.assertEqual(int(driver.cuCtxGetCurrent()[1]),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_external_gpu_process_is_rejected_without_context(self):
        from cuda.bindings import driver
        from dataset1282_analysis.xu_gpu_probe import visible_devices, gpu_snapshot
        from dataset1282_analysis.xu_resources import _check_device, ResourceUnavailable
        rows = gpu_snapshot()
        occupied = [r for r in visible_devices() if rows[r['uuid']]['compute_pids']]
        if not occupied:
            self.skipTest('当前可见 GPU 没有外部计算进程')
        info = {**rows[occupied[0]['uuid']],**occupied[0]}
        with self.assertRaisesRegex(ResourceUnavailable,'External compute'):
            _check_device(info,'v2',{},True)
        self.assertEqual(int(driver.cuCtxGetCurrent()[1]),0)


class RuntimeGuardLiveTests(unittest.TestCase):
    def test_cpu_allocation_requires_slots_for_consumer_and_workers(self):
        from dataset1282_analysis.xu_dispatch import prefetch_cpu_allocation
        from dataset1282_analysis.xu_resources import ResourceUnavailable
        for slots in range(2,33):
            for cap in (1,2,4,16,32):
                threads,workers = prefetch_cpu_allocation(slots,cap,4)
                self.assertLessEqual((workers+1)*threads,slots)
                self.assertLessEqual(threads,cap)
                self.assertGreaterEqual(workers,1)
        with self.assertRaises(ResourceUnavailable):
            prefetch_cpu_allocation(1,1,4)

    def test_actual_stage_limits_consumer_and_prefetch_together(self):
        from dataset1282_analysis.xu_dispatch import Children, plan_stage
        from dataset1282_analysis.xu_pipeline_stages import Stage
        from dataset1282_analysis.observability.status import operation_lock
        from dataset1282_analysis.contracts import read_json
        root = Path(__file__).resolve().parents[1]/'local-validation'
        with tempfile.TemporaryDirectory(dir=root) as directory:
            out = Path(directory)
            args = SimpleNamespace(resource_policy='legacy',profile='smoke',wait_seconds=0,
                                   threads=None,prefetch=True,prefetch_mib=4096,prefetch_workers=4)
            with operation_lock(out,'cpu-budget',out) as owner:
                children = Children(out,owner)
                stage = Stage('crop64',('train','--device','cpu'),('best.pt',))
                _,overlay = plan_stage(stage,args,out,children,1,'launch',{})
            record = read_json(next((out/'resources').glob('crop64-launch-*.json')))
            threads = int(overlay['OMP_NUM_THREADS'])
            workers = int(overlay['XU_PREFETCH_WORKERS'])
            self.assertLessEqual((workers+1)*threads,record['plan']['cpu_slots'])

    def test_actual_child_and_memory_reserve(self):
        self.assertIsNotNone(importlib.util.find_spec('dataset1282_analysis.xu_runtime_guard'))
        from dataset1282_analysis.xu_runtime_guard import RuntimeGuard
        from dataset1282_analysis.xu_resources import ResourceUnavailable
        from dataset1282_analysis.xu_dispatch import Children
        from dataset1282_analysis.observability.status import operation_lock
        root = Path(__file__).resolve().parents[1]/'local-validation'
        with tempfile.TemporaryDirectory(dir=root) as directory:
            out = Path(directory)
            with operation_lock(out,'guard-test',out) as owner:
                children = Children(out,owner)
                process = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
                try:
                    children.attach(process,'actual-child','cpu',out/'child.log')
                    guard = RuntimeGuard(out/'resources','actual-child',{},children)
                    evidence = guard.check()
                    self.assertEqual(evidence['state'],'running')
                    self.assertIn(str(process.pid),evidence['owned'])
                    guard.reserve_bytes = evidence['available_memory_bytes']*100
                    with self.assertRaisesRegex(ResourceUnavailable,'内存'):
                        guard.check()
                    self.assertIsNone(process.poll())
                finally:
                    children.stop()
                    children.detach(process)
                self.assertIsNotNone(process.poll())


class CheckpointLiveTests(unittest.TestCase):
    def test_actual_network_failure_and_resume_keep_weights_and_adam(self):
        from dataset1282_analysis.xu_training import train_branch
        root = Path(__file__).resolve().parents[1]/'local-validation'
        old = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,old)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            out = Path(directory)
            grid = np.arange(64**3).reshape((64,)*3)
            np.save(out/'image.npy',(grid/(64**3)).astype('float32'))
            np.save(out/'label.npy',(grid%11).astype('int8'))
            def load(name,branch):
                return np.load(out/'image.npy'),np.load(out/'label.npy')
            load.source_voxels = lambda _:64**3
            calls = []
            def interrupted(name,branch):
                calls.append(name)
                if len(calls) == 3:
                    return np.load(out/'missing.npy'),np.load(out/'label.npy')
                return load(name,branch)
            interrupted.source_voxels = load.source_voxels
            config = dict(branch='crop64',train_cases={'train':[]},validation_cases={'dev':[]},
                          data_identity='actual-resume-input',profile='smoke',pilot_epochs=2)
            with environment(XU_PREFETCH=1,XU_PREFETCH_WORKERS=2,XU_PREFETCH_BYTES=1024**3):
                train_branch(**config,output_dir=out/'full',load_sample=load)
                with self.assertRaises(FileNotFoundError):
                    train_branch(**config,output_dir=out/'resumed',load_sample=interrupted)
                saved = torch.load(out/'resumed/last.pt',map_location='cpu',weights_only=True)
                self.assertEqual(saved['epoch'],0)
                train_branch(**config,output_dir=out/'resumed',load_sample=load,resume=True)
            full = torch.load(out/'full/last.pt',map_location='cpu',weights_only=True)
            resumed = torch.load(out/'resumed/last.pt',map_location='cpu',weights_only=True)
            for key,value in full['model'].items():
                torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
            for index,state in full['optimizer']['state'].items():
                for key,value in state.items():
                    torch.testing.assert_close(value,resumed['optimizer']['state'][index][key],rtol=0,atol=0)
            self.assertEqual(full['best_loss'],resumed['best_loss'])


if __name__ == '__main__':
    unittest.main()
