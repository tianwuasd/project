import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


class ScopeTests(unittest.TestCase):
    def test_direct_gpu_entry_requires_worker_isolation(self):
        from dataset1282_analysis.xu_gpu_isolation import validate_entry
        env = {}
        with self.assertRaises(ValueError):
            validate_entry(['--device','cuda:2'],env)
        validate_entry(['--device','cpu'],env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'-1')

    def test_guard_detects_owned_pid_outside_its_assigned_uuid(self):
        from dataset1282_analysis.xu_runtime_guard import isolation_violations
        import psutil
        pid = os.getpid()
        owned = {str(pid):dict(create_time=psutil.Process().create_time(),stage='crop64')}
        inventory = {'allowed':dict(compute_pids=[]),'forbidden':dict(compute_pids=[pid])}
        self.assertEqual(isolation_violations(inventory,owned,'crop64',{'allowed'}),{'forbidden':[pid]})
        self.assertEqual(isolation_violations(inventory,owned,'crop128',{'allowed'}),{})

    def test_reordered_group_remaps_tickets_and_arguments(self):
        from dataset1282_analysis.xu_gpu_isolation import launch_scope, report_for_parent
        cards = ['cuda:5','cuda:2']
        tickets = [dict(device=d,uuid=u,owned={},limit_bytes=123) for d,u in zip(cards,['GPU-b','GPU-a'])]
        overlay = dict(XU_CUDA_ADMISSION=json.dumps(tickets[0]),XU_CUDA_ADMISSIONS=json.dumps(tickets),
                       XU_BRANCH_DEVICES=json.dumps(cards))
        before = copy.deepcopy(overlay)
        argv, env, mapping = launch_scope(['--device','cuda:5'],overlay,cards,{'CUDA_VISIBLE_DEVICES':'0,1,2,5'})
        self.assertEqual(argv,['--device','cuda:0'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-b,GPU-a')
        self.assertEqual(json.loads(env['XU_BRANCH_DEVICES']),['cuda:0','cuda:1'])
        self.assertEqual(json.loads(env['XU_CUDA_ADMISSIONS'])[1]['device'],'cuda:1')
        self.assertEqual(mapping[0]['parent_device'],'cuda:5')
        self.assertEqual(overlay,before)
        report = dict(per_device={'cuda:0':{'peak':1},'cuda:1':{'peak':2}},devices=['cuda:0','cuda:1'])
        restored = report_for_parent(report,cards)
        self.assertEqual(list(restored['per_device']),cards)
        self.assertEqual(restored['devices'],cards)
        with self.assertRaises(ValueError):
            launch_scope(['--device','cuda:6'],overlay,cards,{})
        with self.assertRaises(ValueError):
            launch_scope(['--device','cuda:5'],{},cards,{})

    def test_cpu_clears_all_inherited_cuda_admissions(self):
        from dataset1282_analysis.xu_gpu_isolation import launch_scope
        _,env,mapping = launch_scope(['--device','cpu'],{},['cpu'],
                                    {'CUDA_VISIBLE_DEVICES':'0,1','XU_COORDINATOR_DEVICES':'[]',
                                     'XU_CUDA_ADMISSION':'stale','XU_BRANCH_DEVICES':'stale'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'-1')
        self.assertNotIn('XU_COORDINATOR_DEVICES',env)
        self.assertNotIn('XU_CUDA_ADMISSION',env)
        self.assertEqual(mapping,[])

    def test_coordinator_preserves_ordered_authorization_but_has_no_cuda(self):
        from dataset1282_analysis.xu_gpu_isolation import coordinator_environment
        inventory = {f'GPU-{i}':dict(uuid=f'GPU-{i}',physical_index=i,name='card',total_bytes=100)
                     for i in range(4)}
        env = coordinator_environment({'CUDA_VISIBLE_DEVICES':'GPU-3,GPU-1','NVIDIA_VISIBLE_DEVICES':'1,2,3'},inventory)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'-1')
        self.assertEqual([r['uuid'] for r in json.loads(env['XU_COORDINATOR_DEVICES'])],['GPU-3','GPU-1'])
        with self.assertRaises(ValueError):
            coordinator_environment({'CUDA_VISIBLE_DEVICES':'GPU-unknown'},inventory)
        with self.assertRaises(ValueError):
            coordinator_environment({'CUDA_VISIBLE_DEVICES':'0','NVIDIA_VISIBLE_DEVICES':'1'},inventory)

    def test_numeric_cuda_authorization_is_rejected_without_proven_ordinal_mapping(self):
        from dataset1282_analysis.xu_gpu_isolation import authorized_devices
        inventory = {'GPU-0':dict(uuid='GPU-0',physical_index=0,name='card',total_bytes=100)}
        for extra in ({},{'CUDA_DEVICE_ORDER':'PCI_BUS_ID'},{'NVIDIA_VISIBLE_DEVICES':'0'}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError,'UUID'):
                authorized_devices(dict(extra,CUDA_VISIBLE_DEVICES='0'),inventory)

    def test_pair_worker_uses_local_device_and_preserves_parent_identity(self):
        from dataset1282_analysis.xu_pair_probe import worker_card
        from dataset1282_analysis.xu_gpu_isolation import launch_scope, report_for_parent
        for device,uuid in [('cuda:2','GPU-a'),('cuda:5','GPU-b')]:
            parent = dict(device=device,uuid=uuid,name='card',total_bytes=100)
            ticket = dict(device=device,uuid=uuid,policy='v2')
            _,env,_ = launch_scope([],{'XU_CUDA_ADMISSION':json.dumps(ticket)},[device],{})
            local = worker_card(parent,env)
            self.assertEqual(local,dict(parent,device='cuda:0'))
            self.assertEqual(report_for_parent(local,[device]),parent)
            with self.assertRaises(ValueError):
                worker_card(dict(parent,uuid='GPU-other'),env)

    def test_gpu_acceptance_and_pair_worker_entry_reject_unscoped_launch(self):
        root = Path(__file__).resolve().parents[1]
        env = {k:v for k,v in os.environ.items() if not k.startswith('XU_')}
        env['CUDA_VISIBLE_DEVICES'] = '-1'
        for script,args in [('xu_accept.py',['--device','cuda:0']),('xu_pair_probe.py',['_worker'])]:
            with self.subTest(script=script):
                row = subprocess.run([sys.executable,'-B',str(root/'scripts'/script),*args],
                                     env=env,capture_output=True,text=True,encoding='utf-8',timeout=20)
                self.assertNotEqual(row.returncode,0)
                self.assertIn('GPU 计算需要隔离启动',row.stderr)


class LiveIsolationTests(unittest.TestCase):
    def test_worker_readmission_rejects_real_occupied_card_before_cuda_query(self):
        from dataset1282_analysis import xu_gpu_isolation
        self.assertTrue(callable(getattr(xu_gpu_isolation,'admit_worker_scope',None)))
        from dataset1282_analysis.xu_gpu_probe import gpu_snapshot
        from dataset1282_analysis.xu_resources import ResourceUnavailable
        inventory = gpu_snapshot()
        occupied = [r for r in inventory.values() if r['compute_pids']]
        if not occupied:
            self.skipTest('当前没有外部计算进程')
        row = occupied[0]
        tickets = [dict(device='cuda:0',uuid=row['uuid'],policy='v2',owned={},require_pid_evidence=True)]
        with self.assertRaisesRegex(ResourceUnavailable,'External compute'):
            xu_gpu_isolation.admit_worker_scope(tickets,{},inventory)

    def test_real_coordinator_has_no_visible_cuda_or_compute_pid(self):
        from dataset1282_analysis.xu_gpu_probe import gpu_snapshot
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            child = subprocess.Popen([sys.executable,'-B',str(root/'tests/fixtures/xu_isolation_child.py'),
                                      str(out/'ready.json'),str(out/'release'),'coordinator'],cwd=root)
            try:
                deadline = time.monotonic()+25
                while not (out/'ready.json').exists() and child.poll() is None and time.monotonic()<deadline:
                    time.sleep(.03)
                self.assertTrue((out/'ready.json').exists())
                row = json.loads((out/'ready.json').read_text())
                self.assertEqual(row['count'],0)
                self.assertFalse(row['available'])
                self.assertFalse(any(row['pid'] in r['compute_pids'] for r in gpu_snapshot().values()))
            finally:
                (out/'release').touch()
                child.wait(timeout=10)

    def test_real_cpu_worker_hides_gpu_before_torch_import(self):
        from dataset1282_analysis.xu_pipeline import _execute
        from dataset1282_analysis.xu_dispatch import Children
        from dataset1282_analysis.observability.status import operation_lock
        from dataset1282_analysis.xu_gpu_probe import gpu_snapshot
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with operation_lock(out,'isolation-test',out) as owner:
                children = Children(out,owner)
                args = SimpleNamespace(threads=2,children=children,overlay={},device='cpu',assigned_devices=['cpu'])
                with ThreadPoolExecutor(max_workers=1) as pool:
                    job = pool.submit(_execute,'../tests/fixtures/xu_isolation_child.py',
                                      [out/'ready.json',out/'release'],'isolated',out,owner,args)
                    try:
                        deadline = time.monotonic()+25
                        while not (out/'ready.json').exists() and not job.done() and time.monotonic()<deadline:
                            time.sleep(.03)
                        if job.done():
                            job.result()
                        self.assertTrue((out/'ready.json').exists())
                        row = json.loads((out/'ready.json').read_text())
                        self.assertEqual(row['visible'],'-1')
                        self.assertEqual(row['count'],0)
                        self.assertFalse(row['available'])
                        self.assertFalse(row['initialized'])
                        self.assertFalse(any(row['pid'] in card['compute_pids'] for card in gpu_snapshot().values()))
                    finally:
                        (out/'release').touch()
                    job.result(timeout=10)


if __name__ == '__main__':
    unittest.main()
