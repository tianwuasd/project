import os
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from dataset1282_analysis import xu_resources as r
from dataset1282_analysis.resources import GIB


class XuResourceTests(unittest.TestCase):
    def test_combined_tickets_validate_every_actual_device_and_restore_primary(self):
        cards = {f'cuda:{i}': dict(self.card,device=f'cuda:{i}', uuid=f'GPU-{i}') for i in range(2)}
        names = ['cuda:1','cuda:0']
        with patch.object(r,'_device_info',side_effect=lambda d:cards[str(d)]), patch.object(
                r,'_compute_pids',return_value=set()):
            plan = r.stage_resources(names)
            overlay = r.combined_environment(plan,names)
            with patch.dict(os.environ,overlay,clear=True), patch('torch.cuda.set_device') as set_card, patch(
                    'torch.cuda.set_per_process_memory_fraction') as limit:
                result = r.apply_env_limit(names[0])
                self.assertEqual(list(result['per_device']),names)
                self.assertEqual([str(c.args[0]) for c in set_card.call_args_list],names+[names[0]])
                self.assertEqual([str(c.args[1]) for c in limit.call_args_list],names)
                r.require_applied_group(names)
                with self.assertRaises(r.ResourceUnavailable):
                    r.require_applied_group(list(reversed(names)))
                with patch.object(r.os,'getpid',return_value=999_999_999):
                    with self.assertRaises(r.ResourceUnavailable):
                        r.require_applied_group(names)
                os.environ['XU_BRANCH_DEVICES'] = json.dumps(list(reversed(names)))
                with self.assertRaisesRegex(r.ResourceUnavailable,'do not match'):
                    r.apply_env_limit(names[0])
            with patch.dict(os.environ,{'XU_BRANCH_DEVICES':json.dumps(names)},clear=True):
                with self.assertRaises(r.ResourceUnavailable):
                    r.apply_env_limit(names[0])

    def setUp(self):
        self.live = dict(available_memory_bytes=40 * GIB, cpu_available=12.5,
                         cpu_permitted=16, missing_evidence=[], affinity_source='sched_getaffinity',
                         cpu_availability_source='permitted_cpu_idle_sample', cgroup_evidence=[{'version':2}])
        self.host = patch.object(r, 'snapshot', return_value=self.live)
        self.host.start()
        self.addCleanup(self.host.stop)
        self.card = dict(device='cuda:1', uuid='GPU-12345678-aaaa-bbbb-cccc-123456789abc',
                         name='allocated device', free_bytes=28*GIB, total_bytes=40*GIB)

    def test_host_shares_limit_requested_threads_actual_memory_and_recheck(self):
        plan = r.stage_resources(['cpu'], concurrent=2, threads=8, host_demand_bytes=15*GIB)
        self.assertEqual(plan['threads'], 5)
        self.assertEqual(plan['host_per_stage_bytes'], 16*GIB)
        with patch.dict(os.environ, r.limit_environment(plan,'cpu'), clear=True):
            self.assertEqual(r.apply_env_limit('cpu')['host_budget_bytes'],16*GIB)
            self.live['available_memory_bytes'] = 16*GIB
            with self.assertRaisesRegex(r.ResourceUnavailable, 'fresh host RAM'):
                r.apply_env_limit('cpu')
        with self.assertRaises(r.ResourceUnavailable):
            r.stage_resources(['cpu'], concurrent=2, host_demand_bytes=15*GIB)
        self.live['cpu_available'] = None
        with self.assertRaises(r.ResourceUnavailable):
            r.stage_resources(['cpu'])

    def test_uuid_targeted_query_rejects_unmapped_members_and_missing_evidence(self):
        with patch.object(r.subprocess, 'run') as run:
            run.return_value.stdout = self.card['uuid']+', 1777\n'
            self.assertEqual(r._compute_pids(self.card['uuid']), {1777})
            self.assertIn('--id='+self.card['uuid'], run.call_args.args[0])
            run.return_value.stdout = 'GPU-other, 1777\n'
            self.assertIsNone(r._compute_pids(self.card['uuid']))
            run.side_effect = subprocess.TimeoutExpired('nvidia-smi', 5)
            self.assertIsNone(r._compute_pids(self.card['uuid']))
        with patch.object(r,'_device_info',return_value=self.card), patch.object(
                r,'_compute_pids',return_value=None):
            with self.assertRaisesRegex(r.ResourceUnavailable, 'PID evidence'):
                r.stage_resources(['cuda:1'],require_pid_evidence=True)
        with patch.object(r,'_device_info',return_value=self.card), patch.object(
                r,'_compute_pids',return_value={999_999_999}):
            with self.assertRaisesRegex(r.ResourceUnavailable,'External compute processes'):
                r.stage_resources(['cuda:1'],require_pid_evidence=True)

    def test_allocator_rechecked_and_limited_in_real_worker(self):
        pid = os.getpid()
        with patch.object(r,'_device_info',side_effect=lambda device: dict(self.card)), patch.object(
                r,'_compute_pids',return_value={pid}):
            plan = r.stage_resources(['cuda:1'], require_pid_evidence=True)
            limits = r.limit_environment(plan, 'cuda:1')
            self.assertEqual(plan['devices']['cuda:1']['allocator_budget_bytes'],24*GIB)
            with patch.dict(os.environ,limits,clear=True), patch('torch.cuda.set_device'), patch(
                    'torch.cuda.set_per_process_memory_fraction') as set_ceiling:
                self.card['free_bytes'] = 12*GIB
                applied = r.apply_env_limit('cuda:1')
                self.assertEqual(applied['allocator_budget_bytes'], 8*GIB)
                set_ceiling.assert_called_once_with(.2,'cuda:1')
                self.card['uuid'] = 'other-card'
                with self.assertRaisesRegex(r.ResourceUnavailable, 'identity changed'):
                    r.apply_env_limit('cuda:1')

    def test_fractional_reserve_and_duplicate_uuids(self):
        self.assertEqual(r.gpu_budget(40*GIB,80*GIB,'v2'),32*GIB)
        self.assertEqual(r.gpu_budget(4*GIB,24*GIB,'v2'),0)
        self.assertEqual(r.gpu_budget(10*GIB,24*GIB,'legacy'),8*GIB)
        with patch.object(r,'_device_info',side_effect=lambda d:dict(self.card,device=d)), patch.object(
                r,'_compute_pids',return_value=set()):
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                r.stage_resources(['cuda:1','cuda:0'],concurrent=2)

    def test_minimum_cuda_demand_blocks_before_or_between_parent_and_child(self):
        with patch.object(r,'_device_info',side_effect=lambda _:dict(self.card)), patch.object(
                r,'_compute_pids',return_value=set()):
            with self.assertRaisesRegex(r.ResourceUnavailable,'needs .* CUDA bytes'):
                r.stage_resources(['cuda:1'],cuda_demand_bytes={'cuda:1':25*GIB})
            plan = r.stage_resources(['cuda:1'],cuda_demand_bytes=22*GIB)
            self.card['free_bytes'] = 25*GIB
            with patch.dict(os.environ,r.limit_environment(plan,'cuda:1'),clear=True), patch(
                    'torch.cuda.set_device'), patch(
                    'torch.cuda.set_per_process_memory_fraction') as apply:
                with self.assertRaisesRegex(r.ResourceUnavailable,'exceeds fresh budget'):
                    r.apply_env_limit('cuda:1')
                apply.assert_not_called()

    def test_worker_selects_ticket_ordinal_before_first_cuda_query(self):
        events = []
        def inspect(device):
            events.append(('query',str(device)))
            return self.card
        with patch.object(r,'_device_info',side_effect=inspect), patch.object(r,'_compute_pids',return_value=set()):
            plan = r.stage_resources(['cuda:1'])
            events.clear()
            with patch.dict(os.environ,r.limit_environment(plan,'cuda:1'),clear=True), patch(
                    'torch.cuda.set_device',side_effect=lambda d:events.append(('set',str(d)))), patch(
                    'torch.cuda.set_per_process_memory_fraction'):
                for bad in ('cuda:0','cuda','cpu'):
                    with self.assertRaisesRegex(r.ResourceUnavailable,'explicit assigned'):
                        r.apply_env_limit(bad)
                self.assertEqual(events,[])
                r.apply_env_limit('cuda:1')
                self.assertEqual(events,[('set','cuda:1'),('query','cuda:1')])

    def test_linux_v2_rejects_missing_cgroup_but_accepts_absent_numa(self):
        with patch.object(r.sys,'platform','linux'):
            self.live['missing_evidence'] = ['NUMA mapping unavailable']
            self.assertGreater(r.stage_resources(['cpu'])['cpu_slots'],0)
            self.live['missing_evidence'] = ['quota CPU usage delta unavailable']
            with self.assertRaisesRegex(r.ResourceUnavailable,'quota'):
                r.stage_resources(['cpu'])
            self.live['missing_evidence'] = []
            self.live['cgroup_evidence'] = []
            with self.assertRaisesRegex(r.ResourceUnavailable,'cgroup'):
                r.stage_resources(['cpu'])

    def test_child_only_accepts_live_owner_birth_id(self):
        active = {'41': 100.0}
        with patch.object(r,'_compute_pids',return_value={41}), patch.object(
                r,'_identities',return_value=active):
            with self.assertRaisesRegex(r.ResourceUnavailable, 'External compute processes'):
                r._check_device(self.card,'v2',{'41':99.0},True)
            self.assertEqual(r._check_device(self.card,'v2',active,True)['compute_pids'],[41])

    def test_wave_reclaims_realized_share_without_dividing_busy_cpu_again(self):
        w = r.reserve_wave(8,'v2')
        self.live.update(available_memory_bytes=28*GIB,cpu_available=2.)
        # Of the original 16 GiB peer reserve, 12 are already resident.
        with patch.object(r,'_peer_credit',return_value=12*GIB):
            plan = r.stage_resources(['cpu'],concurrent=2,wave_reservation=w,
                                     peer_active_pids={41:1.},host_demand_bytes=15*GIB)
            self.assertEqual(plan['threads'],1)
            self.assertEqual(plan['host_per_stage_bytes'],16*GIB)
            self.assertEqual(plan['wave']['peer_outstanding_bytes'],4*GIB)
            self.assertEqual(plan['wave']['peer_credited_bytes'],12*GIB)
            self.live['available_memory_bytes'] = 13*GIB
            with self.assertRaises(r.ResourceUnavailable):
                r.stage_resources(['cpu'],concurrent=2,wave_reservation=w,host_demand_bytes=15*GIB)
        self.live['available_memory_bytes'] = 28*GIB
        with self.assertRaises(r.ResourceUnavailable):
            r.stage_resources(['cpu'],concurrent=2,wave_reservation=w,
                              peer_active_pids=[41],host_demand_bytes=15*GIB)
        done = r.stage_resources(['cpu'],concurrent=2,wave_reservation=w,
                                 peer_finished=True,host_demand_bytes=15*GIB)
        self.assertEqual(done['wave']['peer_outstanding_bytes'],0)
        self.assertEqual(done['host_per_stage_bytes'],16*GIB)

    def test_child_fresh_ram_recheck_rejects_pressure_and_disappearing_peer(self):
        w = r.reserve_wave(4,'v2')
        plan = r.stage_resources(['cpu'],concurrent=2,wave_reservation=w,host_demand_bytes=14*GIB)
        self.live['available_memory_bytes'] = 26*GIB
        with patch.dict(os.environ,r.limit_environment(plan,'cpu'),clear=True), patch.object(
                r,'_peer_credit',side_effect=[12*GIB,0]):
            with self.assertRaisesRegex(r.ResourceUnavailable,'fresh host RAM'):
                r.apply_env_limit('cpu')

    def test_only_live_direct_child_with_matching_birth_can_repay_memory(self):
        cmd = [sys.executable,'-B','-S','-c',
               "import sys; hold=bytearray(8*1024**2); print('ready',flush=True); sys.stdin.read(1)"]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
            try:
                self.assertEqual(child.stdout.readline().strip(),b'ready')
                pid, parent = child.pid, os.getpid()
                born = r.psutil.Process(pid).create_time()
                self.assertGreater(r._peer_credit({pid:born},parent,GIB),0)
                self.assertIn(str(pid),r._owned_identities({pid:born}))
                self.assertNotIn(str(pid),r._owned_identities({pid:born-1}))
                self.assertEqual(r._peer_credit({pid:born-1},parent,GIB),0)
                self.assertEqual(r._peer_credit({pid:born},parent+1,GIB),0)
                self.assertEqual(r._peer_credit([pid],parent,GIB),0)
                self.assertNotIn(str(os.getppid()),r._owned_identities([os.getppid()]))
            finally:
                child.stdin.close()
                child.wait(timeout=12)
        self.assertEqual(r._peer_credit({pid:born},parent,GIB),0)
