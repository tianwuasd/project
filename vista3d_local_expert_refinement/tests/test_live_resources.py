"""Live resource facts and direct policy contract; no CUDA or candidate trials."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'05_src'))
import live_resources as live


class LiveResourcesTests(unittest.TestCase):
    def ample(self, cpu=191, memory=256*2**30):
        return dict(cpu_budget=cpu, memory_budget_bytes=memory, io_pressure=None)

    def test_permitted_idle_sample_and_budget(self):
        files={'/proc/meminfo':'MemAvailable: 1000000 kB',
               '/proc/self/cgroup':'0::/team/job',
               '/proc/self/mountinfo':'1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
               '/sys/fs/cgroup/team/cpu.max':'23900 100',
               '/sys/fs/cgroup/team/job/cpu.max':'max 100',
               '/sys/fs/cgroup/team/memory.max':'1000000000',
               '/sys/fs/cgroup/team/memory.current':'200000000',
               '/sys/fs/cgroup/team/job/memory.max':'max'}
        def read(path):
            if path == '/proc/stat':
                return '\n'.join(f'cpu{i} 0 0 0 {stat[0]}' for i in range(256))
            if path in files:return files[path]
            raise OSError(path)
        stat=[0]
        value=live.snapshot(read_text=read, affinity=lambda:range(256), cpu_count=lambda:256,
                            sleep=lambda _:stat.__setitem__(0,100))
        self.assertEqual(value['cpu_permitted'],239)
        self.assertEqual(value['cpu_available'],239)
        self.assertEqual(value['cpu_budget'],191)
        self.assertEqual(value['memory_budget_bytes'],640000000)
        self.assertTrue(value['missing_evidence'])

    def test_tiny_affinity_uses_only_permitted_cpu_not_host_idle(self):
        tick=[0]
        def read(path):
            if path=='/proc/stat':
                return f'cpu0 {tick[0]} 0 0 0\ncpu1 0 0 0 {tick[0]}\n'
            raise OSError(path)
        value=live.snapshot(read_text=read, affinity=lambda:{0},cpu_count=lambda:256,
                            sleep=lambda _:tick.__setitem__(0,100))
        self.assertEqual(value['cpu_available'],0)
        self.assertEqual(value['cpu_budget'],0)
        self.assertEqual(live.case_policy(value,12,1)['workers'],0)

    def test_one_core_and_zero_ram_do_not_invent_capacity(self):
        self.assertEqual(live.case_policy(self.ample(0),12,1)['capacity_status'],'known_insufficient_cpu')
        self.assertEqual(live.training_policy(self.ample(0),1,1)['status'],'blocked')
        self.assertEqual(live.case_policy(self.ample(memory=0),12,1)['workers'],0)
        self.assertEqual(live.training_policy(self.ample(memory=0),1,1)['status'],'blocked')
        self.assertEqual(live.training_policy(self.ample(2),3,1)['status'],'blocked')

    def test_nested_v1_limits_and_physical_mapping(self):
        files={'/proc/self/cgroup':'2:cpu,cpuacct:/tenant/job\n3:memory:/tenant/job\n4:cpuset:/tenant/job',
          '/proc/self/mountinfo':'1 0 0:1 / /cg/cpu rw - cgroup cpu,cpuacct rw,cpu,cpuacct\n2 0 0:2 / /cg/mem rw - cgroup memory rw,memory\n3 0 0:3 / /cg/set rw - cgroup cpuset rw,cpuset',
          '/cg/cpu/tenant/cpu.cfs_quota_us':'150000','/cg/cpu/tenant/cpu.cfs_period_us':'100000',
          '/cg/set/tenant/cpuset.cpus':'0-1','/cg/mem/tenant/memory.limit_in_bytes':'1000','/cg/mem/tenant/memory.usage_in_bytes':'100',
          '/sys/devices/system/cpu/cpu0/topology/core_id':'0','/sys/devices/system/cpu/cpu1/topology/core_id':'0',
          '/sys/devices/system/cpu/cpu0/topology/physical_package_id':'0','/sys/devices/system/cpu/cpu1/topology/physical_package_id':'0',
          '/sys/devices/system/node/online':'0','/sys/devices/system/node/node0/cpulist':'0-1'}
        def read(path):
            if path in files:return files[path]
            raise OSError(path)
        value=live.snapshot(read_text=read,affinity=lambda:{0,1,2},cpu_count=lambda:3,sleep=lambda _:None)
        self.assertEqual(value['cpu_permitted'],1.5)
        self.assertEqual(value['cpu_budget'],1)
        self.assertEqual(value['physical_core_count'],1)
        self.assertEqual(value['numa_nodes'],{'0':[0,1]})
        self.assertEqual(value['memory_budget_bytes'],720)

    def test_ancestor_usage_and_fractional_quota_are_not_free_capacity(self):
        tick=[0]
        files={'/proc/self/cgroup':'0::/team/job',
               '/proc/self/mountinfo':'1 0 0:1 / /cg rw - cgroup2 cgroup rw',
               '/cg/team/cpu.max':'100000 100000'}
        def read(path):
            if path=='/proc/stat':return f'cpu0 0 0 0 {tick[0]}\ncpu1 0 0 0 {tick[0]}'
            if path=='/cg/team/cpu.stat':return f'usage_usec {tick[0]*2000}'
            if path in files:return files[path]
            raise OSError(path)
        value=live.snapshot(read_text=read,affinity=lambda:{0,1},cpu_count=lambda:2,
                            sleep=lambda _:tick.__setitem__(0,100),sample_seconds=.2)
        self.assertEqual(value['cpu_available'],0)
        self.assertEqual(value['cpu_budget'],0)

    def test_unknown_resources_are_explicit_bounded_fallback(self):
        policy=live.training_policy({},1,1024)
        self.assertEqual(policy['status'],'bounded_fallback')
        self.assertEqual(policy['rank_settings'][0]['train_workers'],0)
        self.assertEqual(policy['total_cpu_slots'],1)
        self.assertTrue(policy['reasons'])

    def test_empty_effective_cpuset_and_missing_memory_usage_stay_explicit(self):
        files={'/proc/self/cgroup':'0::/job',
               '/proc/self/mountinfo':'1 0 0:1 / /cg rw - cgroup2 cgroup rw',
               '/cg/job/cpuset.cpus.effective':'', '/cg/job/memory.max':'1000',
               '/proc/meminfo':'MemAvailable: 10000 kB'}
        def read(path):
            if path in files:return files[path]
            raise OSError(path)
        value=live.snapshot(read_text=read,affinity=lambda:{0,1},cpu_count=lambda:2,sleep=lambda _:None)
        self.assertEqual(value['cpu_budget'],0)
        self.assertIsNone(value['memory_budget_bytes'])
        self.assertIn('finite cgroup RAM limit has unknown remaining capacity',value['missing_evidence'])
        old=live.case_policy(dict(cpu_slots=10,available_memory_bytes=1000),12,1)
        self.assertEqual(old['cpu_budget'],8)
        budgeted=live.case_policy(dict(cpu_slots=10,cpu_slots_are_budgeted=True,memory_budget_bytes=1000),12,1)
        self.assertEqual(budgeted['cpu_budget'],10)

    def test_case_capacity_has_no_eight_cap_and_io_preserves_compute(self):
        value=live.case_policy(self.ample(),12,2**30)
        self.assertEqual((value['workers'],value['threads_per_worker']),(12,4))
        threaded=live.case_policy(self.ample(),2,2**30,supported_threads=64)
        self.assertEqual(threaded['threads_per_worker'],64)
        pressure=live.case_policy(dict(self.ample(),io_pressure={'avg10':40}),12,2**30)
        self.assertEqual(pressure['workers'],12)
        self.assertEqual(pressure['read_concurrency'],2)

    def test_ample_single_rank_and_eight_rank_aggregate(self):
        value=live.training_policy(self.ample(),1,2**20)
        rank=value['rank_settings'][0]
        self.assertEqual((rank['main_threads'],rank['train_workers'],rank['worker_threads']),(4,16,2))
        self.assertEqual((rank['val_workers'],rank['val_worker_threads']),(4,2))
        self.assertEqual(rank['prefetch'],32)
        eight=live.training_policy(self.ample(),8,2**20)
        self.assertLessEqual(eight['total_cpu_slots'],191)
        self.assertEqual(eight['rank_settings'][0]['worker_threads'],1)

    def test_both_resident_pools_memory_and_no_shm_pipe_gate(self):
        batch=2**20
        value=live.training_policy(dict(self.ample(),shared_memory_available_bytes=0),1,batch)
        rank=value['rank_settings'][0]
        self.assertEqual(rank['resident_pool_bytes'],(16+4)*2*batch)
        self.assertGreaterEqual(value['estimated_ram_bytes'],rank['resident_pool_bytes']+(32+8)*batch)
        bounded=live.training_policy(self.ample(memory=20*batch),1,batch)
        self.assertLessEqual(bounded['estimated_ram_bytes'],20*batch)
        self.assertLess(bounded['rank_settings'][0]['train_workers'],16)
        if bounded['rank_settings'][0]['val_workers']:
            self.assertGreater(bounded['rank_settings'][0]['val_prefetch'],0)
        self.assertEqual(value['transport'],'pipe')
        if bounded['rank_settings'][0]['train_workers']:
            self.assertGreater(bounded['rank_settings'][0]['prefetch'],0)

    def test_feedback_waits_two_windows_and_never_shrinks_for_no_gain(self):
        settings=live.training_policy(self.ample(),1,2**20)
        capacity=live.training_policy(self.ample(cpu=300),1,2**20)
        window=dict(steps=20,seconds=30,wait_fraction=.3)
        self.assertEqual(live.feedback_policy(settings,[window],capacity)['action'],'keep')
        expanded=live.feedback_policy(settings,[window,window],capacity)
        self.assertEqual(expanded['action'],'expand')
        self.assertEqual(expanded['settings']['cpu_budget'],300)
        self.assertEqual(expanded['settings']['rank_settings'][0]['train_workers'],24)
        regroup=live.training_policy(self.ample(),2,2**20)
        self.assertEqual(live.feedback_policy(settings,[],regroup)['boundary'],'full_epoch')
        reduced=live.training_policy(self.ample(cpu=12),1,2**20)
        self.assertEqual(live.feedback_policy(settings,[],reduced)['action'],'recover')
        quiet=dict(window,wait_fraction=.05)
        self.assertEqual(live.feedback_policy(settings,[quiet,quiet],capacity)['action'],'keep')
        storage=dict(window,io_pressure={'avg10':40})
        self.assertEqual(live.feedback_policy(settings,[storage,storage],capacity)['action'],'limit_reads')


if __name__=='__main__':unittest.main()
