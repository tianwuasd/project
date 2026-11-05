"""Portable V2 resource adapters and owned-process cleanup boundaries."""
import unittest
from unittest.mock import patch,MagicMock
from pathlib import Path
import sys
import os
import subprocess
import json
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class ResourcePolicyTests(unittest.TestCase):
    def test_linux_nested_cgroup_remaining_memory_is_used(self):
        from chd_stunet._vendor.vista_live_resources import snapshot
        gib=1024**3
        files={
            '/proc/self/cgroup':'0::/work\n',
            '/proc/self/mountinfo':'1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
            '/sys/fs/cgroup/work/cpu.max':'400000 100000',
            '/sys/fs/cgroup/work/cpuset.cpus.effective':'0-7',
            '/sys/fs/cgroup/work/memory.max':str(16*gib),
            '/sys/fs/cgroup/work/memory.current':str(8*gib),
            '/sys/fs/cgroup/memory.max':str(12*gib),
            '/sys/fs/cgroup/memory.current':str(6*gib),
            '/proc/meminfo':f'MemAvailable: {80*gib//1024} kB\n',
        }
        ticks=[0]
        def read(path):
            if path=='/proc/stat':
                ticks[0]+=100
                return '\n'.join(f'cpu{i} 0 0 0 {ticks[0]} 0 0 0 0' for i in range(8))
            if path=='/sys/fs/cgroup/work/cpu.stat':
                return 'usage_usec 0\n'
            if path not in files:
                raise FileNotFoundError(path)
            return files[path]
        facts=snapshot(read_text=read,affinity=lambda:range(8),cpu_count=lambda:8,sleep=lambda _:None)
        self.assertEqual(facts['available_memory_bytes'],6*gib)
        self.assertLessEqual(facts['cpu_budget'],3)
        self.assertTrue(facts['cgroup_evidence'])

    def test_policy_keeps_at_least_four_gib_even_on_small_host(self):
        from chd_stunet.acceleration.preprocess_cpu import policy
        gib=1024**3
        facts=dict(cpu_budget=8,available_memory_bytes=6*gib,memory_budget_bytes=int(4.8*gib))
        result=policy(facts,[dict(estimate=gib)]*10)
        self.assertEqual(result['memory_budget_bytes'],2*gib)
        self.assertEqual(result['workers'],2)
        self.assertEqual(result['threads_per_worker'],1)

    def test_owned_cleanup_refuses_pid_reuse(self):
        from chd_stunet.acceleration.preprocess_cpu import stop_owned
        child=MagicMock()
        child.pid=1234
        child.poll.return_value=None
        with patch('chd_stunet.acceleration.preprocess_cpu.psutil.Process') as process:
            process.return_value.create_time.return_value=999
            with self.assertRaisesRegex(RuntimeError,'birth identity changed'):
                stop_owned(dict(process=child,born=100))
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_compute_pid_is_direct_child_and_venv_is_preserved(self):
        from chd_stunet.acceleration.preprocess_cpu import interpreter_command
        env=dict(os.environ)
        python=interpreter_command(env)
        code='import os,sys,json,numpy;print(json.dumps(dict(pid=os.getpid(),python=sys.executable,prefix=sys.prefix,numpy=numpy.__version__)))'
        child=subprocess.Popen([python,'-X','utf8','-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        stdout,stderr=child.communicate(timeout=30)
        self.assertEqual(child.returncode,0,stderr)
        record=json.loads(stdout)
        self.assertEqual(record['pid'],child.pid)
        self.assertEqual(Path(record['python']).resolve(),Path(sys.executable).resolve())
        self.assertEqual(Path(record['prefix']).resolve(),Path(sys.prefix).resolve())

    def test_next_case_uses_memory_after_previous_child_exits(self):
        from chd_stunet.acceleration.preprocess_cpu import execute_jobs
        from chd_stunet.observability.runlog import RunLog,write_json
        import contextlib,io
        root=Path(__file__).resolve().parents[1]/'outputs/tests-tmp'
        root.mkdir(parents=True,exist_ok=True)
        gib=1024**3
        active=[False]
        spawned=[]
        def facts():
            memory=(1 if active[0] else 8)*gib
            return dict(cpu_budget=8,available_memory_bytes=memory,memory_budget_bytes=int(.8*memory))
        class Child:
            def __init__(self,pid):self.pid=pid
            def poll(self):
                active[0]=False
                return 0
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            folder=Path(temporary)
            jobs=[dict(name=f'case{i}',estimate=gib,entry='unused',task=str(folder/f'{i}.task.json'),result=str(folder/f'{i}.json')) for i in range(2)]
            for job in jobs:write_json(job['task'],dict(result=job['result']))
            def launch(*args,**kwargs):
                pid=100000+len(spawned)
                job=jobs[len(spawned)]
                write_json(job['result'],dict(success=True,worker_pid=pid))
                spawned.append(pid);active[0]=True
                return Child(pid)
            with patch('chd_stunet.acceleration.preprocess_cpu.snapshot',side_effect=facts), \
                 patch('chd_stunet.acceleration.preprocess_cpu.subprocess.Popen',side_effect=launch), \
                 patch('chd_stunet.acceleration.preprocess_cpu.psutil.Process') as process, \
                 contextlib.redirect_stdout(io.StringIO()):
                process.return_value.create_time.return_value=1
                result=execute_jobs(jobs,folder,RunLog(folder),stage_name='test',workers=1,readers=1)
            self.assertEqual(result['completed'],2)
            self.assertEqual(len(spawned),2)


if __name__=='__main__':
    unittest.main()
