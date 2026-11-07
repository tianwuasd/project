import os
import subprocess
import sys
import unittest

import psutil

from dataset1282_analysis import xu_resources as r
from dataset1282_analysis.resources import GIB
from test_xu_acceleration_live import environment


class XuResourceTests(unittest.TestCase):
    def test_actual_host_admission_and_worker_limit(self):
        plan = r.stage_resources(['cpu'],threads=1,policy='v2')
        self.assertEqual(plan['threads'],1)
        self.assertGreater(plan['host_per_stage_bytes'],0)
        with environment(**r.limit_environment(plan,'cpu')):
            result = r.apply_env_limit('cpu')
            self.assertGreater(result['host_budget_bytes'],0)
            self.assertLessEqual(result['host_budget_bytes'],plan['host_per_stage_bytes'])
        with self.assertRaises(r.ResourceUnavailable):
            r.stage_resources(['cpu'],host_demand_bytes=psutil.virtual_memory().total*100)

    def test_explicit_reserves_and_invalid_admission(self):
        self.assertEqual(r.gpu_budget(40*GIB,80*GIB,'v2'),32*GIB)
        self.assertEqual(r.gpu_budget(4*GIB,24*GIB,'v2'),0)
        with self.assertRaises(ValueError):
            r.stage_resources(['cpu','cpu'])
        with self.assertRaises(r.ResourceUnavailable):
            r.require_applied_group(['cuda:0','cuda:1'])

    def test_actual_child_identity_and_private_memory_credit(self):
        command = [sys.executable,'-c','import sys; a=bytearray(8*1024**2);print("ready",flush=True);sys.stdin.read()']
        with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as child:
            self.assertEqual(child.stdout.readline().strip(),b'ready')
            birth = psutil.Process(child.pid).create_time()
            self.assertIn(str(child.pid),r._owned_identities({child.pid:birth}))
            self.assertGreater(r._peer_credit({child.pid:birth},os.getpid(),GIB),0)
            self.assertNotIn(str(child.pid),r._owned_identities({child.pid:birth-1}))
            self.assertEqual(r._peer_credit({child.pid:birth-1},os.getpid(),GIB),0)
            self.assertEqual(r._peer_credit({child.pid:birth},os.getpid()+1,GIB),0)
            child.stdin.close()
            child.wait(timeout=10)
        self.assertEqual(r._peer_credit({child.pid:birth},os.getpid(),GIB),0)
