import csv
import subprocess
import sys
import threading
import unittest

import torch

from dataset1282_analysis.xu_resources import _check_device, gpu_budget
from dataset1282_analysis.xu_runtime_guard import gpu_card_status


class SharedGpuLiveTests(unittest.TestCase):
    def test_real_external_cuda_process_with_memory_is_admitted(self):
        if not torch.cuda.is_available():
            self.skipTest('当前机器没有可用 CUDA 设备')
        child = subprocess.Popen(
            [sys.executable, '-B', '-c',
             'import sys,torch; x=torch.ones(64*1024*1024,device="cuda",dtype=torch.uint8); '
             'torch.cuda.synchronize(); print("READY",flush=True); sys.stdin.read()'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        timer = threading.Timer(30, child.kill)
        timer.start()
        try:
            self.assertEqual(child.stdout.readline().strip(), 'READY', child.stderr.read() if child.poll() is not None else '')
            report = subprocess.run(
                ['nvidia-smi', '--query-gpu=uuid,memory.free,memory.total',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            uuid, free_mib, total_mib = next(csv.reader(report.stdout.splitlines()))
            info = dict(device='cuda:0', uuid=uuid.strip(), free_bytes=int(free_mib.strip())*1024**2,
                        total_bytes=int(total_mib.strip())*1024**2, compute_pids=[child.pid])
            if gpu_budget(info['free_bytes'], info['total_bytes'], 'v2') == 0:
                self.skipTest('当前显卡剩余显存低于保留值')
            checked = _check_device(info, 'v2', {}, True)
            status, low_memory = gpu_card_status(info, 'cuda:0', {})
            self.assertEqual(checked['external_pids'], [child.pid])
            self.assertEqual(checked['allocator_budget_bytes'],
                             gpu_budget(info['free_bytes'], info['total_bytes'], 'v2'))
            self.assertEqual(status['external_pids'], [child.pid])
            self.assertFalse(low_memory)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            child.stdin.close()
            child.wait(timeout=10)
            child.stdout.close()
            child.stderr.close()
            timer.cancel()


if __name__ == '__main__':
    unittest.main()
