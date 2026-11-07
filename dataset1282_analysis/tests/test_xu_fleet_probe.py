import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dataset1282_analysis.xu_gpu_probe import gpu_snapshot, visible_devices


class FleetProbeTests(unittest.TestCase):
    def test_real_cli_refuses_an_external_compute_process(self):
        inventory = gpu_snapshot()
        occupied = [d for d in visible_devices() if inventory[d['uuid']]['compute_pids']]
        if not occupied:
            self.skipTest('当前可见 GPU 没有外部计算进程')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'probe'
            result = subprocess.run([sys.executable,'-B',str(root/'scripts/xu_fleet_probe.py'),
                                     '--devices',occupied[0]['device'],'--stage','inference',
                                     '--exclude-physical-gpus',
                                     '--output',str(out)],capture_output=True,text=True,timeout=40)
            self.assertEqual(result.returncode,1,(result.stdout or '')+(result.stderr or ''))
            row = json.loads((out/'result.json').read_text())
            self.assertEqual(row['state'],'failed')
            self.assertFalse(row['groups'])
            self.assertIn('External compute',row['unavailable'][occupied[0]['device']])
            self.assertFalse(list(out.glob('*-admission.json')))

    def test_real_cli_protects_existing_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            evidence = out/'keep.txt'
            evidence.write_text('retained',encoding='utf-8')
            result = subprocess.run([sys.executable,'-B',str(root/'scripts/xu_fleet_probe.py'),
                                     '--devices','all-visible','--output',str(out)],
                                    capture_output=True,text=True,timeout=40)
            self.assertEqual(result.returncode,1)
            self.assertEqual(evidence.read_text(),'retained')
