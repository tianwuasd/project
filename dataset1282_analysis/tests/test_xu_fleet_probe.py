import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dataset1282_analysis.xu_gpu_probe import gpu_snapshot, visible_devices
from dataset1282_analysis.xu_resources import gpu_budget


class FleetProbeTests(unittest.TestCase):
    def test_real_cli_admits_an_occupied_gpu_with_memory(self):
        inventory = gpu_snapshot()
        occupied = [d for d in visible_devices() if inventory[d['uuid']]['compute_pids'] and
                    gpu_budget(inventory[d['uuid']]['free_bytes'],
                               inventory[d['uuid']]['total_bytes'],'v2') >= 1024**3]
        if not occupied:
            self.skipTest('当前没有显存充足且存在其他任务的显卡')
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'probe'
            result = subprocess.run([sys.executable,'-B',str(root/'scripts/xu_fleet_probe.py'),
                                     '--devices',occupied[0]['device'],'--stage','inference',
                                     '--exclude-physical-gpus',
                                     '--output',str(out)],capture_output=True,text=True,timeout=40)
            self.assertEqual(result.returncode,0,(result.stdout or '')+(result.stderr or ''))
            logs=list(out.glob('fleet-inference-*/acceptance.log'))
            self.assertEqual(len(logs),1)
            events=[json.loads(line) for line in logs[0].read_text(encoding='utf-8').splitlines()]
            row=next(event['contents'] for event in events
                     if event['event']=='report' and event['path'].endswith('result.json'))
            self.assertEqual(row['state'],'passed')
            self.assertEqual(len(row['groups']),1)
            self.assertNotIn(occupied[0]['device'],row['unavailable'])
            self.assertTrue(list(out.glob('*-admission.json')))
            self.assertFalse(list(out.rglob('.scratch')))
            self.assertTrue(all(p.suffix=='.log' for p in out.rglob('*') if p.is_file()))

    def test_real_cli_protects_existing_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            evidence = out/'keep.txt'
            evidence.write_text('retained',encoding='utf-8')
            result = subprocess.run([sys.executable,'-B',str(root/'scripts/xu_fleet_probe.py'),
                                     '--devices','cpu','--output',str(out)],
                                    capture_output=True,text=True,timeout=40)
            self.assertEqual(result.returncode,1)
            self.assertEqual(evidence.read_text(),'retained')
            self.assertFalse(list(out.rglob('.scratch')))
            self.assertTrue(all(p==evidence or p.suffix=='.log' for p in out.rglob('*') if p.is_file()))
