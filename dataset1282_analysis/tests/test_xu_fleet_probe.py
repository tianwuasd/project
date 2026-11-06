import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset1282_analysis import xu_capacity, xu_fleet
from dataset1282_analysis.contracts import atomic_json, read_json
from dataset1282_analysis.xu_resources import ResourceUnavailable


def cli_module():
    path = Path(__file__).resolve().parents[1]/'scripts/xu_fleet_probe.py'
    spec = importlib.util.spec_from_file_location('fleet_probe_cli',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FleetProbeTests(unittest.TestCase):
    def exercise(self, small):
        cli = cli_module()
        cards = [f'cuda:{i}' for i in range(12)]
        inspected, admitted, launched = [], [], []
        def available_device(devices,**kwargs):
            device, = devices
            inspected.append(device)
            if device==cards[10]:
                raise ResourceUnavailable('external compute PID')
            free = 1 if small and device==cards[11] else 1024
            return {'devices':{device:{'allocator_budget_bytes':free}}}
        def demands(stage,devices,spec):
            self.assertEqual(stage,'3d64')
            sizes = shares(devices)
            return {d:count*100 for d,count in sizes.items()}
        def shares(devices):
            each, extra = divmod(4,len(devices))
            return {d:each+(i<extra) for i,d in enumerate(devices)}
        def admit(devices,**kwargs):
            admitted.append(list(devices))
            self.assertEqual(kwargs['cuda_demand_bytes'],demands('3d64',devices,{}))
            self.assertEqual((kwargs['host_demand_bytes'],kwargs['policy']),(512,'v2'))
            return {'group':list(devices)}
        def run(script,argv,name,out,owner,args):
            self.assertEqual(script,'xu_capacity.py')
            self.assertEqual(argv[:8],['--stage','3d64','--device',args.assigned_devices[0],
                                       '--policy','v2','--supervision','marginal'])
            self.assertEqual(args.overlay['group'],args.assigned_devices)
            launched.append(list(args.assigned_devices))
            dist = shares(args.assigned_devices)
            atomic_json(argv[-1]/'capacity.json', dict(state='passed',optimizer_steps=1,
                per_device={d:{'peak_cuda_allocated_bytes':10} for d in dist},
                distribution={'assigned_devices':list(dist),'cases_by_device':dist,
                              'batches_by_layout':{','.join(map(str,dist.values())):1}}))
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out = Path(temp.name)/'probe'
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:12))
        with (patch.dict(sys.modules,torch=torch),
              patch.dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,range(12)))),
              patch('dataset1282_analysis.xu_dispatch.device_identity', side_effect=lambda n,t:dict(device=n,uuid=n)),
              patch.object(xu_fleet,'stage_resources',side_effect=available_device),
              patch.object(xu_capacity,'specification',return_value={'estimated_host_bytes':512}),
              patch.object(xu_capacity,'device_demands',side_effect=demands),
              patch.object(cli,'stage_resources',side_effect=admit),
              patch.object(cli,'combined_environment',side_effect=lambda p,g:p),
              patch.object(cli,'_execute',side_effect=run),contextlib.redirect_stdout(io.StringIO())):
            code = cli.main(['--devices','auto','--stage','3d64','--threads','2','--output',str(out)])
        result = read_json(out/'result.json')
        self.assertEqual(inspected,cards)
        self.assertEqual(result['unavailable'],{cards[10]:'external compute PID'})
        self.assertEqual(admitted,launched)
        self.assertEqual(launched,[g['devices'] for g in result['groups']])
        self.assertTrue(all(sum(g['work']['cases_by_device'].values())==4 for g in result['groups']))
        used = [device for group in launched for device in group]
        self.assertEqual(len(used),len(set(used)))
        self.assertCountEqual(used,[d for d in cards if d!=cards[10] and (not small or d!=cards[11])])
        return code,result,[len(group) for group in launched]

    def test_12_visible_cards_exhaust_eligible_pool_in_fixed_global_batches(self):
        code,row,widths = self.exercise(False)
        self.assertEqual((code,row['state'],widths),(0,'passed',[4,4,3]))

    def test_insufficient_last_card_is_retained_as_unqualified_after_other_groups_pass(self):
        code,row,widths = self.exercise(True)
        self.assertEqual((code,row['state'],widths),(1,'failed',[4,4,2]))
        self.assertEqual(row['unqualified'],['cuda:11'])
