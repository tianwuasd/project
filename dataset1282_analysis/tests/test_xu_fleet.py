from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset1282_analysis.contracts import atomic_json, read_json
from dataset1282_analysis import xu_fleet as fleet
from dataset1282_analysis.xu_dispatch import select_devices
from dataset1282_analysis.xu_pipeline_stages import Stage


class FleetTests(unittest.TestCase):
    def test_auto_enumerates_every_authorized_visible_gpu(self):
        torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:12))
        with (patch.dict('os.environ',CUDA_VISIBLE_DEVICES='allocated'),
              patch('dataset1282_analysis.xu_dispatch.device_identity',side_effect=lambda name,_:dict(device=name,uuid=name))):
            self.assertEqual([r['device'] for r in select_devices(None,'auto',torch)],
                             [f'cuda:{i}' for i in range(12)])

    def test_fixed_batch_work_and_inference_scale_past_eight(self):
        for count in (1,2,3,4,5,8,12,16):
            cards = {f'cuda:{i}':99 for i in range(count)}
            zero = lambda _,n:[0]*n
            layout=fleet.distribute(['left','right'],dict(left=4,right=4),cards,zero)
            if count==1:
                self.assertIsNone(layout)
            else:
                used=layout['left']+layout['right']
                self.assertEqual(len(set(used)),min(count,8))
                self.assertLessEqual(max(len(v) for v in layout.values()),4)
            inference=fleet.distribute(['infer'],dict(infer=128),cards,zero)
            self.assertEqual(len(inference['infer']),count)

    def test_heterogeneous_card_needs_and_serial_fallback(self):
        def need(name,width):
            return ([80,40] if name=='large' else [25,25])[:width]
        cards={'cuda:2':80,'cuda:1':35,'cuda:0':26}
        layout=fleet.distribute(['small','large'],{'small':2,'large':2},cards,need)
        self.assertEqual(layout['large'],['cuda:2'])
        self.assertEqual(set(layout['small']),{'cuda:0','cuda:1'})
        self.assertIsNone(fleet.distribute(['large'],{'large':2},{'cuda:0':20},need))

    def test_dynamic_selection_excludes_busy_and_reuses_durable_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            pool=[f'cuda:{i}' for i in range(6)]
            args=SimpleNamespace(device_names=pool,profile='smoke',resource_policy='v2', wait_seconds=0)
            stages=[Stage(n,('train','--device',pool[0]),()) for n in ('crop64','crop128')]
            children=SimpleNamespace(pids=lambda:{})
            ready=({pool[4]:100},{d:'busy' for d in pool if d!=pool[4]})
            with (patch.object(fleet,'available',return_value=ready),
                  patch.object(fleet,'stage_resources',return_value=dict(host_budget_bytes=10**11))):
                batches=fleet.assign_wave(stages,args,out,None,children,'id')
            self.assertEqual(len(batches),2)
            self.assertEqual([g for wave in batches for _,g in wave],[[pool[4]],[pool[4]]])
            for st in stages:
                atomic_json(out/'receipts'/(st.name+'.json'),{})
            with patch.object(fleet,'available',side_effect=AssertionError('read-only replay queried GPU')):
                saved=fleet.assign_wave(stages,args,out,None,children,'id')
            self.assertEqual([g for wave in saved for _,g in wave],[[pool[4]],[pool[4]]])
            bad=out/'assignments/crop64.json'
            row=read_json(bad);row['devices']=[pool[0]]
            atomic_json(bad,row)
            with self.assertRaisesRegex(ValueError,'Invalid GPU assignment'):
                fleet.assign_wave(stages,args,out,None,children,'id')
