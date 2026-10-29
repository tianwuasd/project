"""User batch cap and the real six-rank supply starvation regression."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import cooperative_control as control
import live_resources as live


class CappedBatchTests(unittest.TestCase):
    def test_production_six_rank_capacity_and_sampling_use_eight(self):
        from server_data import write_json
        devices=[f'GPU-{i}' for i in range(6)];seen=[]
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            write_json(work/'04_data/manifests/splits_v1.json',dict(folds={'f':dict(dataset_id=813)}))
            write_json(work/'04_data/derived/nnUNet_preprocessed/Dataset813_f/WholeheartBaselinePlans.json',
                dict(configurations={'3d_fullres':dict(batch_size=2,patch_size=[8,8,8])}))
            def trial(w,f,m,g,local,resource,kind,**kwargs):
                seen.append(resource)
                return dict(status='passed',local_batch_size=local,rank_results=[])
            with patch.object(control,'admit',return_value=(devices,{g:20*2**30 for g in devices},[])),\
                 patch('task_resources.probe',return_value=dict(cpu_budget=199,memory_budget_bytes=157324879462)),\
                 patch.object(control,'trial',side_effect=trial):
                value=control.tune(work,'f','B0','formal',devices)
            self.assertEqual(value['global_batch_size'],8)
            self.assertEqual(value['train_patches_per_epoch'],2000)
            self.assertEqual(value['local_batch_sizes'],[2,2,1,1,1,1])
            self.assertTrue(all(x['global_batch_size']<=8 for x in seen))

    def test_four_six_eight_share_eight_patches(self):
        from batch_distribution import layout
        for world,expected in [(4,[2]*4),(5,[2,2,2,1,1]),(6,[2,2,1,1,1,1]),(7,[2,1,1,1,1,1,1]),(8,[1]*8)]:
            with self.subTest(world=world):
                self.assertEqual(layout(2,world),expected)
        self.assertEqual(layout(2,2),[2,2])
        self.assertEqual(layout(1,6),[1]*6)
        for bad in (0,-1,True,1.5):
            with self.assertRaises(ValueError):layout(bad,6)
        with self.assertRaises(ValueError):layout(2,9)

    def test_actual_payload_and_private_memory_follow_local_batch(self):
        facts=dict(cpu_budget=199,memory_budget_bytes=157324879462,cpu_ids=list(range(256)))
        with patch('batch_pool.shared_free_bytes',return_value=1000*2**30):
            value=control.stage_config(facts,[160,192,192],2,6,
                dict(payload_bytes_per_sample=37074272,private_bytes_per_sample=754974720),global_batch_size=8)
        self.assertEqual(value['global_batch_size'],8)
        self.assertEqual(value['local_batch_sizes'],[2,2,1,1,1,1])
        rows=value['rank_settings']
        self.assertEqual(rows[0]['batch_bytes'],2*rows[-1]['batch_bytes'])
        self.assertEqual(rows[0]['worker_private_bytes'],2*rows[-1]['worker_private_bytes'])
        self.assertLessEqual(value['estimated_ram_bytes'],facts['memory_budget_bytes'])

    def test_reproduced_thirteen_workers_three_slots_is_not_admitted(self):
        value=live.training_policy(dict(cpu_budget=199,memory_budget_bytes=157324879462,
            worker_private_bytes=1509949440),6,74148544)
        for row in value['rank_settings']:
            self.assertGreaterEqual(row['prefetch'],row['train_workers'])
            self.assertGreaterEqual(row['val_prefetch'],row['val_workers'])
        self.assertLessEqual(value['estimated_ram_bytes'],157324879462)

    def test_live_expansion_cannot_create_idle_reserved_workers(self):
        value=live.training_policy(dict(cpu_budget=199,memory_budget_bytes=157324879462,
            worker_private_bytes=1509949440),6,74148544)
        window=dict(steps=20,seconds=40,wait_fraction=.5)
        result=live.feedback_policy(value,[window,window],value)['settings']
        for row in result['rank_settings']:
            self.assertGreaterEqual(row['prefetch'],row['train_workers'])
            self.assertGreaterEqual(row['val_prefetch'],row['val_workers'])
        self.assertLessEqual(result['estimated_ram_bytes'],result['memory_budget_bytes'])


if __name__=='__main__':unittest.main()
