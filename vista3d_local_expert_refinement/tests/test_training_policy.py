"""Resource decisions preserve scientific budgets and explain rejected options."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'05_src'))
from training_policy import (distribution, cpu_candidates, select_trial,
                             validate_gpu_count, receipt_compatibility)


class TrainingPolicyTests(unittest.TestCase):
    def test_gpu_limit_is_dynamic_not_five_and_unknown_capacity_is_not_invented(self):
        self.assertEqual(validate_gpu_count('8', maximum=8), 8)
        self.assertEqual(validate_gpu_count(12, maximum=16), 12)
        self.assertIsNone(validate_gpu_count('auto', maximum=8))
        for value in [True, 1.5, '2.0', -1, 0]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_gpu_count(value)
        with self.assertRaises(ValueError):
            validate_gpu_count(9, maximum=8)

    def test_global_batch_is_not_multiplied_to_fill_cards(self):
        value = distribution(2, ['GPU-a', 'GPU-b', 'GPU-c'], 8)
        self.assertEqual(value['world_size'], 2)
        self.assertEqual(value['local_batch_size'], 1)
        self.assertEqual(value['global_batch_size'], 2)
        self.assertEqual(value['unused_gpu_uuids'], ['GPU-c'])
        self.assertIn('batch', value['reason'])
        # The implementation supports all physical cards when the budget permits.
        value = distribution(8, [f'GPU-{i}' for i in range(8)], 8)
        self.assertEqual(value['world_size'], 8)

    def test_partition_is_balanced_and_can_start_on_one_card(self):
        self.assertEqual(distribution(6, ['GPU-a'])['world_size'], 1)
        self.assertEqual(distribution(6, ['GPU-a','GPU-b','GPU-c'])['local_batch_size'], 2)
        self.assertEqual(distribution(3, ['GPU-a','GPU-b'])['world_size'], 1)
        for batch, devices in [(0,['GPU-a']), (True,['GPU-a']), (2,[]), (2,['GPU-a','GPU-a'])]:
            with self.subTest(batch=batch,devices=devices), self.assertRaises(ValueError):
                distribution(batch, devices)

    def test_explicit_auto_batch_can_use_all_cards_and_reports_sampling_change(self):
        devices=[f'GPU-{i}' for i in range(8)]
        value=distribution(2,devices,allow_batch_growth=True,per_device_batch=1)
        self.assertEqual(value['world_size'],8)
        self.assertEqual(value['global_batch_size'],8)
        self.assertEqual(value['reference_global_batch_size'],2)
        self.assertEqual(value['sampling_ratio'],4.)
        self.assertEqual(value['train_patches_per_epoch'],2000)
        self.assertEqual(value['train_iterations_per_epoch'],250)
        with self.assertRaises(ValueError):distribution(2,devices,per_device_batch=2)

    def test_cpu_pool_budget_includes_all_ranks_and_validation_workers(self):
        snapshot = dict(cpu_slots=24, available_memory_bytes=64*2**30,
                        shared_memory_available_bytes=16*2**30, io_pressure={'avg10':0})
        values = cpu_candidates(snapshot, [160,192,192], 2, 2)
        self.assertTrue(any(v['augmentation_workers'] > 0 for v in values))
        for value in values:
            self.assertLessEqual(value['total_cpu_slots'], 24)
            self.assertLessEqual(value['estimated_ram_bytes'], 32*2**30)
            self.assertLessEqual(value['estimated_shm_bytes'], 8*2**30)
        tiny = dict(snapshot, cpu_slots=1, shared_memory_available_bytes=64*2**20)
        self.assertEqual([v['augmentation_workers'] for v in cpu_candidates(tiny,[160,192,192],2,1)], [0])

    def test_unknown_or_pressured_shared_resources_keep_serial_fallback(self):
        unknown = cpu_candidates(dict(cpu_slots=12), [160,192,192], 2, 1)
        self.assertTrue(all(v['augmentation_workers']==0 for v in unknown))
        pressured = cpu_candidates(dict(cpu_slots=24, available_memory_bytes=2**30,
                                        shared_memory_available_bytes=2**30,
                                        io_pressure={'avg10':30}),[160,192,192],2,2)
        self.assertEqual(len(pressured), 1)
        self.assertEqual(pressured[0]['augmentation_workers'], 0)

    def test_selection_requires_enough_samples_and_comparable_budget(self):
        def trial(seconds, threads=2, **extra):
            return dict(status='passed', measured_training_steps=10, global_batch_size=2,
                        median_step_seconds=seconds, validation_step_seconds=.2,
                        total_cpu_slots=threads, **extra)
        candidates = [trial(2.,1), trial(1.,4), trial(1.02,2)]
        self.assertEqual(select_trial(candidates, global_batch=2)['total_cpu_slots'], 2)
        candidates[1]['global_batch_size']=6
        candidates[2]['measured_training_steps']=1
        self.assertEqual(select_trial(candidates, global_batch=2)['total_cpu_slots'], 1)
        with self.assertRaises(ValueError):
            select_trial([dict(candidates[0],median_step_seconds=float('nan'))], global_batch=2)

    def test_receipt_requires_complete_status_matching_identity_ttl_and_capacity(self):
        identity={'dataset':'wholeheart','model':'B0','plan':'abc','code':'def'}
        value=dict(status='passed', identity=identity, created_at=100., ttl_seconds=3600,
                   required_cpu_slots=8, required_ram_bytes=2**30,
                   required_shm_bytes=2**29, committed=True)
        snapshot=dict(cpu_slots=16,available_memory_bytes=8*2**30,shared_memory_available_bytes=2**30)
        self.assertEqual(receipt_compatibility(value,identity,snapshot,now=1000), [])
        self.assertTrue(receipt_compatibility(value,identity,snapshot,now=4000))
        self.assertTrue(receipt_compatibility(dict(value,status='failed'),identity,snapshot,now=1000))
        self.assertTrue(receipt_compatibility(value,dict(identity,plan='changed'),snapshot,now=1000))
        self.assertTrue(receipt_compatibility(value,identity,dict(snapshot,cpu_slots=2),now=1000))


if __name__=='__main__': unittest.main()
