"""Metadata-only resource normalization; real API exercised in loader smoke."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from nnunet_batches import build_loaders


class BatchSourceConfigTests(unittest.TestCase):
    def trainer(self):
        return SimpleNamespace(is_cascaded=False,do_split=lambda:(['fit'],['cal']),
            configure_rotation_dummyDA_mirroring_and_inital_patch_size=lambda:([],False,[16]*3,[]),
            plans_manager=SimpleNamespace(plans={}),dataset_json={},preprocessed_dataset_folder='synthetic',
            batch_size=2,oversample_foreground_percent=.33,probabilistic_oversampling=False,
            _get_deep_supervision_scales=lambda:[[1]*3],current_epoch=3,local_rank=1,
            device=SimpleNamespace(type='cpu'),num_iterations_per_epoch=250,num_val_iterations_per_epoch=50)

    def test_rank_threads_total_prefetch_phase_and_actual_transport_budget(self):
        settings=dict(main_threads=2,train_workers=3,val_workers=1,worker_threads=4,
                      val_worker_threads=2,prefetch=5,val_prefetch=2)
        train,val=build_loaders(self.trainer(),dict(batch_bytes=1024,rank_settings=[{},settings]))
        try:
            self.assertEqual((train.workers,train.worker_threads,train.prefetch,train.phase_steps),(3,4,5,250))
            self.assertEqual((val.workers,val.worker_threads,val.prefetch,val.phase_steps),(1,2,2,50))
            self.assertEqual((train.max_shared_bytes,val.max_shared_bytes),(5120,2048))
            self.assertEqual(train.loader.worker_threads,4)
        finally: train.close();val.close()

    def test_legacy_aliases_keep_total_prefetch_and_no_fabricated_shm_budget(self):
        train,val=build_loaders(self.trainer(),dict(augmentation_workers=2,validation_workers=1,threads=3,prefetch=4))
        try:
            self.assertEqual((train.workers,train.worker_threads,train.prefetch),(2,3,4))
            self.assertEqual((val.workers,val.worker_threads,val.prefetch),(1,3,4))
            self.assertEqual(train.max_shared_bytes,0)
        finally: train.close();val.close()

    def test_serial_source_uses_rank_main_threads_instead_of_idle_worker_setting(self):
        train,val=build_loaders(self.trainer(),dict(main_threads=1,train_workers=0,val_workers=0,
            worker_threads=2,val_worker_threads=2,prefetch=0,val_prefetch=0))
        try:self.assertEqual((train.worker_threads,val.worker_threads),(1,1))
        finally:train.close();val.close()


if __name__=='__main__': unittest.main()
