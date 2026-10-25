"""Explicit CPU integration check with generated volumes only; never uses patient data.

Run separately: python -B tests/synthetic_api_smoke.py
Requires the pinned training environment. Fingerprint sample count is reduced for this fixture.
"""
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))


def main():
    import numpy as np
    import nibabel as nib
    from cohort import make_splits
    from test_cohort import cases
    selected = make_splits(cases())['ct_holdG']
    allowed = set(selected['train'] + selected['val'])
    with tempfile.TemporaryDirectory(prefix='vista_synthetic_') as temp:
        root = Path(temp); raw = root/'raw'; work = root/'work'
        rng = np.random.default_rng(42)
        affine = np.diag([1.,1.25,1.5,1.])
        labels = np.zeros((24,24,24), dtype=np.int16)
        for i, value in enumerate([205,420,500,550,600,820,850]):
            x,y,z = 3+(i%2)*10,3+((i//2)%2)*10,3+(i//4)*10
            labels[x:x+6,y:y+6,z:z+6] = value
        for c in cases():
            folder = raw/c['group'];folder.mkdir(parents=True,exist_ok=True)
            image = rng.normal(100,30,labels.shape).astype(np.float32)
            nib.save(nib.Nifti1Image(image,affine),folder/(c['case_id']+'_image.nii.gz'))
            target = folder/(c['case_id']+'_label.nii.gz')
            if c['case_id'] in allowed:
                nib.save(nib.Nifti1Image(labels,affine),target)
            else:
                target.write_bytes(b'FORBIDDEN LABEL: not a NIfTI')
        os.environ['WHOLE_HEART_WORKSPACE'] = str(work)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from server_data import bind_workspace, read_json
        from model_catalog import save_selection
        code = Path(__file__).resolve().parents[1]
        bind_workspace(code,work,raw,2)
        save_selection(work,['ct_holdG'],['B0'])
        from prepare_nnunet import configure_paths
        configure_paths()
        from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
        original = DatasetFingerprintExtractor.__init__
        def small_sample(self,*args,**kwargs):
            original(self,*args,**kwargs)
            self.num_foreground_voxels_for_intensitystats = 8000
        from server_prepare import prepare
        with patch.object(DatasetFingerprintExtractor,'__init__',small_sample):
            prepare('ct_holdG',2)
        with patch('server_prepare.export_fold', side_effect=AssertionError('completed cache must be reused')):
            prepare('ct_holdG',2)
        prep=work/'04_data/derived/nnUNet_preprocessed/Dataset813_ct_holdG'
        assert read_json(prep/'fingerprint_provenance.json')['fit_ids']==selected['train']
        from project_trainers import BaselineNNUNet
        import torch
        torch.set_num_threads(2)
        trainer=BaselineNNUNet(read_json(prep/'WholeheartBaselinePlans.json'),'3d_fullres',0,read_json(prep/'dataset.json'),device=torch.device('cpu'))
        trainer.initialize()
        assert trainer.do_split()==(selected['train'],selected['val'])
        os.environ['nnUNet_n_proc_DA'] = '0'
        trainer.on_train_start()
        trainer.on_train_epoch_start()
        batch=next(trainer.dataloader_train)
        with torch.no_grad():
            assert np.isfinite(trainer.validation_step(next(trainer.dataloader_val))['loss']).all()
        loss=trainer.train_step(batch)['loss']
        assert np.isfinite(loss).all()
        checkpoint=work/'cpu_checkpoint.pth'
        trainer.save_checkpoint(str(checkpoint))
        from smoke_train import state_hash
        before=state_hash(trainer.network)
        trainer.load_checkpoint(str(checkpoint))
        assert before==state_hash(trainer.network)
        assert trainer.optimizer.state
        assert np.isfinite(trainer.train_step(batch)['loss']).all()
        trainer.set_deep_supervision_enabled(False)
        trainer.network.eval()
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,
            perform_everything_on_device=False,device=torch.device('cpu'),verbose=False,allow_tqdm=False)
        predictor.manual_initialization(trainer.network,trainer.plans_manager,trainer.configuration_manager,
            [trainer.network.state_dict()],trainer.dataset_json,'BaselineNNUNet',None)
        reader=trainer.plans_manager.image_reader_writer_class()
        path=raw/'A ct_train'/next(c['case_id']+'_image.nii.gz' for c in cases() if c['case_id'] in allowed and c['group']=='A ct_train')
        images,properties=reader.read_images([str(path)])
        segmentation=predictor.predict_single_npy_array(images,properties,None,None,False)
        output=work/'prediction.nii.gz'; reader.write_seg(segmentation,str(output),properties)
        restored=nib.load(output)
        assert restored.shape==labels.shape and np.allclose(restored.affine,affine)
        from data_pipeline import inverse_labels, map_labels
        assert np.array_equal(map_labels(inverse_labels(np.asarray(restored.dataobj))),np.asarray(restored.dataobj))
        print('PASS: source-only export/fingerprint/planning/preprocessing/cache reuse; CPU gradient, checkpoint restore + step; original-grid inference and label roundtrip. Synthetic data only.',flush=True)


if __name__=='__main__':main()
