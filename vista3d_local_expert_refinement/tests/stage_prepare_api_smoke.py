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
        from task_resources import probe
        observed=probe()
        observed['cpu_budget']=min(4,observed['cpu_budget'])
        if observed['cpu_budget']<1:raise RuntimeError('no currently permitted CPU budget for local smoke')
        with patch.object(DatasetFingerprintExtractor,'__init__',small_sample),patch('task_resources.probe',return_value=observed):
            try:prepare('ct_holdG',2)
            except BaseException:
                for path in (work/'00_admin/preparation_logs').rglob('*.log'):print(path.name,path.read_text(encoding='utf-8')[-3500:])
                raise
        with patch('server_prepare.export_fold', side_effect=AssertionError('completed cache must be reused')):
            prepare('ct_holdG',2)
        prep=work/'04_data/derived/nnUNet_preprocessed/Dataset813_ct_holdG'
        assert read_json(prep/'fingerprint_provenance.json')['fit_ids']==selected['train']
        from evaluation_phases import preprocess_case
        result=preprocess_case(work,dict(fold='ct_holdG',method='B0'),selected['train'][0],2)
        data=np.load(result['array'],allow_pickle=False)
        assert data.ndim==4 and np.isfinite(data).all()
        Path(result['array']).unlink();Path(result['properties']).unlink()
        from assessment_prepare import prepare as assess
        try:
            with patch('task_resources.probe',return_value=observed):scope=assess(work,'ct_holdG',2)
        except BaseException:
            for path in (work/'00_admin/preparation_logs').rglob('*.log'):print(path.name,path.read_text(encoding='utf-8')[-3500:])
            raise
        assert len(scope['train_ids'])==2 and len(scope['validation_ids'])==1
        observations=list((work/'00_admin/case-observations').glob('*.json'))
        assert len(observations)==38
        assert all(read_json(p)['blosc_calls'] for p in observations)
        import torch
        assert not torch.cuda.is_initialized()
        print('PASS: real official synthetic fit-only fingerprint30, planning, preprocessing38, reuse, CPU prediction-prep and assessment subset3; target excluded, actual Blosc evidence and no CUDA.',flush=True)

if __name__=='__main__':main()
