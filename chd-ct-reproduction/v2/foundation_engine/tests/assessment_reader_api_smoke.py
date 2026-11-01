"""Actual pinned NibabelIO asymmetric voxel/affine roundtrip, CPU only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-file');args=parser.parse_args()
    if importlib.metadata.version('nnunetv2')!='2.8.1':raise ValueError('Pinned nnU-Net2.8.1 required')
    import numpy as np
    import nibabel as nib
    import torch
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIO
    from assessment_predict import restore_reader_grid
    with tempfile.TemporaryDirectory(prefix='vista_assessment_reader_') as temp:
        work=Path(temp);os.environ['WHOLE_HEART_WORKSPACE']=str(work)
        from evaluation import save_prediction,score_case
        from data_pipeline import map_labels
        data=np.zeros((3,4,5),dtype=np.uint8);data[1,2,4]=2;data[2,1,0]=3
        affine=np.array([[-1.25,0,0,12],[0,2.5,0,20],[0,0,3.75,-4],[0,0,0,1]],dtype=float)
        image=work/'synthetic.nii.gz';nib.save(nib.Nifti1Image(data,affine),image)
        original=nib.load(image);reader=NibabelIO();images,properties=reader.read_images([str(image)])
        assert list(images.shape)==[1,5,4,3]
        prediction=restore_reader_grid(reader,images[0].astype(np.uint8),properties,original,work/'prediction.nii.gz')
        assert np.array_equal(prediction,data)
        assert prediction[1,2,4]==2 and prediction[2,1,0]==3
        save_prediction(work/'prediction.nii.gz',prediction,original)
        saved=nib.load(work/'prediction.nii.gz')
        assert saved.shape==(3,4,5) and np.allclose(saved.affine,affine,atol=1e-4,rtol=0)
        assert np.array_equal(map_labels(np.asarray(saved.dataobj)),data)
        assert score_case(prediction,data,affine)['macro_dice']==1.
        assert not list(work.glob('*.axes.tmp.nii.gz'))
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',reader='actual nnunetv2 2.8.1 NibabelIO',original_shape=[3,4,5],
        reader_shape=[5,4,3],asymmetric_voxel_positions_equal=True,affine_equal=True,
        official_label_write_reload_equal=True,macro_dice=1.,cuda_initialized=False,real_ct=False)
    if args.report_file:Path(args.report_file).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
