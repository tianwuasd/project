import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


class ImageCHDContractTests(unittest.TestCase):
    def test_catalog_has_true_pretraining_and_separate_random_control(self):
        from model_catalog import parse_models,trainer_class,parse_folds
        self.assertEqual(parse_models('S1'),['S1'])
        self.assertEqual(trainer_class('S1').__name__,'STUNetPretrained')
        self.assertEqual(trainer_class('S0').__name__,'STUNetRandom')
        self.assertEqual(parse_folds('ct-first'),['imagechd_fold0'])
        with self.assertRaises(ValueError):parse_models('B0')

    def test_extra_annotation_is_ignore_instead_of_background(self):
        import numpy as np
        from data_pipeline import map_labels,inverse_labels
        a=np.array([0,1,2,3,4,5,6,7,8,15,255])
        self.assertEqual(map_labels(a).tolist(),[0,1,2,3,4,5,6,7,8,8,8])
        self.assertEqual(inverse_labels(a[:8]).tolist(),a[:8].tolist())
        for bad in ([-1],[1.2],[float('nan')]):
            with self.assertRaises(ValueError):map_labels(np.array(bad))

    def test_author_spacing_copy_keeps_raw_pixels_and_source(self):
        import numpy as np
        import nibabel as nib
        from imagechd_geometry import save_derived
        from server_data import digest
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);source=p/'input.nii.gz';out=p/'copy.nii.gz'
            data=np.arange(6*7*8,dtype=np.int16).reshape(6,7,8)
            affine=np.diag([-1.,1.,1.,1.]);affine[:3,3]=[11,22,33]
            nib.save(nib.Nifti1Image(data,affine),source);sha=digest(source)
            save_derived(source,out,[.4,.6,1.2])
            result=nib.load(out)
            self.assertEqual(digest(source),sha)
            np.testing.assert_array_equal(np.asanyarray(result.dataobj),data)
            np.testing.assert_allclose(result.affine[:3,3],[11,22,33])
            np.testing.assert_allclose(np.linalg.norm(result.affine[:3,:3],axis=0),[.4,.6,1.2])

    def test_target_exactly_once_and_quarantine_never_used(self):
        from imagechd_cohort import make_splits
        cases=[]
        for i in range(6):
            roles={str(f):('test' if i//2==f else 'val' if i%2 else 'train') for f in range(3)}
            cases.append(dict(case_id=str(i),status='eligible',roles=roles))
        cases.append(dict(case_id='missing_myo',status='quarantined',roles={}))
        splits=make_splits(cases)
        self.assertEqual(sorted(x for s in splits.values() for x in s['target']),list('012345'))
        self.assertFalse(any('missing_myo' in s[k] for s in splits.values() for k in ('train','val','target')))
        cases[0]['roles']['1']='test'
        with self.assertRaises(ValueError):make_splits(cases)

    def test_missing_or_bad_weights_cannot_become_random_training(self):
        from stunet_weights import load_pretrained
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wrong.model';path.write_bytes(b'not official weights')
            with self.assertRaises(ValueError):load_pretrained(torch.nn.Linear(1,1),path)

    def test_metrics_exclude_extra_label_predictions_and_keep_empty_failure(self):
        import numpy as np
        from evaluation import score_case
        reference=np.zeros((5,5,5),dtype=np.uint8)
        reference[1,1,1]=7;reference[3,3,3]=8
        prediction=np.zeros_like(reference);prediction[3,3,3]=7
        pa=score_case(prediction,reference,np.eye(4))['classes']['PA']
        self.assertEqual(pa['dice'],0)
        self.assertEqual(pa['predicted_voxels'],0)
        self.assertEqual(pa['reference_voxels'],1)
        self.assertEqual(pa['ignored_source_voxels'],1)

    def test_summary_handles_random_control_without_pretrained_method(self):
        from server_data import write_json,digest,read_json
        from evaluation_summary import summarize
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frozen=root/'08_results/evaluation/frozen_models.json'
            write_json(frozen,dict(models=[dict(fold='imagechd_fold0',method='S0',targets=['synthetic'])]))
            output=root/'synthetic_output';output.write_bytes(b'synthetic')
            score=dict(macro_dice=0.,defined_classes=7,both_empty_classes=0,one_empty_classes=7,
                classes={name:dict(dice=0.,hd_mm='inf',hd95_mm='inf',status='one_empty',
                    predicted_voxels=0,reference_voxels=1,ignored_source_voxels=0)
                    for name in ['LV','RV','LA','RA','MYO','AO','PA']})
            write_json(root/'08_results/evaluation/imagechd_fold0_S0/synthetic.json',dict(status='completed',
                modality='CT',signature=dict(frozen_sha256=digest(frozen)),
                outputs={v:dict(path=str(output),sha256=digest(output)) for v in ['raw','lcc']},
                results={v:score for v in ['raw','lcc']}))
            summarize(root)
            result=read_json(root/'09_reports/final_evaluation/summary.json')
            self.assertEqual(result['results']['CT_raw']['methods']['S0']['case_mean_dice'],0.)
            self.assertEqual(result['classes']['S0/raw/PA']['empty_prediction'],['synthetic'])


if __name__=='__main__':unittest.main()
