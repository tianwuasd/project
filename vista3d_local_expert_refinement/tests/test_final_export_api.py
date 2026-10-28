"""Real synthetic source export, pixel/hash/split and partial-commit parity."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('nibabel'),'requires actual NIfTI CPU runtime')
class ParallelSourceExport(unittest.TestCase):
    def test_parallel_export_preserves_pixels_hashes_split_and_partial_commits(self):
        import numpy as np
        import nibabel as nib
        import cpu_case_queue
        from cohort import make_splits
        from test_cohort import cases as inventory
        from server_data import write_json,read_json,digest
        from data_pipeline import export_fold,inverse_labels,OFFICIAL
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir();rows=inventory();split=make_splits(rows)['ct_holdG']
            ids=split['train']+split['val'];raw=np.resize(OFFICIAL,(6,7,8));affine=np.diag([1.,1.25,1.5,1.])
            for row in rows:
                cid=row['case_id'];row.update(image=str(source/(cid+'_image.nii.gz')),label=str(source/(cid+'_label.nii.gz')))
                if cid in ids:
                    nib.save(nib.Nifti1Image(np.arange(raw.size,dtype=np.float32).reshape(raw.shape)+int(cid[4:]),affine),row['image'])
                    nib.save(nib.Nifti1Image(raw,affine),row['label']);row['image_sha256']=digest(row['image'])
                else:Path(row['label']).write_bytes(b'FORBIDDEN')
            work=root/'work'
            def manifests(w):
                write_json(w/'04_data/manifests/case_inventory.json',dict(cases=rows))
                write_json(w/'04_data/manifests/splits_v1.json',dict(folds={'ct_holdG':split}))
            manifests(work);started=[];overlap=[];original=cpu_case_queue.subprocess.Popen
            def start(*args,**kwargs):
                process=original(*args,**kwargs);started.append(process)
                overlap.append(sum(p.poll() is None for p in started));return process
            facts=dict(cpu_budget=4,memory_budget_bytes=20*2**30,io_pressure={'avg10':40})
            with patch('task_resources.probe',return_value=facts),patch.object(cpu_case_queue.subprocess,'Popen',side_effect=start):
                output=export_fold(work,'ct_holdG',split,rows)
            self.assertGreater(max(overlap),1);self.assertLessEqual(max(overlap),4)
            self.assertEqual(read_json(output/'source_only_split.json'),split)
            self.assertEqual(read_json(output/'dataset.json')['numTraining'],38)
            self.assertEqual([r['case_id'] for r in read_json(work/'09_reports/ct_holdG_conversion.json')],ids)
            for cid in ids:
                row=next(r for r in rows if r['case_id']==cid)
                self.assertEqual(digest(output/'imagesTr'/f'{cid}_0000.nii.gz'),row['image_sha256'])
                label=nib.load(output/'labelsTr'/f'{cid}.nii.gz')
                np.testing.assert_array_equal(inverse_labels(np.asanyarray(label.dataobj)),raw)
                np.testing.assert_array_equal(label.affine,affine)
            # Failure at a later source preserves complete early cases, never commits dataset/report.
            bad=next(r for r in rows if r['case_id']==ids[-1]);Path(bad['label']).write_bytes(b'invalid label')
            failed=root/'failed';manifests(failed)
            with patch('task_resources.probe',return_value=facts),self.assertRaises(RuntimeError):export_fold(failed,'ct_holdG',split,rows)
            partial=failed/'04_data/derived/nnUNet_raw/Dataset813_ct_holdG'
            self.assertTrue(list((partial/'export_case_records').glob('*.json')))
            self.assertFalse((partial/'dataset.json').exists());self.assertFalse((failed/'09_reports/ct_holdG_conversion.json').exists())

if __name__=='__main__':unittest.main()
