"""Template adapters exercised on actual geometry and immutable local artifacts."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


class SkillAdaptationTests(unittest.TestCase):
    def test_assessment_GPU_entry_preprocesses_the_author_spacing_copy(self):
        import nibabel as nib
        import numpy as np
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIO
        from assessment_predict import predict
        from server_data import write_json,digest
        class SeenInput(Exception):pass
        reader=NibabelIO()
        trainer=Mock()
        trainer.plans_manager=SimpleNamespace(image_reader_writer_class=lambda:reader)
        trainer.dataset_json={};trainer.configuration_manager=object()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work';training=root/'training'
            image=root/'input.nii.gz';nib.save(nib.Nifti1Image(np.ones((4,5,6),dtype=np.float32),np.eye(4)),image)
            sha=digest(image)
            write_json(work/'00_admin/assessment.json',dict(job_kind='assessment',scope=dict(prediction_ids=['x'])))
            write_json(work/'04_data/manifests/case_inventory.json',dict(cases=[dict(case_id='x',image=str(image),image_sha256=sha,spacing_mm=[2.,3.,4.])]))
            for name in ['plans.json','dataset.json']:write_json(training/name,{})
            def inspect_input(images,labels,properties,*args,**kwargs):
                self.assertEqual(list(properties[0]['spacing']),[4.,3.,2.])
                raise SeenInput()
            with patch('model_catalog.trainer_class',return_value=Mock(return_value=trainer)),\
                 patch('nnunetv2.inference.predict_from_raw_data.nnUNetPredictor'),\
                 patch('nnunetv2.inference.data_iterators.PreprocessAdapterFromNpy',side_effect=inspect_input),\
                 self.assertRaises(SeenInput):
                predict(work,'imagechd_fold0','S1',training,'x','gpu')
            self.assertEqual(digest(image),sha)
            self.assertTrue((work/'04_data/derived/imagechd_inference/x/geometry.json').is_file())

    def test_assessment_CPU_tail_uses_author_spacing_and_keeps_original_grid(self):
        import nibabel as nib
        import numpy as np
        import torch
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIO
        from assessment_predict import predict
        from imagechd_geometry import save_derived
        from server_data import write_json,digest,read_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work';training=root/'training';case='synthetic'
            data=np.zeros((4,5,6),dtype=np.uint8);data[1,2,3]=1
            original=root/'image.nii.gz';label=root/'label.nii.gz';derived=root/'author.nii.gz'
            affine=np.eye(4);affine[:3,3]=[11,22,33]
            nib.save(nib.Nifti1Image(data,affine),original);nib.save(nib.Nifti1Image(data,affine),label)
            sha=digest(original);spacing=[2.,3.,4.]
            save_derived(original,derived,spacing)
            write_json(work/'00_admin/assessment.json',dict(job_kind='assessment',scope=dict(prediction_ids=[case])))
            write_json(work/'04_data/manifests/case_inventory.json',dict(cases=[dict(case_id=case,
                image=str(original),label=str(label),image_sha256=sha,spacing_mm=spacing)]))
            checkpoint=training/'fold_0/checkpoint_latest.pth';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'synthetic checkpoint')
            plans=dict(image_reader_writer='NibabelIO',transpose_forward=[0,1,2],transpose_backward=[0,1,2],
                configurations={'3d_fullres':dict(architecture={},spacing=list(reversed(spacing)),
                    resampling_fn_probabilities='resample_data_or_seg_to_shape',resampling_fn_probabilities_kwargs=dict(is_seg=False,order=1,order_z=0,force_separate_z=None))})
            dataset=dict(channel_names={'0':'IMAGECHD_GRAY'},labels={'background':0,'LV':1,'RV':2,'LA':3,'RA':4,'MYO':5,'AO':6,'PA':7,'ignore':8})
            write_json(training/'plans.json',plans);write_json(training/'dataset.json',dataset)
            reader=NibabelIO();images,properties=reader.read_images([str(derived)])
            shape=list(images.shape[1:]);properties.update(shape_before_cropping=shape,
                shape_after_cropping_and_before_resampling=shape,bbox_used_for_cropping=[[0,s] for s in shape])
            predicted=np.zeros_like(data);predicted[2,2,3]=1
            reader_seg=predicted.transpose(2,1,0)
            logits=np.full((8,*reader_seg.shape),-10.,dtype=np.float32)
            for c in range(8):logits[c,reader_seg==c]=10
            base=work/'08_results/assessment/imagechd_fold0_S0';base.mkdir(parents=True)
            array=base/(case+'.logits.npy');props=base/(case+'.properties.pkl');np.save(array,logits)
            with props.open('wb') as stream:pickle.dump(properties,stream)
            write_json(base/(case+'.gpu.json'),dict(case_id=case,checkpoint_sha256=digest(checkpoint),
                logits_sha256=digest(array),properties_sha256=digest(props),seconds=.1))
            with patch.dict(os.environ,{'WHOLE_HEART_CPU_THREADS':'1'}):
                result=predict(work,'imagechd_fold0','S0',training,case,phase='cpu')
            self.assertEqual(result['status'],'passed')
            saved=nib.load(result['prediction'])
            np.testing.assert_array_equal(np.asanyarray(saved.dataobj),predicted)
            np.testing.assert_allclose(saved.affine,affine)
            self.assertEqual(result['metrics']['classes']['LV']['hd95_mm'],2.)
            self.assertEqual(digest(original),sha)
            self.assertFalse(torch.cuda.is_initialized())

    def artifact_fixture(self,root):
        from server_data import write_json,digest
        from v2_host_identity import claim
        runtime=root/'runtime';results=root/'results';claim(runtime,results)
        job=results/'job';work=job/'workspace'
        plan=dict(directory=str(job),workspace=str(work),runtime=str(runtime),name='synthetic',tasks=['train'])
        write_json(job/'plan.json',plan)
        write_json(job/'state.json',dict(status='completed',plan_sha256=digest(job/'plan.json'),steps={'train':dict(status='completed')}))
        write_json(work/'09_reports/summary.json',dict(synthetic=True))
        return job,runtime

    def test_artifact_manifest_is_ImageCHD_STU_and_rejects_VISTA_identity(self):
        from panel_publication import upload
        from server_data import read_json,write_json
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'VISTA_CATALOG_VERSION':'1'}):
            job,runtime=self.artifact_fixture(Path(d))
            with redirect_stdout(io.StringIO()):package=upload.prepare_upload(job,runtime)
            _,manifest,_=upload._validate_package(package)
            self.assertEqual(manifest.get('project_id'),'chd-ct-panel-v2/stunet')
            self.assertEqual(manifest['release_tag'],'chd-v2-stunet-artifacts-'+manifest['package_id'])
            manifest['project_id']='vista3d'
            write_json(package/'manifest.json',manifest)
            with self.assertRaises(ValueError):upload._validate_package(package)

    def test_artifact_rejects_a_different_runtime_or_job_root(self):
        from panel_publication import upload
        from v2_host_identity import claim
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job,runtime=self.artifact_fixture(root)
            foreign=root/'other';claim(foreign,root/'other-results')
            with self.assertRaises(ValueError):upload._finished(job,foreign)

    def test_artifact_verified_retry_keeps_same_package_and_CHD_branch(self):
        from panel_publication import upload,ssh_upload
        from server_data import digest,read_json
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'VISTA_CATALOG_VERSION':'1'}):
            job,runtime=self.artifact_fixture(Path(d))
            with redirect_stdout(io.StringIO()):package=upload.prepare_upload(job,runtime)
            package,manifest,assets=upload._validate_package(package)
            before={p.name:digest(p) for p in package.iterdir() if p.is_file()}
            commit='a'*40
            with patch.object(ssh_upload,'require_ssh'),patch.object(ssh_upload,'build_commit',return_value=(Path(d)/'repo',commit)),\
                 patch.object(ssh_upload,'verify_tree'),patch.object(ssh_upload,'remote_head',return_value=commit) as remote,\
                 patch.object(ssh_upload,'git',side_effect=AssertionError('already-published retry must not push')):
                url=ssh_upload.publish(package,manifest,assets)
            self.assertIn('/chd-v2-stunet-artifacts/',url)
            self.assertTrue(all(c.args[1].startswith('refs/heads/chd-v2-stunet-artifacts/') for c in remote.call_args_list))
            self.assertEqual(before,{name:digest(package/name) for name in before})
            self.assertEqual(commit,read_json(package/'receipt.json')['commit'])


if __name__=='__main__':unittest.main()
