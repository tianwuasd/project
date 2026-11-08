from dataclasses import asdict
from pathlib import Path
import os
import shutil
import tempfile
import unittest

import blosc2
import nibabel as nib
import numpy as np
import torch

from dataset1282_analysis.contracts import atomic_json, digest, read_json, sha256_file
from dataset1282_analysis.prediction.heldout import HeldoutCache
from dataset1282_analysis.prediction.workflow import BloodInputs, _branches, branch_record, case_record
from dataset1282_analysis.prediction.source import implementation
from dataset1282_analysis.prediction import PROTOCOL
from dataset1282_analysis.xu_fusion import grid_prediction, vote, _save_case
from dataset1282_analysis.xu_geometry import PreparedCache
from dataset1282_analysis.xu_targets import anatomy
from dataset1282_analysis.xu_training import train_branch
from dataset1282_analysis.xu_workflow import input_id, side
from test_xu_targets import on_disk_phantom


class PredictionForwardTests(unittest.TestCase):
    def test_real_models_label_free_case_resume_and_native_export(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch).resolve()
            existing = os.environ.get('DATASET1282_PREDICTION_TEST_MODELS')
            prior = Path(existing).resolve() if existing else None
            training = (PreparedCache(prior/'training-input',allow_technical_subset=True)
                        if prior else on_disk_phantom(base/'training-input'))
            out = base/'prediction'
            geo = training.geometry('train_a')
            # 合成测试输入只有 CT；固定小框用于验证分支与几何接口。
            folder = out/'prepared/held'
            folder.mkdir(parents=True)
            array = blosc2.asarray(training.read_image('dev_b')[None],urlpath=str(folder/'image.b2nd'),mode='w')
            del array
            atomic_json(folder/'geometry.json',asdict(geo))
            job = dict(cases={'held':dict(role='target_primary',sha256={'image':'synthetic-ct'})})
            atomic_json(out/'run.json',job)
            atomic_json(folder/'complete.json',dict(job=digest(job),case='held',source_sha256='synthetic-ct',
                        artifacts={f:sha256_file(folder/f) for f in ('image.b2nd','geometry.json')}))
            images = HeldoutCache(out)
            (out/'source').mkdir()
            box = [(2,18)]*3
            book = dict(boxes={n:box for r in ('train','development') for n in training.case_ids(r)})
            source = dict(run=dict(profile='smoke',supervision='marginal'),boxes=book)
            train_cases = {n:[] for n in training.case_ids('train')}
            dev_cases = {n:[] for n in training.case_ids('development')}
            for branch in ('crop64','crop128','all64','all128'):
                roi = None if branch.startswith('crop') else box
                def load(name,key):
                    return training.read_image(name,(side(key),)*3,roi),training.read_label(name,(side(key),)*3,roi)
                if prior:
                    # 复用已完成短测的权重，加载器继续核对原始训练身份。
                    result = read_json(prior/branch/'training.json')
                    self.assertEqual(result['state'],'complete')
                    self.assertGreater(result['best_epoch'],0)
                    checkpoint = prior/branch/'best.pt'
                else:
                    result = train_branch(branch=branch,output_dir=base/branch,train_cases=train_cases,
                        validation_cases=dev_cases,load_sample=load,profile='smoke',supervision='marginal',
                        data_identity=input_id(training,'marginal',None if roi is None else book))
                    checkpoint = Path(result['best_checkpoint'])
                shutil.copyfile(checkpoint,out/'source'/(branch+'.pt'))
            reference = base/'reference.nii.gz'
            hidden_reference = base/'sealed-reference.nii.gz'
            nib.save(nib.Nifti1Image(training.read_native_label('dev_b').astype(np.float32),np.eye(4)),reference)
            job['cases']['held'].update(label=str(reference),known_absent=[],device='iCT 256')
            job['cases']['held']['sha256']['label'] = sha256_file(reference)
            reference.rename(hidden_reference)
            frozen = dict(protocol=PROTOCOL,checkpoints={p.stem:sha256_file(p) for p in (out/'source').glob('*.pt')})
            atomic_json(out/'source/manifest.json',frozen)
            job.update(source=digest(frozen),implementation=implementation())
            atomic_json(out/'run.json',job)
            marker = read_json(folder/'complete.json')
            marker['job'] = digest(job)
            atomic_json(folder/'complete.json',marker)
            images = HeldoutCache(out)
            _branches(out,images,training,source,job,'cpu',('crop64','crop128'))
            boxes = {'held':dict(box=box,quality=dict(accepted=True))}
            _branches(out,images,training,source,job,'cpu',('all64','all128'),boxes)
            stamps = {p:p.stat().st_mtime_ns for p in (out/'branches').rglob('*.json')}
            _branches(out,images,training,source,job,'cpu',('crop64','crop128'))
            _branches(out,images,training,source,job,'cpu',('all64','all128'),boxes)
            self.assertEqual(stamps,{p:p.stat().st_mtime_ns for p in stamps})
            labels = {b:grid_prediction(branch_record(out,b,'held',job),geo.cache_shape,
                          None if b.startswith('crop') else box,box,(128,)*3)
                      for b in ('crop64','crop128','all64','all128')}
            small,qc = vote(labels,anatomy(training))
            (out/'cases').mkdir()
            result = _save_case(out,'held',geo,box,small,qc,images,False)
            atomic_json(out/'cases/held.json',dict(job=digest(job),case='held',role='target_primary',
                        state='predicted',roi=boxes['held'],result=result))
            self.assertEqual(case_record(out,'held',job)['state'],'predicted')
            native = nib.load(out/'cases/held.nii.gz')
            self.assertEqual(native.shape,geo.native_shape_xyz)
            np.testing.assert_allclose(native.affine,np.diag([*geo.spacing_xyz,1]))
            np.testing.assert_array_equal(np.asanyarray(native.dataobj),geo.restore_label(small,box))
            from dataset1282_analysis.prediction.evaluation import evaluate
            atomic_json(out/'complete.json',dict(job=digest(job),cases={'held':case_record(out,'held',job)}))
            hidden_reference.rename(reference)
            report = evaluate(out)
            self.assertEqual(report['groups']['target_primary']['evaluated'],1)
            self.assertEqual(report['groups']['target_primary']['roi_rejected'],0)
            selected = os.environ.get('DATASET1282_PREDICTION_TEST_TRAINING')
            if selected:
                # 使用已完成的真实短训练血池权重，输入仍为独立合成 CT。
                from dataset1282_analysis.xu_blood import load_blood_predictor
                root = Path(selected)
                original = read_json(root/'run.json')
                original_cache = PreparedCache(original['prepared'],allow_technical_subset=True)
                self.assertEqual(original_cache.scope,'technical_subset')
                predictor = load_blood_predictor(BloodInputs(original_cache,images),root/'blood/best.pt',
                                                 'cpu',original['profile'],original['supervision'])
                prediction = predictor('held',box,(16,)*3)
                self.assertEqual(prediction.shape,(16,)*3)
                self.assertEqual(prediction.dtype,np.uint8)
                self.assertLessEqual(prediction.max(),2)
            marker = out/'branches/crop64/held.json'
            altered = read_json(marker)
            altered['sha256'] = '0'*64
            atomic_json(marker,altered)
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                branch_record(out,'crop64','held',job)


if __name__ == '__main__':
    unittest.main()
