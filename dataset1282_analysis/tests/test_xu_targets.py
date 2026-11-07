import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from dataset1282_analysis.contracts import atomic_json, digest, read_json, sha256_file
from dataset1282_analysis.xu_geometry import CaseGeometry, PreparedCache, _resize
from dataset1282_analysis.xu_targets import (SEMANTICS, TargetStore, anatomy, blood_eligible,
                                             blood_target, build_targets, cache_coordinates,
                                             proximal_target, review_views)


MAPPING = dict(LV=7, RV=2, LA=3, RA=4, MYO=1, AO=5, PA=6, SVC=8, IVC=9, PV=10)


class Fixture:
    def __init__(self,root):
        self.run_dir=root
        self.request_digest='synthetic-request'
        self.geo=CaseGeometry((20,20,20),(1.,1.,1.),(0,1,2),((0,20),)*3,(20,20,20))
        self.seg=np.zeros((20,20,20),dtype=np.int8)
        self.seg[0]= -1
        self.seg[2:18,4:10,4:9]=MAPPING['AO']
        self.seg[2:18,4:10,12:16]=MAPPING['PA']
        for i, key in enumerate(('LV','RV','LA','RA','MYO','PV','IVC','SVC')):
            self.seg[3:7,12+i%4,2+i]=MAPPING[key]
        self.supervision={name:dict(present_classes=list(range(1,11)),unknown_classes=[],known_absent_classes=[])
                          for name in ('train_a','dev_b')}
        labels={SEMANTICS[k]:v for k,v in MAPPING.items()}
        atomic_json(root/'preprocessed/Dataset282_CHDSource/dataset.json',dict(labels=dict(background=0,**labels)))

    def case_ids(self,role,complete_only=False):
        return ('train_a',) if role=='train' else ('dev_b',)

    def geometry(self,name):
        return self.geo

    def read_label(self,name):
        return self.seg

    def read_image(self,name,shape,box):
        return _resize(self.seg[tuple(slice(a,b) for a,b in box)].astype(np.float32),shape,1)


def on_disk_phantom(root):
    """Two synthetic scans with complete stage, role and process manifests."""
    import blosc2
    import nibabel as nib
    fixture=Fixture(root)
    names=('train_a','dev_b')
    roles=dict(zip(names,('train','development')))
    dataset=root/'preprocessed/Dataset282_CHDSource'
    records={n:dict(role=r,known_absent=[],shape=[20]*3,spacing=[1.]*3) for n,r in roles.items()}
    cohort=dict(cases={n:dict(role=r,patient=n,included=True,index_scan=True) for n,r in roles.items()})
    request=dict(scope='technical_subset',method='HU-marginal-v1',nnunetv2='2.8.1',staged_extension='.nii',
                 cases=records,cohort_digest=digest(cohort))
    req_id=digest(request)
    plans=dict(image_reader_writer='NibabelIOWithReorient',transpose_forward=[0,1,2],
               transpose_backward=[0,1,2],configurations={'3d_fullres':dict(data_identifier='cached')})
    for n in names: fixture.supervision[n]['ambiguous_background_allowed']=[0]
    for filename,value in [('partial_supervision',fixture.supervision),('splits_final',[{'train':[names[0]],'val':[names[1]]}]),
                           ('nnUNetPlans',plans)]:
        atomic_json(dataset/(filename+'.json'),value)
    files={x.name:sha256_file(x) for x in dataset.glob('*.json')}
    fit=dict(data_identifier='cached',fit_cases=[names[0]],plans_sha256=files['nnUNetPlans.json'])
    for filename,value in [('fit',fit),('cohort',cohort),('request',request)]:
        atomic_json(root/(filename+'.json'),value)
    def commit(job, paths):
        atomic_json(root/'commits'/(job+'.json'),dict(contract=req_id,job=dict(id=job),
                    artifacts={x.relative_to(root).as_posix():sha256_file(x) for x in paths}))
    commit('fit',[root/'fit.json'])
    atomic_json(dataset/'training_contract.json',dict(request_digest=req_id,artifacts=files))
    atomic_json(root/'complete.json',dict(request_digest=req_id,state='completed',scope='technical_subset',
                fit=fit,coordinate_frame='local_array_excel_mm',train_cases=1,development_cases=1))
    (dataset/'cached').mkdir()
    for i,n in enumerate(names):
        meta=root/'staged'/n/'metadata.json'
        atomic_json(meta,dict(case=n,source_spacing_xyz=[1.]*3,working_affine=np.eye(4).tolist()))
        label=meta.parent/'label.nii'
        nib.save(nib.Nifti1Image(fixture.seg.clip(0).T.astype('uint8'),np.eye(4)),label)
        commit('stage-'+n,[meta,label])
        qc=root/'qc'/(n+'.json')
        atomic_json(qc,dict(prepared_shape=[1,20,20,20],supervision=fixture.supervision[n],
                            bbox_transposed=[[0,20]]*3))
        ct=dataset/'cached'/(n+'.b2nd')
        sg=dataset/'cached'/(n+'_seg.b2nd')
        noise=np.random.default_rng(i).normal(size=fixture.seg.shape)
        blosc2.asarray((fixture.seg+noise*.05)[None].astype('float32'),urlpath=str(ct),mode='w')
        blosc2.asarray(fixture.seg[None],urlpath=str(sg),mode='w')
        commit('process-'+n,[qc,ct,sg])
    return PreparedCache(root,allow_technical_subset=True)


def certified_phantom_roi(cache, path):
    from dataset1282_analysis.xu_roi_workflow import audit, localize
    args=SimpleNamespace(profile='smoke',supervision='complete',device='cpu',crop64='one',crop128='two',
                         padding_percent=0)
    def locator(image):
        mask=np.zeros(image.shape,np.uint8)
        mask[tuple(slice(n//16, n-n//16) for n in mask.shape)]=1
        return mask
    with patch('dataset1282_analysis.xu_workflow.frozen_predictor',return_value=(locator,'a'*64)):
        localize(cache,args,path)
    args.candidates=path/'candidates.json'
    audit(cache,args,path)
    return path/'boxes.json'


def review():
    def vessel(x):
        return dict(start=dict(point_xyz_mm=[x,6,3],normal_xyz=[0,0,1]),
                    end=dict(point_xyz_mm=[x,6,10],normal_xyz=[0,0,1]),seed_xyz_mm=[x,6,7])
    return dict(approved=True,review_id='synthetic-rater',AO=vessel(6),PA=vessel(13))


class AuxiliaryTests(unittest.TestCase):
    def test_physical_band_and_unknown_edges(self):
        seg=np.full((7,7),5,np.int8)
        seg[0,:]=0
        seg[:,0]=0
        seg[3:,6]=-1
        y=blood_target(seg,(2.,.5),[5],.75)
        self.assertEqual(y[1,4],1)  # 2 mm to observed background in axis 0
        self.assertEqual(y[4,1],2)  # 0.5 mm in axis 1
        self.assertEqual(y[4,5],-1)  # unknown tissue, not a manufactured wall
        self.assertEqual(y[4,6],-1)
        self.assertTrue(blood_eligible(dict(present_classes=[5],known_absent_classes=[6],unknown_classes=[]),[5,6]))
        self.assertFalse(blood_eligible(dict(present_classes=[5],known_absent_classes=[],unknown_classes=[6]),[5,6]))

    def test_reviewed_planes_use_local_mm_and_component_seed(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Fixture(Path(d))
            self.assertEqual(anatomy(cache),MAPPING)
            cache.seg[2:18,15:17,3:5]=MAPPING['AO']  # disconnected
            y=proximal_target(cache.seg,cache.geo,MAPPING,review())
            self.assertEqual(y[5,6,6],6)
            self.assertEqual(y[5,6,13],6)
            self.assertEqual(y[12,6,6],0)
            self.assertEqual(y[5,16,4],0)
            self.assertEqual(y[0,5,4],-1)
            self.assertTrue(np.all(y[cache.seg==MAPPING['MYO']]==5))
            bad=review()
            bad['AO']['seed_xyz_mm']=[6,6,18]
            with self.assertRaises(ValueError):
                proximal_target(cache.seg,cache.geo,MAPPING,bad)
            g=CaseGeometry((15,20,25),(0.5,1.,2.),(2,0,1),((1,5),(0,10),(2,6)),(8,10,2))
            axes,origin,step=cache_coordinates(g)
            self.assertEqual(axes,[0,2,1])
            np.testing.assert_allclose(step,[.25,2.,2.])
            np.testing.assert_allclose(origin,[.375,0,2.5])

    def test_missing_planes_recorded_and_resume_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch('dataset1282_analysis.xu_workflow.artifact_identity',return_value='fixture'):
            base=Path(d)
            cache=Fixture(base/'prepared')
            args=SimpleNamespace(width_mm=1.,reviews=None,resume=False)
            out=base/'generated'
            report=build_targets(cache,args,out)
            self.assertEqual(report['available'],dict(blood=2,init=0))
            store=TargetStore(out,cache)
            value=store.read('dev_b','blood',((0,2),(0,3),(0,4)))
            self.assertEqual(value.shape,(2,3,4))
            with self.assertRaises(ValueError): store.read('dev_b','init')
            args.resume=True
            self.assertEqual(build_targets(cache,args,out)['available'],report['available'])
            with (out/'cases/train_a/blood.b2nd').open('ab') as f:
                f.write(b'damaged')
            with self.assertRaisesRegex(ValueError,'differs'):
                build_targets(cache,args,out)

    def test_two_init_network_sizes_and_checkpoint_selection(self):
        import torch
        from dataset1282_analysis.xu_init import _model, load_init_predictor, train_init
        old=torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,old)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            cache=on_disk_phantom(base/'prepared')
            view=base/'views'
            view.mkdir()
            result=review_views(cache,SimpleNamespace(case='dev_b',center_xyz_mm=[6.,6.,7.]),view)
            self.assertTrue(Path(result['figure']).is_file())
            template=read_json(view/'review-template.json')
            self.assertFalse(template['cases']['dev_b']['approved'])
            self.assertIsNone(template['cases']['dev_b']['AO']['start']['point_xyz_mm'])
            reviews=base/'review.json'
            atomic_json(reviews,dict(request_digest=cache.request_digest,coordinate_frame='local_array_excel_mm',
                                     cases={name:review() for name in cache.supervision}))
            out=base/'targets'
            build_targets(cache,SimpleNamespace(width_mm=1.,reviews=reviews,resume=False),out)
            book_path=certified_phantom_roi(cache,base/'roi')
            book=read_json(book_path)
            for branch in ('init64','init128'):
                directory=base/branch
                directory.mkdir()
                args=SimpleNamespace(targets=out,boxes=book_path,seed=2,branch=branch,
                                     profile='smoke',supervision='complete',device='cpu',resume=False)
                summary=train_init(cache,args,directory)
                self.assertEqual(summary['best_epoch'],1)
                args.resume=True
                self.assertEqual(train_init(cache,args,directory)['history'],summary['history'])
                side=int(branch[4:])
                pred=load_init_predictor(directory/'best.pt',cache=cache,branch=branch,profile='smoke',
                                         supervision='complete',book=book)
                y=pred(np.zeros((side,)*3,np.float32))
                self.assertEqual(y.shape,(side,)*3)
                self.assertLessEqual(y.max(),6)
                with self.assertRaisesRegex(ValueError,'Incompatible'):
                    load_init_predictor(directory/'best.pt',cache=cache,branch=branch,profile='smoke',
                                        supervision='marginal',book=book)
            with torch.device('meta'):
                paper=_model('init128','paper')
                self.assertEqual(paper.head.out_channels,7)
                self.assertEqual(paper.spec.width,48)
