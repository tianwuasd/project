"""Research-environment assertions and a real batch/Trainer optimizer smoke."""
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataset1282_analysis.contracts import atomic_json,read_json


def main(left,right,report):
    torch.set_num_threads(1)
    torch.manual_seed(37)
    np.random.seed(37)
    start=time.monotonic()
    for key,sub in [('nnUNet_raw','raw'),('nnUNet_preprocessed','preprocessed'),('nnUNet_results','results')]:
        os.environ[key]=str(right/sub)
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from dataset1282_analysis.training.trainer import nnUNetTrainerMarginal
    from dataset1282_analysis.training.marginal import MarginalBatchLoss
    dataset='Dataset282_CHDSource'
    dirs=[p/'preprocessed'/dataset for p in (left,right)]
    fits=[read_json(p/'fit.json') for p in (left,right)]
    fingerprints=[read_json(p/'dataset_fingerprint.json') for p in dirs]
    assert fingerprints[0]==fingerprints[1],'Storage/concurrency changed the fingerprint'
    assert fits[0]['spacing_xyz']==fits[1]['spacing_xyz']
    request=read_json(right/'request.json')
    names=sorted(request['cases'])
    splits=read_json(dirs[1]/'splits_final.json')[0]
    assert set(fits[1]['fit_cases'])==set(splits['train'])
    assert not set(fits[1]['fit_cases']) & set(splits['val'])
    arrays=[nnUNetDatasetBlosc2(str(p/f['data_identifier']),names) for p,f in zip(dirs,fits)]
    max_error=0.
    for name in names:
        a,sa,_,pa=arrays[0].load_case(name)
        b,sb,_,pb=arrays[1].load_case(name)
        assert a.shape==b.shape==sa.shape==sb.shape
        for z in range(0,a.shape[1],8):
            aa,bb=a[:,z:z+8],b[:,z:z+8]
            error=float(np.max(np.abs(aa-bb)))
            max_error=max(max_error,error)
            np.testing.assert_allclose(aa,bb,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(sa[:,z:z+8],sb[:,z:z+8])
        assert pa['partial_supervision']==pb['partial_supervision']
        r=request['cases'][name]
        from dataset1282_analysis.contracts import sha256_file
        assert all(sha256_file(r[k])==r['sha256'][k] for k in ('image','label'))
    trainer=nnUNetTrainerMarginal(read_json(dirs[1]/'nnUNetPlans.json'),'3d_fullres',0,
                                  read_json(dirs[1]/'dataset.json'),device=torch.device('cpu'))
    assert trainer.do_split()==(splits['train'],splits['val'])
    # Small technical network isolates adapter correctness; it is not a trained research model.
    trainer.network=torch.nn.Sequential(torch.nn.Conv3d(1,8,3,padding=1),torch.nn.LeakyReLU(),torch.nn.Conv3d(8,11,1))
    trainer.loss=MarginalBatchLoss()
    trainer.optimizer=torch.optim.SGD(trainer.network.parameters(),lr=.01)
    trainer.grad_scaler=None
    trainer.enable_deep_supervision=False
    used=[]
    losses=[]
    for role in ('train','val'):
        for name in splits[role]:
            ds=nnUNetDatasetBlosc2(str(dirs[1]/fits[1]['data_identifier']),[name])
            loader=nnUNetDataLoader(ds,1,(16,16,16),(16,16,16),trainer.label_manager,oversample_foreground_percent=1.)
            batch=loader.generate_train_batch()
            assert list(batch['keys'])==[name]
            before=[p.detach().clone() for p in trainer.network.parameters()]
            value=trainer.train_step(batch) if role=='train' else trainer.validation_step(batch)
            assert np.isfinite(value['loss'])
            if role=='train':
                assert any(not torch.equal(a,b) for a,b in zip(before,trainer.network.parameters()))
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.network.parameters())
            else:
                unknown=trainer.partial_metadata[name]['unknown_classes']
                assert all(value['tp_hard'][c-1]+value['fp_hard'][c-1]+value['fn_hard'][c-1]==0 for c in unknown)
            used.append(dict(case=name,role=role,unknown=trainer.partial_metadata[name]['unknown_classes']))
            losses.append(float(value['loss']))
    assert any(x['unknown'] for x in used),'Short-test selection needs a partial-label case'
    atomic_json(report,dict(passed=True,compared_cases=len(names),max_absolute_CT_error=max_error,
                labels_identical=True,fingerprint_identical=True,source_hashes_unchanged=True,
                training_adapter_steps=sum(x['role']=='train' for x in used),validation_steps=len(splits['val']),
                smoke_network='two-layer Conv3d, 16-cube patches; not formal nnU-Net training',
                cases=used,losses=losses,seconds=time.monotonic()-start))


if __name__=='__main__': main(*map(Path,sys.argv[1:4]))
