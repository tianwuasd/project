"""Real nnU-Net Blosc arrays and full-affine SciPy metrics, CPU only."""
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))

def main():
    import numpy as np
    import torch
    import blosc2
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    from case_execution import leaf_threads
    from metrics import binary_metrics
    rng=np.random.default_rng(42);data=rng.normal(size=(1,9,12,15)).astype(np.float32)
    seg=rng.integers(0,8,size=data.shape,dtype=np.int16);original=blosc2.asarray;records=[]
    with tempfile.TemporaryDirectory() as temp:
        for threads in [1,3,7]:
            path=str(Path(temp)/f'case{threads}')
            with leaf_threads(threads) as evidence:
                nnUNetDatasetBlosc2.save_case(data,seg,{'test':True},path,chunks=(1,9,12,15),blocks=(1,3,4,5))
            assert original is blosc2.asarray
            assert len(evidence['blosc_calls'])==6
            assert all(row['nthreads']==threads for row in evidence['blosc_calls'])
            for ending,value in [('.b2nd',data),('_seg.b2nd',seg)]:
                restored=blosc2.open(path+ending,mode='r')
                assert np.array_equal(restored[:],value)
                records.append(dict(threads=threads,ending=ending,codec=str(restored.cparams.codec),
                    clevel=restored.cparams.clevel,filters=str(restored.cparams.filters),chunks=list(restored.chunks),blocks=list(restored.blocks)))
        for ending in ['.b2nd','_seg.b2nd']:
            rows=[{k:v for k,v in r.items() if k!='threads'} for r in records if r['ending']==ending]
            assert rows[0]==rows[1]==rows[2]
    affine=np.array([[1.2,.2,0,3],[.1,0,-2,4],[0,3,.1,-1],[0,0,0,1]])
    for pred,ref in [(seg[0]==2,seg[0]==3),(seg[0]==9,seg[0]==3),(seg[0]==9,seg[0]==9)]:
        assert binary_metrics(pred,ref,affine,1)==binary_metrics(pred,ref,affine,3)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='passed',compression=records,metric_threads=[1,3],full_affine=True,cuda_initialized=False),indent=2))

if __name__=='__main__':main()
