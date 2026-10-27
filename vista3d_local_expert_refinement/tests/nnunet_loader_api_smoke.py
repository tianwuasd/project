"""Actual pinned loader/augmentation API on tiny synthetic CPU cases; no CUDA/CT."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import inspect

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'05_src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-file')
    args=parser.parse_args()
    import importlib.metadata
    import numpy as np
    import torch
    if importlib.metadata.version('nnunetv2')!='2.8.1':raise ValueError('Requires pinned nnU-Net2.8.1')
    torch.set_num_threads(1)
    from batch_pool import EpochBatchIterator
    from nnunet_batches import NNUNetBatchSource
    checks=[]
    with tempfile.TemporaryDirectory(prefix='vista_real_loader_api_') as tmp:
        root=Path(tmp)
        for name in ['nnUNet_raw','nnUNet_preprocessed','nnUNet_results']:os.environ[name]=str(root/name)
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
        folder=root/'cases';folder.mkdir()
        grids=np.indices((24,24,24))
        segmentation=(grids.sum(0)%8).astype(np.int16)[None]
        data=(grids[0]*.1+grids[1]*.01+grids[2]*.001).astype(np.float32)[None]
        locations={label:np.argwhere(segmentation==label)[:100] for label in range(1,8)}
        for index in range(2):
            nnUNetDatasetNumpy.save_case(data+index,segmentation,{'class_locations':locations},str(folder/f'case{index}'))
        dataset=dict(channel_names={'0':'CT'},labels={str(i):i for i in range(8)},numTraining=2,file_ending='.nii.gz')
        dataset['labels'].pop('0');dataset['labels']['background']=0
        plans=dict(dataset_name='Dataset813_ct_holdG',plans_name='SyntheticLoaderAPI',
            configurations={'3d_fullres':dict(patch_size=[16,16,16],batch_size=2,
                use_mask_for_norm=[False],architecture={'network_class_name':
                'dynamic_network_architectures.architectures.unet.PlainConvUNet',
                'arch_kwargs':{},'_kw_requires_import':[]})})
        base=dict(plans=plans,dataset_json=dataset,configuration='3d_fullres',folder=str(folder),
            local_batch=2,oversample=.5,probabilistic_oversampling=False,rotation=[-.2,.2],
            dummy_2d=False,initial_patch=[20,20,20],mirror_axes=[0,1,2],
            deep_supervision=[[1.,1.,1.],[.5,.5,.5]],ids=['case0','case1'])
        # Observe inside the real upstream math/transform region, where its old
        # hard limit1 used to override configured outer threads.
        source=NNUNetBatchSource(dict(base,stage='train'))
        source.configure_worker_threads(2);source._build()
        original=source._loader.transforms
        inside=[]
        def observe_transform(**batch):
            from threadpoolctl import threadpool_info
            inside.append(dict(torch_threads=torch.get_num_threads(),native=threadpool_info()))
            return original(**batch)
        source._loader.transforms=observe_transform
        next(source)
        assert inside and all(row['torch_threads']==2 for row in inside),inside
        # Intel's libiompstubs reports1 permanently; it is not a compute pool.
        assert all(pool['num_threads']==2 for row in inside for pool in row['native']
                   if 'stubs' not in pool['filepath']),inside
        assert torch.get_num_threads()==1
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        assert 'threadpool_limits(limits=1' in inspect.getsource(nnUNetDataLoader.generate_train_batch)
        provenance=source.observations()['adapter_provenance']
        checks.append('actual_inner_transform_native_and_torch_threads_2_without_global_mutation')
        for stage in ['validation','train']:
            spec=dict(base,stage=stage)
            serial=EpochBatchIterator(NNUNetBatchSource(spec),workers=0,seed=17,stage=stage)
            expected=[next(serial) for _ in range(2)]
            serial.close()
            parallel=EpochBatchIterator(NNUNetBatchSource(spec),workers=2,prefetch=4,seed=17,stage=stage,timeout_seconds=120,
                worker_threads=2,phase_steps=2)
            try:
                actual=next(parallel)
                assert np.array_equal(actual['keys'],expected[0]['keys'])
                assert torch.allclose(actual['data'],expected[0]['data'],rtol=0,atol=1e-6)
                for a,b in zip(actual['target'],expected[0]['target']):assert torch.equal(a,b)
                state=parallel.state_dict()
                pids=parallel.observations()['worker_pids']
                parallel.set_epoch(0)
                repeated=next(parallel)
                assert torch.allclose(repeated['data'],expected[0]['data'],rtol=0,atol=1e-6)
                assert parallel.observations()['worker_pids']==pids
                parallel.pause()
                assert all(row['loader_build_count']==1 for row in parallel.observations()['workers'])
                assert parallel.observations()['shared_batches']>0
            finally:parallel.close()
            restored=EpochBatchIterator(NNUNetBatchSource(spec),workers=2,prefetch=4,seed=17,stage=stage,timeout_seconds=120)
            try:
                restored.load_state_dict(state);actual=next(restored)
                assert np.array_equal(actual['keys'],expected[1]['keys'])
                assert torch.allclose(actual['data'],expected[1]['data'],rtol=0,atol=1e-6)
                for a,b in zip(actual['target'],expected[1]['target']):assert torch.equal(a,b)
            finally:restored.close()
            checks.append(stage+'_actual_loader_serial_spawn_and_resume')
        source=NNUNetBatchSource(dict(base,stage='validation'))
        iterator=EpochBatchIterator(source,workers=0,seed=3,stage='validation')
        try:next(iterator)
        finally:iterator.close()
        assert len(pickle.dumps(source))<10000
        assert pickle.loads(pickle.dumps(source))._loader is None
        checks.append('opened_case_cache_excluded_from_spawn_payload')
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',nnunet='2.8.1',torch=torch.__version__,device='cpu',
                cuda_initialized=False,real_ct=False,full_size_gpu=False,checks=checks)
    result.update(adapter_provenance=provenance,inner_observations=inside)
    if args.report_file:Path(args.report_file).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
