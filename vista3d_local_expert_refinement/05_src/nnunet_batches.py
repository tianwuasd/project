"""Lazy metadata-only adapter for the pinned nnU-Net loader/augmentations."""
import json


class NNUNetBatchSource:
    def __init__(self,spec):
        self.spec=json.loads(json.dumps(spec))
        self._loader=None
        self._seed=0

    def __iter__(self):return self

    def __getstate__(self):
        # Never pickle opened blosc/memmap/case caches into every worker.
        return dict(spec=self.spec,seed=self._seed)

    def __setstate__(self,state):
        self.spec=state['spec'];self._seed=state['seed'];self._loader=None

    def reseed(self,seed):
        self._seed=int(seed)
        if self._loader is not None:
            # batchgenerators DataLoader's private shuffle RNG. Infinite nnU-Net
            # sampling also uses the module NumPy RNG seeded by EpochBatchIterator.
            private=getattr(self._loader,'rs',None)
            if private is not None and hasattr(private,'seed'):
                private.seed(self._seed%(2**32))

    def _build(self):
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        from project_trainers import BaselineNNUNet
        s=self.spec;manager=PlansManager(s['plans'])
        labels=manager.get_label_manager(s['dataset_json'])
        config=manager.get_configuration(s['configuration'])
        if s['stage']=='train':
            transforms=BaselineNNUNet.get_training_transforms(
                config.patch_size,s['rotation'],s['deep_supervision'],s['mirror_axes'],s['dummy_2d'],
                use_mask_for_norm=config.use_mask_for_norm,is_cascaded=False,
                foreground_labels=labels.foreground_labels,
                regions=labels.foreground_regions if labels.has_regions else None,
                ignore_label=labels.ignore_label)
            initial=s['initial_patch']
        else:
            transforms=BaselineNNUNet.get_validation_transforms(s['deep_supervision'],
                is_cascaded=False,foreground_labels=labels.foreground_labels,
                regions=labels.foreground_regions if labels.has_regions else None,
                ignore_label=labels.ignore_label)
            initial=config.patch_size
        dataset_cls=infer_dataset_class(s['folder'])
        dataset=dataset_cls(s['folder'],identifiers=s['ids'],folder_with_segs_from_previous_stage=None)
        self._loader=nnUNetDataLoader(dataset,s['local_batch'],initial,config.patch_size,labels,
            oversample_foreground_percent=s['oversample'],sampling_probabilities=None,pad_sides=None,
            transforms=transforms,probabilistic_oversampling=s['probabilistic_oversampling'])
        self.reseed(self._seed)

    def __next__(self):
        if self._loader is None:self._build()
        return next(self._loader)


def build_loaders(trainer,resource):
    from batch_pool import EpochBatchIterator
    if trainer.is_cascaded:
        raise ValueError('当前自动加载适配仅支持已注册的3d_fullres B0，不静默改变级联输入')
    train_ids,val_ids=trainer.do_split()
    rotation,dummy,initial,mirrors=trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    base=dict(plans=trainer.plans_manager.plans,dataset_json=trainer.dataset_json,
        configuration='3d_fullres',folder=trainer.preprocessed_dataset_folder,
        local_batch=int(trainer.batch_size),oversample=float(trainer.oversample_foreground_percent),
        probabilistic_oversampling=bool(trainer.probabilistic_oversampling),rotation=rotation,
        dummy_2d=bool(dummy),initial_patch=list(map(int,initial)),mirror_axes=mirrors,
        deep_supervision=[list(map(float,scale)) for scale in trainer._get_deep_supervision_scales()])
    outputs=[]
    for stage,ids,key in [('train',train_ids,'augmentation_workers'),('validation',val_ids,'validation_workers')]:
        workers=int(resource.get(key,0))
        source=NNUNetBatchSource(dict(base,stage=stage,ids=list(ids)))
        outputs.append(EpochBatchIterator(source,workers=workers,seed=0,epoch=trainer.current_epoch,
            rank=trainer.local_rank,stage=stage,prefetch=max(1,workers)*int(resource.get('prefetch',2)),
            timeout_seconds=resource.get('batch_timeout_seconds',180),pin_memory=trainer.device.type=='cuda'))
    return tuple(outputs)


def close_loaders(trainer):
    from batch_pool import EpochBatchIterator
    from training_timing import TimedIterator
    for name in ['dataloader_train','dataloader_val']:
        loader=getattr(trainer,name,None)
        if isinstance(loader,TimedIterator):loader=loader.source
        if isinstance(loader,EpochBatchIterator):loader.close()
