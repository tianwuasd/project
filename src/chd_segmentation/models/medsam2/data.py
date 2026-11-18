
import numpy as np
import torch

from ...preprocessing.prepare import load_prepared
from .native import NativeCases, restore_native, window_image
from training.dataset.vos_raw_dataset import VOSRawDataset,VOSFrame,VOSVideo
from training.dataset.vos_segment_loader import NPZSegmentLoader
from training.dataset.vos_dataset import VOSDataset

_read_gate = None
_prefetch_depth = 2
_rank = 0


class BudgetDataLoader(torch.utils.data.DataLoader):
    def __init__(self,dataset,**kwargs):
        workers = kwargs['num_workers']
        kwargs['prefetch_factor'] = max(1,_prefetch_depth//workers) if workers else None
        if _read_gate is not None:
            from .supply import OrderedDataset,TaggedBatches
            dataset = OrderedDataset(dataset,_read_gate,_rank)
            kwargs['batch_sampler'] = TaggedBatches(kwargs['batch_sampler'])
        super().__init__(dataset,**kwargs)


class ResourceWorkerInit:
    def __init__(self,common,owner,gate):
        self.common,self.owner,self.gate = common,owner,gate

    def __call__(self,worker_id):
        from ...training.resources import register_member
        register_member(self.common,self.owner)
        torch.set_num_threads(1)


def set_read_gate(gate,prefetch_depth,rank=0):
    global _read_gate,_prefetch_depth,_rank
    _read_gate = gate
    _prefetch_depth = prefetch_depth
    _rank = rank
    import training.dataset.sam2_datasets as official_data
    official_data.DataLoader = BudgetDataLoader


def sample_supply(gate,rank):
    global _read_gate,_rank
    _read_gate,_rank = gate,rank


class ManifestRawDataset(VOSRawDataset):
    def __init__(self,prepared_path,names,window,log_dir=None,logging=None,reserve_ram_bytes=4*2**30,input_source=None):
        from ..cases import ModelCases
        self.cases = ModelCases(load_prepared(prepared_path),
            dict(model='medsam2',input_source=input_source or {}),with_label=True)
        self.names,self.window = list(names),window

    def __len__(self):
        return len(self.names)

    def get_video(self,index):
        if _read_gate is not None:
            _read_gate.reserve(_rank,self.names[index])
        entry = self.cases.get(self.names[index])
        label = entry['label']
        if (label<0).any() or any((label==value).any() for value in entry['unknown_classes']):
            raise ValueError('MedSAM2训练标签包含未知区域或类别声明冲突')
        images = entry['image'][0].astype(np.float32)/255.
        frames = [VOSFrame(frame_index,None,torch.from_numpy(frame)[None].expand(3,-1,-1))
                  for frame_index,frame in enumerate(images)]
        return VOSVideo(entry['case_id'],index,frames),NPZSegmentLoader(label)


class HeartVOSDataset(VOSDataset):
    def _get_datapoint(self,index):
        if isinstance(index,torch.Tensor):
            index = index.item()
        video,segments = self.video_dataset.get_video(index)
        sampled = self.sampler.sample(video,segments,epoch=self.curr_epoch)
        datapoint = self.construct(video,sampled,segments)
        for transform in self._transforms:
            datapoint = transform(datapoint,epoch=self.curr_epoch)
        return datapoint
