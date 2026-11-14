from time import perf_counter

import numpy as np
import torch

from ...preprocessing.prepare import load_prepared
from ...preprocessing.transforms import CaseTransform
from training.dataset.vos_raw_dataset import VOSRawDataset,VOSFrame,VOSVideo
from training.dataset.vos_segment_loader import NPZSegmentLoader
from training.dataset.vos_dataset import VOSDataset

_read_gate = None
_prefetch_depth = 2


class BudgetDataLoader(torch.utils.data.DataLoader):
    def __init__(self,dataset,**kwargs):
        workers = kwargs['num_workers']
        kwargs['prefetch_factor'] = max(1,_prefetch_depth//workers) if workers else None
        super().__init__(dataset,**kwargs)


class ResourceWorkerInit:
    def __init__(self,common,owner,gate):
        self.common,self.owner,self.gate = common,owner,gate

    def __call__(self,worker_id):
        global _read_gate
        from ...training.resources import register_member
        register_member(self.common,self.owner)
        torch.set_num_threads(1)
        _read_gate = self.gate


def set_read_gate(gate,prefetch_depth):
    global _read_gate,_prefetch_depth
    _read_gate = gate
    _prefetch_depth = prefetch_depth
    import training.dataset.sam2_datasets as official_data
    official_data.DataLoader = BudgetDataLoader


def window_image(image,window):
    low,high = window
    if not np.isfinite(window).all() or low>=high:
        raise ValueError('MedSAM2 CT窗范围需要两个递增的有限数值')
    clipped = np.clip(image,low,high)
    minimum,maximum = float(clipped.min()),float(clipped.max())
    if maximum<=minimum:
        raise ValueError('MedSAM2 CT窗内没有有效灰度变化')
    return ((clipped-minimum)/(maximum-minimum)*255.).astype(np.uint8)


class NativeCases:
    def __init__(self,prepared,with_label=True):
        self.prepared = prepared
        self.rows = {row['case_id']:row for row in prepared['manifest']['cases']}
        self.reader = CaseTransform(prepared,4*2**30,with_label=with_label)
        self.current_name,self.current = None,None

    def get(self,name):
        if _read_gate is None:
            return self._get(name)
        with _read_gate:
            return self._get(name)

    def _get(self,name):
        row = self.rows[name]
        self.reader.check_sources(row)
        if name==self.current_name:
            return self.current
        started = perf_counter()
        self.current_name,self.current = None,None
        data,label,image,spacing,roi,read_seconds = self.reader.read_case(row)
        self.current = dict(case_id=name,image=np.ascontiguousarray(data.transpose(2,1,0))[None],
            label=np.ascontiguousarray(label.transpose(2,1,0)) if label is not None else None,
            unknown_classes=row['unknown_classes'],source_shape_xyz=list(image.shape),
            source_affine=image.affine.tolist(),source_units=image.header.get_xyzt_units()[0],roi_xyz=roi,
            timing=dict(read_decompress_seconds=read_seconds,native_seconds=perf_counter()-started),
            medsam_native=True)
        self.current_name = name
        return self.current

    def clear(self):
        self.current_name,self.current = None,None


def restore_native(probability,entry):
    label = probability.argmax(0).astype(np.uint8).transpose(2,1,0)
    result = np.zeros(entry['source_shape_xyz'],dtype=np.uint8)
    region = tuple(slice(low,high) for low,high in entry['roi_xyz'])
    if result[region].shape!=label.shape:
        raise ValueError('MedSAM2导出尺寸与源图范围不同')
    result[region] = label
    return result


class ManifestRawDataset(VOSRawDataset):
    def __init__(self,prepared_path,names,window):
        self.cases = NativeCases(load_prepared(prepared_path),with_label=True)
        self.names,self.window = list(names),window

    def __len__(self):
        return len(self.names)

    def get_video(self,index):
        entry = self.cases.get(self.names[index])
        label = entry['label']
        if (label<0).any() or any((label==value).any() for value in entry['unknown_classes']):
            raise ValueError('MedSAM2训练标签包含未知区域或类别声明冲突')
        images = window_image(entry['image'][0],self.window).astype(np.float32)/255.
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
