import torch


FIELDS = ('images', 'masks', 'valid', 'boxes')


def pin_clip(clip, device):
    with torch.cuda.device(device):
        return dict(clip, **{name: clip[name].pin_memory() for name in FIELDS})


class CudaClipTransfer:
    def __init__(self, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.pending = None

    def stage(self, clip):
        if self.pending is not None:
            raise RuntimeError('MedSAM2 已有一个待消费的 CUDA 片段')
        if not all(clip[name].is_pinned() for name in FIELDS):
            raise RuntimeError('MedSAM2 异步传输要求锁页的 CPU 张量')
        with torch.cuda.stream(self.stream):
            tensors = {name: clip[name].to(self.device, non_blocking=True) for name in FIELDS}
            complete = self.stream.record_event()
        self.pending = clip, tensors, complete

    def take(self, clip):
        if self.pending is None:
            return clip
        source, tensors, complete = self.pending
        if source is not clip:
            raise RuntimeError('MedSAM2 CUDA 预传输片段与消费顺序不一致')
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(complete)
        # 计算流使用传输流分配的张量，释放时由 PyTorch 等待计算流。
        for tensor in tensors.values():
            tensor.record_stream(compute)
        self.pending = None
        return tensors

    def clear(self):
        if self.pending is not None:
            self.stream.synchronize()
            self.pending = None
