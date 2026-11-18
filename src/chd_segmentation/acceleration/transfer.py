from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TransferHandle:
    tensors: Any
    completion_event: Any
    host_buffers: list

    def release(self):
        if self.completion_event is not None and not self.completion_event.query():
            self.completion_event.synchronize()
        self.host_buffers.clear()


def transfer_batch(tensors, device, *, client):
    device = torch.device(device)
    if client.boundary().action == 'stop':
        raise InterruptedError('数据传输已取消')
    buffers = []
    def transfer(value):
        if isinstance(value, torch.Tensor):
            if device.type == 'cuda' and value.device.type == 'cpu':
                value = value if value.is_pinned() else value.pin_memory()
                buffers.append(value)
            return value.to(device, non_blocking=device.type == 'cuda')
        if isinstance(value, dict):
            return {key: transfer(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(transfer(item) for item in value)
        return value
    result = transfer(tensors)
    completed = None
    if device.type == 'cuda':
        completed = torch.cuda.Event()
        completed.record(torch.cuda.current_stream(device))
    return TransferHandle(result, completed, buffers)
