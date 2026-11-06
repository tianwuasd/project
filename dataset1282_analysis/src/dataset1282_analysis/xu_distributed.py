"""Shard a fixed batch of independent cases, then gather scores for its original global loss."""
import json
import os
import re

import torch
from torch.nn import parallel


def split_sizes(count, workers):
    if count < 1 or workers < 1:
        raise ValueError('A branch needs a nonempty batch and at least one device')
    active = min(count, workers)
    base, remainder = divmod(count, active)
    return tuple(base + (i < remainder) for i in range(active))


def _check_model_device(model, primary):
    if any(t.device != primary for t in (*model.parameters(), *model.buffers())):
        raise ValueError('The authoritative parameters and buffers must be on the assigned primary device')


def _replica_forward(model, chunks, devices, frozen=None):
    # Unindexed 'cuda' is permitted by the legacy one-device CLI.
    primary = devices[0]
    if primary.index is None and primary.type == 'cuda':
        primary = torch.device('cuda', torch.cuda.current_device())
    if chunks[0].device != primary:
        raise ValueError('Input must be on the assigned primary device')
    if len(devices) == 1:
        return model(chunks[0])
    _check_model_device(model, primary)
    replicas = frozen or parallel.replicate(model, devices, detach=not torch.is_grad_enabled())
    inputs = [(part.to(device),) for part, device in zip(chunks, devices)]
    scores = parallel.parallel_apply(replicas, inputs, devices=devices)
    # Gather and replicate are autograd-aware. Backward sums replica gradients into the
    # original model on the primary; only its optimizer ever owns or updates parameters.
    return parallel.gather(scores, devices[0], dim=0)


class BranchForward:
    """An invocation is one logical training/validation or feature-extraction batch."""

    def __init__(self, primary, batch_size, *, require_all=False):
        if batch_size < 1:
            raise ValueError('Expected a positive fixed branch batch size')
        source = torch.device(primary)
        if source.type not in ('cpu', 'cuda'):
            raise ValueError('Expected a CPU or CUDA branch')
        setting = os.environ.get('XU_BRANCH_DEVICES')
        if setting is None:
            devices = [source]
        else:
            names = json.loads(setting)
            if (not isinstance(names, list) or not names
                    or not all(isinstance(s, str) and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', s) for s in names)
                    or len(set(names)) != len(names)):
                raise ValueError('Expected an explicit ordered list of unique cuda:<index> branch devices')
            devices = [torch.device(name) for name in names]
            if source != devices[0]:
                raise ValueError('Branch primary disagrees with the first assigned device')
            if len(devices) > 1:
                from .xu_resources import require_applied_group
                require_applied_group(names)
        if require_all and len(devices) > batch_size:
            raise ValueError('Assigned device count exceeds the fixed branch batch size')
        self.devices = tuple(devices)
        self.batch_size = batch_size
        self.max_active = min(batch_size, len(devices))
        self.stats = dict(assigned_devices=list(map(str, devices)),
                          cases_by_device={str(device): 0 for device in devices}, batches_by_layout={})
        self._frozen = None

    @staticmethod
    def _frozen_signature(model):
        if any(layer.training for layer in model.modules()):
            raise ValueError('Persistent replicas require an evaluation-mode model')
        tensors = (*model.parameters(), *model.buffers())
        if any(t.requires_grad or t.is_inference() for t in tensors):
            raise ValueError('Persistent replicas require frozen, versioned parameters and buffers')
        return tuple((id(t), t._version) for t in tensors)

    def freeze(self, model):
        """Bind detached, read-only replicas for this phase's fixed feature extractor."""
        if self._frozen is not None:
            raise ValueError('A frozen group is already bound to this phase')
        signature = self._frozen_signature(model)
        with torch.no_grad():
            if self.max_active > 1:
                _check_model_device(model, self.devices[0])
                replicas = parallel.replicate(model, self.devices[:self.max_active], detach=True)
            else:
                replicas = (model,)
        self._frozen = model, signature, replicas

    def forward(self, model, images):
        count = len(images)
        if count > self.batch_size:
            raise ValueError('Input exceeds the declared fixed branch batch size')
        sizes = split_sizes(count, self.max_active)
        chunks = images.split(sizes, dim=0) if len(sizes) > 1 else (images,)
        active = self.devices[:len(sizes)]
        if self._frozen is None:
            scores = _replica_forward(model, chunks, active)
        else:
            original, signature, replicas = self._frozen
            if (torch.is_grad_enabled() or original is not model
                    or self._frozen_signature(model) != signature):
                raise ValueError('Frozen replica use requires unchanged weights in a no-grad phase')
            scores = _replica_forward(model, chunks, active, replicas[:len(sizes)])
        if not isinstance(scores, torch.Tensor) or scores.ndim < 1 or scores.shape[0] != count:
            raise ValueError('Expected one output per input, in the original batch order')
        layout = ','.join(map(str, sizes + (0,) * (len(self.devices) - len(sizes))))
        for device, size in zip(self.devices, sizes):
            self.stats['cases_by_device'][str(device)] += size
        layouts = self.stats['batches_by_layout']
        layouts[layout] = layouts.get(layout, 0) + 1
        return scores
