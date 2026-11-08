import math
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .xu_supply import BatchSupply
from .xu_supply_budget import blood_batch_bytes


def prepare_slices(views, centres, temporal):
    windows = [views.window(z) if temporal else [z] for z in centres]
    images = torch.from_numpy(np.stack([[views.image(i) for i in w] for w in windows]))
    labels = torch.from_numpy(np.stack([views.label(z) for z in centres])).long()
    return images, labels


def augment_slices(images, labels, rng):
    if rng is None:
        return images, labels
    matrices = []
    for _ in range(len(images)):
        angle = rng.uniform(-math.pi/12, math.pi/12)
        flip = rng.choice([-1, 1])
        matrices.append([[flip*math.cos(angle), -math.sin(angle), 0],
                         [flip*math.sin(angle), math.cos(angle), 0]])
    grid = F.affine_grid(images.new_tensor(matrices), images.shape, align_corners=False)
    images = F.grid_sample(images, grid, align_corners=False)
    labels = F.grid_sample((labels[:,None]+1).float(), grid, mode='nearest', align_corners=False)[:,0].long()-1
    return images, labels


class SliceSupply(BatchSupply):
    def __init__(self, views, centres, batch_size, temporal):
        self.views = views
        self.temporal = temporal
        self.voxels = math.prod(views.cache_shape)
        super().__init__(list(centres), batch_size, 1,
                         lambda batch: prepare_slices(views, batch, temporal), lambda _: self.voxels)

    def _reserve(self, batch):
        return blood_batch_bytes(self.voxels, len(batch), self.temporal)


class PhaseTiming:
    def __init__(self):
        self.started = perf_counter()
        self.totals = dict(cpu_prepare_seconds=0., cpu_wait_seconds=0., transfer_call_seconds=0.,
                           pin_seconds=0., step_seconds=0., device_finish_seconds=0.,
                           prefetch_batches=0, memory_denied_batches=0)
        self.peaks = dict(queue_peak=0, reserved_peak_bytes=0, workers_applied=0)

    def add(self, supply, step_seconds, finish_seconds):
        for key in self.totals:
            self.totals[key] += (step_seconds if key == 'step_seconds' else
                                finish_seconds if key == 'device_finish_seconds' else supply.stats[key])
        for key in self.peaks:
            self.peaks[key] = max(self.peaks[key], supply.stats[key])

    def report(self, slices, cases, total):
        seconds = perf_counter()-self.started
        return dict(seconds=seconds, slices_per_second=slices/max(seconds, 1e-9),
                    phase_eta_seconds=seconds*(total-cases)/cases if cases else None,
                    **self.totals, **self.peaks)
