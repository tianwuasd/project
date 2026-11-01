"""ImageCHD grid adapter around the unchanged valid-convolution backbone.

Symmetric mirror extension supplies context at image edges. Valid outputs are
center-cropped to the requested grid, never stretched back over the whole image.
"""

import torch
from torch import nn

from .gate import SpatialProbabilityGate
from .unet import UNet, center_crop

GRID_ADAPTER = "symmetric-mirror-valid-center-crop-v1"


def mirror_pad(x, shape):
    if len(shape) != x.ndim - 2 or any(new < old for new, old in zip(shape, x.shape[2:])):
        raise ValueError("镜像扩展尺寸无效")
    for axis, new in enumerate(shape, start=2):
        old = x.shape[axis]
        if old < 1:
            raise ValueError("输入不能有空维度")
        left = (new - old) // 2
        indices = torch.arange(-left, new - left, device=x.device).remainder(2 * old)
        indices = torch.where(indices < old, indices, 2 * old - 1 - indices)
        x = x.index_select(axis, indices)
    return x


class GridUNet(nn.Module):
    def __init__(self, dim, classes, base=None, levels=None, spatial_gate=False):
        super().__init__()
        self.core = UNet(dim, classes, base=base, levels=levels)
        self.gate = SpatialProbabilityGate(dim, classes, self.core.base) if spatial_gate else None
        self.feature_channels = self.core.feature_channels

    def features(self, x):
        shape = x.shape[2:]
        extended = mirror_pad(x, self.core.input_shape_for_output(shape))
        return center_crop(self.core.features(extended), shape)

    def forward(self, x):
        logits = self.core.head(self.features(x))
        return self.gate(logits) if self.gate is not None else logits
