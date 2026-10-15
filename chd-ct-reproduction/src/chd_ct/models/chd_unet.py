"""Xu et al., MedIA 2023, Figure 4 U-Nets adapted to ImageCHD labels.

This is a paper-derived implementation, not the authors' source code.
Unspecified choices (same padding, IN affine parameters, activation slope,
and probability normalization) are documented in docs/paper-mapping.md.
The older Freiburg source port remains available in unet.py / grid.py.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .grid import mirror_pad
from .unet import center_crop

ARCHITECTURE = "chd-media2023-fig4-v1"
GRID_ADAPTER = "same-pad-symmetric-lattice-crop-v1"


def run_block(module, x, activation_checkpoint):
    if activation_checkpoint and module.training and torch.is_grad_enabled():
        return checkpoint(module, x, use_reentrant=False)
    return module(x)


def conv_norm_activation(dim, inputs, outputs):
    conv = nn.Conv3d if dim == 3 else nn.Conv2d
    norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
    return [
        conv(inputs, outputs, 3, padding=1),
        norm(outputs, affine=True),
        nn.LeakyReLU(0.01),
    ]


class PaperUNet(nn.Module):
    """Two convolutions per scale, with a same-resolution feature output."""

    def __init__(self, dim, classes, base, levels, activation_checkpoint=False):
        super().__init__()
        if dim not in (2, 3) or min(classes, base) < 1 or levels < 2:
            raise ValueError("dim must be 2/3; classes/base >= 1; levels >= 2")
        self.dim, self.base, self.levels = dim, base, levels
        self.stride = 2 ** (levels - 1)
        self.feature_channels = base
        self.activation_checkpoint = activation_checkpoint
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        pool = nn.MaxPool3d if dim == 3 else nn.MaxPool2d
        up = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d
        channels = [base * 2**i for i in range(levels)]

        def block(cin, cout):
            return nn.Sequential(
                *conv_norm_activation(dim, cin, cout),
                *conv_norm_activation(dim, cout, cout),
            )

        self.encoder = nn.ModuleList(
            block(1 if i == 0 else channels[i - 1], cout) for i, cout in enumerate(channels)
        )
        self.pool = pool(2, stride=2)
        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(levels - 2, -1, -1):
            self.ups.append(up(channels[i + 1], channels[i], 2, stride=2))
            self.decoder.append(block(2 * channels[i], channels[i]))
        self.head = conv(base, classes, 1)

    def features(self, x):
        if x.ndim != self.dim + 2 or any(n % self.stride or n < 2 * self.stride for n in x.shape[2:]):
            raise ValueError("输入需要匹配下采样网格且最深层每轴至少为 2；请使用 CHDUNet 网格适配")
        skips = []
        for i, block in enumerate(self.encoder):
            x = run_block(block, x if i == 0 else self.pool(x), self.activation_checkpoint)
            skips.append(x)
        for up, block, skip in zip(self.ups, self.decoder, reversed(skips[:-1])):
            x = run_block(block, torch.cat([skip, up(x)], dim=1), self.activation_checkpoint)
        return x

    def forward(self, x):
        return self.head(self.features(x))


class PositionCorrelation(nn.Module):
    """Figure 4(a): pool4 -> three 3x3x3 conv blocks -> transpose-conv4.

    Return log scores whose softmax equals the normalized pointwise product
    of the U-Net probabilities and a sigmoid spatial prior.
    """

    def __init__(self, classes):
        super().__init__()
        self.pool = nn.MaxPool3d(4, stride=4)
        self.network = nn.Sequential(
            *conv_norm_activation(3, classes, 64),
            *conv_norm_activation(3, 64, 64),
            *conv_norm_activation(3, 64, classes),
        )
        self.up = nn.ConvTranspose3d(classes, classes, 4, stride=4)

    def forward(self, logits):
        prior = self.up(self.network(self.pool(logits.softmax(1))))
        return logits.log_softmax(1) + F.logsigmoid(prior)


class CHDUNet(nn.Module):
    """Paper U-Net plus coordinate-preserving padding for non-paper grids.

    The 64/128/512 paper input grids require no extra padding. Tiny or odd
    grids are symmetrically extended then center-cropped, never resized here.
    """

    def __init__(self, dim, classes, base, levels, spatial_gate=False, activation_checkpoint=False):
        super().__init__()
        if spatial_gate and dim != 3:
            raise ValueError("Figure 4 position correlation is implemented for 3D only")
        self.core = PaperUNet(dim, classes, base, levels, activation_checkpoint=activation_checkpoint)
        self.gate = PositionCorrelation(classes) if spatial_gate else None
        self.feature_channels = base

    def _extended_features(self, x):
        stride = max(self.core.stride, 4 if self.gate is not None else 1)
        shape = tuple(max(2 * stride, ((n + stride - 1) // stride) * stride) for n in x.shape[2:])
        return self.core.features(mirror_pad(x, shape))

    def features(self, x):
        return center_crop(self._extended_features(x), x.shape[2:])

    def forward(self, x):
        logits = self.core.head(self._extended_features(x))
        if self.gate is not None:
            logits = run_block(self.gate, logits, self.core.activation_checkpoint)
        return center_crop(logits, x.shape[2:])
