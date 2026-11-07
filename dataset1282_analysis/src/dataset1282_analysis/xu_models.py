"""Fixed-grid 3D branches of Xu et al. (MedIA 2023), Figure 4(a) and Table 3.

Port of this repository family's chd-ct-reproduction/chd_ct/models/chd_unet.py.
The previous seven-structure project used the same encoder, decoder, and
spatial prior, but the heads here have ten foreground structures plus background.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class BranchSpec:
    side: int
    width: int
    levels: int
    prior_width: int
    batch: int
    epochs: int


def branch_spec(branch, profile='paper'):
    sides = {'crop64': 64, 'crop128': 128, 'all64': 64, 'all128': 128}
    if branch not in sides or profile not in ('paper', 'smoke'):
        raise ValueError('Expected crop64/crop128/all64/all128 with paper or smoke profile')
    side = sides[branch]
    if profile == 'smoke':
        return BranchSpec(side, 2, 3, 4, 1, 1)
    return BranchSpec(side, 64 if side == 64 else 48, 5, 64, 4, 15)


def _conv(cin, cout):
    return [nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01)]


def _block(cin, cout):
    return nn.Sequential(*_conv(cin, cout), *_conv(cout, cout))


class Xu3DSegmenter(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        channels = [spec.width * 2**i for i in range(spec.levels)]
        self.encoder = nn.ModuleList(_block(cin, cout)
                                     for cin, cout in zip([1, *channels[:-1]], channels))
        self.pool = nn.MaxPool3d(2)
        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in reversed(range(spec.levels - 1)):
            self.ups.append(nn.ConvTranspose3d(channels[i+1], channels[i], 2, stride=2))
            self.decoder.append(_block(2 * channels[i], channels[i]))
        self.head = nn.Conv3d(channels[0], 11, 1)
        self.prior = nn.Sequential(nn.MaxPool3d(4),
                                   *_conv(11, spec.prior_width),
                                   *_conv(spec.prior_width, spec.prior_width),
                                   *_conv(spec.prior_width, 11),
                                   nn.ConvTranspose3d(11, 11, 4, stride=4))

    def block(self, module, x):
        if self.training and torch.is_grad_enabled() and self.spec.levels == 5:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:] != (1, self.spec.side, self.spec.side, self.spec.side):
            raise ValueError(f'Expected batch of single-channel {self.spec.side}-cubed volumes')
        skips = []
        for i, block in enumerate(self.encoder):
            x = self.block(block, self.pool(x) if i else x)
            skips.append(x)
        for up, block, skip in zip(self.ups, self.decoder, reversed(skips[:-1])):
            x = self.block(block, torch.cat((skip, up(x)), dim=1))
        logits = self.head(x)
        prior = self.block(self.prior, logits.softmax(1))
        return logits.log_softmax(1) + F.logsigmoid(prior)


def build_model(branch, profile='paper'):
    return Xu3DSegmenter(branch_spec(branch, profile))
