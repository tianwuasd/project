"""Xu-style slice encoder and separately trained seven-slice context classifier."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.01),
        nn.Conv2d(cout, cout, 3, padding=1), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.01),
    )


class SliceUNet(nn.Module):
    """Penultimate layer is the input to the temporal classifier."""
    def __init__(self, profile):
        super().__init__()
        self.activation_checkpointing = True
        levels, self.width = {'paper': (8, 32), 'smoke': (3, 2)}[profile]
        channels = [self.width * 2**i for i in range(levels)]
        self.encoder = nn.ModuleList(block(before, after) for before, after
                                     in zip([1, *channels[:-1]], channels))
        self.up = nn.ModuleList(nn.ConvTranspose2d(high, low, 2, stride=2)
                                for low, high in reversed(list(zip(channels, channels[1:]))))
        self.decoder = nn.ModuleList(block(2 * w, w) for w in reversed(channels[:-1]))
        self.head = nn.Conv2d(self.width, 3, 1)

    def _block(self, layer, x):
        if self.activation_checkpointing and self.training and len(self.encoder) == 8 and torch.is_grad_enabled():
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def features(self, x):
        skips = []
        for i, layer in enumerate(self.encoder):
            x = self._block(layer, F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for up, layer, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            x = self._block(layer, torch.cat([up(x), skip], 1))
        return x

    def forward(self, x):
        return self.head(self.features(x))


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hidden):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(cin + hidden, 4 * hidden, 3, padding=1)

    def forward(self, x, h, c):
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c


class BloodSequence(nn.Module):
    """Two bidirectional ConvLSTM layers. Classify the center of a seven-slice window."""
    def __init__(self, width):
        super().__init__()
        self.activation_checkpointing = True
        self.layers = nn.ModuleList(nn.ModuleList([ConvLSTMCell(cin, width),
                                                   ConvLSTMCell(cin, width)])
                                    for cin in (width, 2 * width))
        self.head = nn.Conv2d(2 * width, 3, 1)

    def _scan(self, cell, frames):
        batch, _, height, width = frames[0].shape
        h = frames[0].new_zeros(batch, cell.hidden, height, width)
        c = torch.zeros_like(h)
        result = []
        for x in frames:
            if self.activation_checkpointing and self.training and cell.hidden == 32:
                h, c = checkpoint(cell, x, h, c, use_reentrant=False)
            else:
                h, c = cell(x, h, c)
            result.append(h)
        return result

    def forward(self, frames):
        if len(frames) != 7:
            raise ValueError('Expected seven ordered slices from one scan')
        for i, (fwd, rev) in enumerate(self.layers):
            if i == 1:
                before = self._scan(fwd, frames[:4])[-1]
                after = self._scan(rev, frames[3:][::-1])[-1]
                return self.head(torch.cat([before, after], 1))
            a = self._scan(fwd, frames)
            b = self._scan(rev, frames[::-1])
            frames = [torch.cat([u, v], 1) for u, v in zip(a, reversed(b))]
