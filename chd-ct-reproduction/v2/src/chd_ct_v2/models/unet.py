"""PyTorch port of the Freiburg authors' published valid-convolution U-Nets.

2D: 2015-10-02 release, phseg_v5-train.prototxt (including its final upconv).
3D: MICCAI 2016 no-BN prototxt. Original definitions/licenses: reference/.
The core has no spatial gate or same-size padding. See grid.py and gate.py.
"""

import math

import torch
from torch import nn

ARCHITECTURE = "freiburg-caffe-2015-2d-2016-3d-nobn-port-v1"


def center_crop(x, shape):
    if len(shape) != x.ndim - 2 or any(n > old for n, old in zip(shape, x.shape[2:])):
        raise ValueError("中心裁剪尺寸无效")
    slices = tuple(slice((old - n) // 2, (old - n) // 2 + n) for old, n in zip(x.shape[2:], shape))
    return x[(slice(None), slice(None), *slices)]


class UNet(nn.Module):
    def __init__(self, dim, classes, base=None, levels=None, spatial_gate=False):
        super().__init__()
        if spatial_gate:
            raise ValueError("gate 是独立扩展，请使用 GridUNet(spatial_gate=True)")
        base = (64 if dim == 2 else 32) if base is None else base
        levels = (5 if dim == 2 else 4) if levels is None else levels
        if dim not in (2, 3) or classes < 1 or base < 1 or levels < 2:
            raise ValueError("dim must be 2/3; classes/base >= 1; levels >= 2")
        self.dim, self.levels, self.base = dim, levels, base
        self.stride = 2 ** (levels - 1)
        self.shrink = 12 * self.stride - 8
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        pool = nn.MaxPool3d if dim == 3 else nn.MaxPool2d
        up = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d

        def block(cin, middle, cout, dropout=False):
            layers = [conv(cin, middle, 3), nn.ReLU(), conv(middle, cout, 3), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        channels = [base * 2 ** (i + (dim == 3)) for i in range(levels)]
        self.feature_channels = channels[0]
        self.encoder = nn.ModuleList()
        for i, cout in enumerate(channels):
            middle = cout // 2 if dim == 3 else cout
            self.encoder.append(
                block(1 if i == 0 else channels[i - 1], middle, cout, dropout=dim == 2 and i >= levels - 2)
            )
        self.pool = pool(2, stride=2, ceil_mode=True)
        self.ups, self.decoder = nn.ModuleList(), nn.ModuleList()
        current = channels[-1]
        for i in range(levels - 2, -1, -1):
            # Preserve the source's channel counts, including 2D's last 128-channel upconv.
            up_channels = current if dim == 3 else (channels[i] if i > 0 else 2 * base)
            self.ups.append(nn.Sequential(up(current, up_channels, 2, stride=2), nn.ReLU()))
            self.decoder.append(block(up_channels + channels[i], channels[i], channels[i]))
            current = channels[i]
        self.head = conv(current, classes, 1)
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                # Caffe filler default FAN_IN uses weight.count / weight.shape(0), also for deconv.
                fan_in = module.weight.numel() / module.weight.shape[0]
                if self.dim == 2:
                    bound = math.sqrt(3 / fan_in)
                    nn.init.uniform_(module.weight, -bound, bound)
                else:
                    nn.init.normal_(module.weight, std=math.sqrt(2 / fan_in))
                nn.init.zeros_(module.bias)
        if self.dim == 3:
            nn.init.constant_(self.encoder[0][0].bias, -0.1)

    def input_shape_for_output(self, shape):
        """Smallest accepted input whose valid output covers each requested axis."""
        if len(shape) != self.dim or any(n < 1 for n in shape):
            raise ValueError("输出尺寸无效")
        bottom = [max(4, math.ceil((n - 4) / self.stride) + 4) for n in shape]
        return tuple(self.stride * (n + 8) - 4 for n in bottom)

    def features(self, x):
        minimum = self.input_shape_for_output((1,) * self.dim)[0]
        if x.ndim != self.dim + 2 or x.shape[1] != 1:
            raise ValueError("输入需为 [batch, 1, spatial...] single-channel CT")
        if any(n < minimum or (n + 4) % self.stride for n in x.shape[2:]):
            raise ValueError("输入 input 不满足官方有效卷积网格；使用 GridUNet 自动处理边界")
        skips = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x if i == 0 else self.pool(x))
            skips.append(x)
        for up, decoder, skip in zip(self.ups, self.decoder, reversed(skips[:-1])):
            x = up(x)
            # Official Caffe concat order: upsampled feature first, cropped encoder second.
            x = decoder(torch.cat([x, center_crop(skip, x.shape[2:])], dim=1))
        return x

    def forward(self, x):
        return self.head(self.features(x))
