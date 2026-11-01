"""U-Net family with instance normalization and an optional spatial probability gate."""

import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self, dim, classes, base=32, levels=5, spatial_gate=False):
        super().__init__()
        if dim not in (2, 3) or levels < 2 or base < 1:
            raise ValueError("dim must be 2/3, levels >= 2, base >= 1")
        self.dim, self.levels = dim, levels
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
        pool = nn.MaxPool3d if dim == 3 else nn.MaxPool2d
        up = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d

        def block(cin, cout):
            return nn.Sequential(
                conv(cin, cout, 3, padding=1),
                norm(cout, affine=True),
                nn.LeakyReLU(0.01),
                conv(cout, cout, 3, padding=1),
                norm(cout, affine=True),
                nn.LeakyReLU(0.01),
            )

        channels = [base * 2**i for i in range(levels)]
        self.encoder = nn.ModuleList(
            [block(1 if i == 0 else channels[i - 1], c) for i, c in enumerate(channels)]
        )
        self.pool = pool(2)
        self.ups = nn.ModuleList(
            [up(channels[i], channels[i - 1], 2, stride=2) for i in range(levels - 1, 0, -1)]
        )
        self.decoder = nn.ModuleList(
            [block(channels[i - 1] * 2, channels[i - 1]) for i in range(levels - 1, 0, -1)]
        )
        self.head = conv(base, classes, 1)
        self.gate = (
            nn.Sequential(
                conv(classes, base, 3, padding=1),
                nn.LeakyReLU(0.01),
                conv(base, base, 3, padding=1),
                nn.LeakyReLU(0.01),
                conv(base, classes, 1),
            )
            if spatial_gate
            else None
        )

    def features(self, x):
        if min(x.shape[2:]) < 2**self.levels:
            raise ValueError(f"Each input dimension must be >= {2**self.levels} for instance normalization")
        skips = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x if i == 0 else self.pool(x))
            skips.append(x)
        mode = "trilinear" if self.dim == 3 else "bilinear"
        for up, decoder, skip in zip(self.ups, self.decoder, reversed(skips[:-1])):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return x

    def forward(self, x):
        logits = self.head(self.features(x))
        if self.gate is not None:
            # Softmax(log p + log gate) = normalized pointwise probability multiplication.
            coarse = F.interpolate(
                logits.softmax(1),
                scale_factor=0.25,
                mode="trilinear" if self.dim == 3 else "bilinear",
                align_corners=False,
            )
            prior = self.gate(coarse)
            prior = F.interpolate(
                prior,
                size=logits.shape[2:],
                mode="trilinear" if self.dim == 3 else "bilinear",
                align_corners=False,
            )
            logits = logits.log_softmax(1) + F.logsigmoid(prior)
        return logits
