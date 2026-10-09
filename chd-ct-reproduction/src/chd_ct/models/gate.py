"""Optional project extension; not part of the authors' U-Net definition."""

from torch import nn
from torch.nn import functional as F


class SpatialProbabilityGate(nn.Module):
    def __init__(self, dim, classes, base):
        super().__init__()
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.mode = "trilinear" if dim == 3 else "bilinear"
        self.network = nn.Sequential(
            conv(classes, base, 3, padding=1),
            nn.LeakyReLU(0.01),
            conv(base, base, 3, padding=1),
            nn.LeakyReLU(0.01),
            conv(base, classes, 1),
        )

    def forward(self, logits):
        shape = tuple(max(1, n // 4) for n in logits.shape[2:])
        coarse = F.interpolate(logits.softmax(1), size=shape, mode=self.mode, align_corners=False)
        prior = F.interpolate(
            self.network(coarse), size=logits.shape[2:], mode=self.mode, align_corners=False
        )
        return logits.log_softmax(1) + F.logsigmoid(prior)
