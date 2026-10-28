"""Joint image/label transforms; amplitudes are explicit reproduction assumptions."""

import math

import torch
from torch.nn import functional as F


def augment_batch(x, y):
    # A sequence uses the same flip on every frame.
    if x.ndim == 5 and y.ndim == 3:
        if torch.rand(()) < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y
    dim = x.ndim - 2
    theta = torch.eye(dim, dim + 1, device=x.device)[None].repeat(x.shape[0], 1, 1)
    angle = (torch.rand(x.shape[0], device=x.device) - 0.5) * (math.pi / 6)
    scale = 0.9 + 0.2 * torch.rand(x.shape[0], device=x.device)
    theta[:, 0, 0] = angle.cos() * scale
    theta[:, 1, 1] = angle.cos() * scale
    theta[:, 0, 1] = -angle.sin() * scale
    theta[:, 1, 0] = angle.sin() * scale
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    if dim == 3:
        noise = torch.randn((x.shape[0], 3, 4, 4, 4), device=x.device) * 0.015
        elastic = F.interpolate(noise, size=x.shape[2:], mode="trilinear", align_corners=False).movedim(1, -1)
        grid = grid + elastic
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[
        :, 0
    ].long()
    if torch.rand(()) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    return x, y
