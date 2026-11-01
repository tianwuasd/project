"""Joint image/label transforms; amplitudes are explicit reproduction assumptions."""

import math

import torch
from torch.nn import functional as F


def augment_batch(x, y):
    # One rotation/grid per case, shared by every frame and the center label.
    sequence = x.ndim == 5 and y.ndim == 3
    frame_shape = x[:, 0].shape if sequence else x.shape
    dim = len(frame_shape) - 2
    theta = torch.eye(dim, dim + 1, device=x.device, dtype=x.dtype)[None].repeat(x.shape[0], 1, 1)
    angle = (torch.rand(x.shape[0], device=x.device) - 0.5) * (math.pi / 6)
    # Section 5.1: scaling/deformation are 3D only; 2D uses rotation/flipping.
    scale = 0.9 + 0.2 * torch.rand(x.shape[0], device=x.device) if dim == 3 else 1.0
    theta[:, 0, 0] = angle.cos() * scale
    theta[:, 1, 1] = angle.cos() * scale
    theta[:, 0, 1] = -angle.sin() * scale
    theta[:, 1, 0] = angle.sin() * scale
    if dim == 3:
        theta[:, 2, 2] = scale
    grid = F.affine_grid(theta, frame_shape, align_corners=False)
    if dim == 3:
        noise = torch.randn((x.shape[0], 3, 4, 4, 4), device=x.device) * 0.015
        elastic = F.interpolate(noise, size=x.shape[2:], mode="trilinear", align_corners=False).movedim(1, -1)
        grid = grid + elastic
    if sequence:
        b, t, c, h, w = x.shape
        frame_grid = grid[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, h, w, 2)
        x = F.grid_sample(
            x.reshape(b * t, c, h, w),
            frame_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).reshape(b, t, c, h, w)
    else:
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # Newly exposed image borders have no known label, rather than known BG.
    support = F.grid_sample(
        torch.ones_like(y[:, None], dtype=x.dtype),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[
        :, 0
    ].long()
    y = y.masked_fill(support == 0, 255)
    if dim == 2 and torch.rand(()) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    return x, y
