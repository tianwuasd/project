import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..stunet.streaming import tiles
from .network import Decoder, Residual


class LayerInference:
    def __init__(self, workspace, device):
        self.workspace, self.device = workspace, device

    def sections(self, shape, channels):
        return tiles(shape, max(1, self.workspace.block_bytes // (max(1, channels) * 32)))

    def tensor(self, value, section):
        return torch.from_numpy(np.array(value[(slice(None), *section)], copy=True))[None].to(self.device)

    def save(self, target, section, value):
        result = value[0].float().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('定位器分层验证出现非有限数值')
        target[(slice(None), *section)] = result

    def unary(self, value, function):
        result = self.workspace.create('layer', value.shape, np.float32)
        for section in self.sections(value.shape[1:], value.shape[0]):
            with self.workspace.boundary(), torch.inference_mode():
                self.save(result, section, function(self.tensor(value, section)))
            self.workspace.release(value)
        return result

    def conv(self, module, value):
        shape = tuple((n + 2*p - d*(k-1)-1)//s+1 for n, p, d, k, s in
                      zip(value.shape[1:], module.padding, module.dilation, module.kernel_size, module.stride))
        result = self.workspace.create('layer', (module.out_channels, *shape), np.float32)
        for section in self.sections(shape, max(value.shape[0] * math.prod(module.stride), module.out_channels)):
            with self.workspace.boundary(), torch.inference_mode():
                begin = [t.start*s-p for t, s, p in zip(section, module.stride, module.padding)]
                end = [(t.stop-1)*s-p+d*(k-1)+1 for t, s, p, d, k in
                       zip(section, module.stride, module.padding, module.dilation, module.kernel_size)]
                src = tuple(slice(max(0, a), min(n, b)) for a, b, n in zip(begin, end, value.shape[1:]))
                padding = tuple(v for a, b, n in reversed(list(zip(begin, end, value.shape[1:])))
                                for v in (max(0, -a), max(0, b-n)))
                x = F.pad(self.tensor(value, src), padding)
                out = F.conv3d(x, module.weight, module.bias, module.stride, 0, module.dilation, module.groups)
                self.save(result, section, out)
                del x, out
            self.workspace.release(value)
        return result

    def norm(self, module, value):
        groups = module.num_groups
        width = value.shape[0] // groups
        means, second = np.zeros(groups), np.zeros(groups)
        count = 0
        # GroupNorm跨整个病例归约，分块只改变求和次序。
        for section in self.sections(value.shape[1:], value.shape[0]):
            with self.workspace.boundary():
                x = np.array(value[(slice(None), *section)], dtype=np.float64).reshape(groups, -1)
                local_count = x.shape[1]
                local_mean = x.mean(1)
                delta = local_mean - means
                local_second = ((x-local_mean[:, None])**2).sum(1)
                second += local_second + delta**2 * count * local_count / (count+local_count)
                means += delta * local_count / (count+local_count)
                count += local_count
                del x
            self.workspace.release(value)
        mean = torch.as_tensor(np.repeat(means, width), dtype=torch.float32, device=self.device)[None, :, None, None, None]
        inv = torch.as_tensor(np.repeat(1/np.sqrt(second/count+module.eps), width), dtype=torch.float32,
                              device=self.device)[None, :, None, None, None]
        def normalize(x):
            result = (x-mean)*inv
            if module.affine:
                result = result*module.weight[None, :, None, None, None] + module.bias[None, :, None, None, None]
            return result
        return self.unary(value, normalize)

    def interpolate(self, value, shape):
        result = self.workspace.create('layer', (value.shape[0], *shape), np.float32)
        for section in self.sections(shape, value.shape[0]):
            with self.workspace.boundary(), torch.inference_mode():
                axes = [(np.arange(t.start, t.stop, dtype=np.float64)+.5)*old/new-.5
                        for t, old, new in zip(section, value.shape[1:], shape)]
                low = [max(0, int(np.floor(a.min()))) for a in axes]
                high = [min(n, int(np.floor(a.max()))+2) for a, n in zip(axes, value.shape[1:])]
                src = tuple(slice(a, b) for a, b in zip(low, high))
                grid_axes = [torch.as_tensor(2*(a-l+.5)/(h-l)-1, dtype=torch.float32, device=self.device)
                             for a, l, h in zip(axes, low, high)]
                d, h, w = torch.meshgrid(*grid_axes, indexing='ij')
                grid = torch.stack((w, h, d), -1)[None]
                out = F.grid_sample(self.tensor(value, src), grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
                self.save(result, section, out)
                del grid, out, d, h, w
            self.workspace.release(value)
        return result

    def combine(self, left, right, concatenate=False, activate=False):
        channels = left.shape[0]+right.shape[0] if concatenate else left.shape[0]
        result = self.workspace.create('layer', (channels, *left.shape[1:]), np.float32)
        for section in self.sections(left.shape[1:], channels):
            with self.workspace.boundary(), torch.inference_mode():
                a, b = self.tensor(left, section), self.tensor(right, section)
                out = torch.cat((a, b), 1) if concatenate else a+b
                self.save(result, section, F.silu(out) if activate else out)
                del a, b, out
            self.workspace.release(left, right)
        return result

    def call(self, module, value):
        if isinstance(module, nn.Identity):
            return value
        if isinstance(module, nn.Sequential):
            current = value
            for layer in module:
                result = self.call(layer, current)
                if current is not value and result is not current:
                    self.workspace.drop(current)
                current = result
            return current
        if isinstance(module, Residual):
            body, skip = self.call(module.body, value), self.call(module.skip, value)
            result = self.combine(body, skip, activate=True)
            self.workspace.drop(body)
            if skip is not value:
                self.workspace.drop(skip)
            return result
        if isinstance(module, nn.Conv3d):
            return self.conv(module, value)
        if isinstance(module, nn.GroupNorm):
            return self.norm(module, value)
        if isinstance(module, nn.SiLU):
            return self.unary(value, F.silu)
        raise TypeError('定位器验证不支持的层：' + type(module).__name__)

    def decoder(self, module, value, skip):
        if not isinstance(module, Decoder):
            raise TypeError('需要定位器Decoder')
        enlarged = self.interpolate(value, skip.shape[1:])
        projected = self.call(module.project, enlarged)
        self.workspace.drop(enlarged)
        joined = self.combine(projected, skip, concatenate=True)
        self.workspace.drop(projected)
        result = self.call(module.block, joined)
        self.workspace.drop(joined)
        return result


def localizer_forward(model, image, valid, device, workspace):
    shape = tuple((n+31)//32*8 for n in image.shape[1:])
    downsampled = workspace.create('layer', (2, *shape), np.float32)
    inference = LayerInference(workspace, device)
    for section in inference.sections(shape, 128):
        with workspace.boundary(), torch.inference_mode():
            axes = [np.arange(t.start*4, t.stop*4) for t in section]
            indices = np.ix_(*(a.clip(0, n-1) for a, n in zip(axes, image.shape[1:])))
            x = np.array(image[(slice(None), *indices)], copy=True)
            mask = np.array(valid[indices], copy=True)
            for axis, (positions, n) in enumerate(zip(axes, image.shape[1:])):
                sizes = [1, 1, 1]
                sizes[axis] = len(positions)
                mask &= (positions < n).reshape(sizes)
            x = F.avg_pool3d(torch.from_numpy(x)[None], 4, 4)
            mask = F.avg_pool3d(torch.from_numpy(mask)[None, None].float(), 4, 4)
            downsampled[(slice(None), *section)] = torch.cat((x, mask), 1)[0].numpy()
        workspace.release(image, valid)
    l0 = inference.call(model.l0, downsampled)
    workspace.drop(downsampled)
    l1 = inference.call(model.l1, l0)
    l2 = inference.call(model.l2, l1)
    l3 = inference.call(model.l3, l2)
    d2 = inference.decoder(model.d2, l3, l2)
    workspace.drop(l3)
    d1 = inference.decoder(model.d1, d2, l1)
    workspace.drop(d2)
    workspace.drop(l1)
    d0 = inference.decoder(model.d0, d1, l0)
    workspace.drop(d1)
    workspace.drop(l0)
    logits = inference.call(model.head, d0)
    workspace.drop(d0)
    return logits, l2
