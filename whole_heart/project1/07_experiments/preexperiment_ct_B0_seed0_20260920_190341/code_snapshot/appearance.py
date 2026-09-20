"""依据公开算法独立实现的外观增强；仅改图像，不改分割标签。

参数与 CARE 作者 commit 1e151b7 核对。Bias Field 是粗网格乘性场；
Bezier 为逐通道单调灰度映射。数学行为可和本地参考代码逐随机种子比较。
"""
import math
import numpy as np
import torch
from torch.nn.functional import interpolate
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class BiasField(ImageOnlyTransform):
    def __init__(self, magnitude=0.25):
        super().__init__()
        self.magnitude = magnitude

    def _apply_to_image(self, img, **params):
        grid = np.random.uniform(1 - self.magnitude, 1 + self.magnitude, (3, 3, 3))
        field = interpolate(torch.from_numpy(grid).float()[None, None], size=img.shape[1:], mode="trilinear", align_corners=True)[0, 0]
        return img * field.to(device=img.device, dtype=img.dtype)


class MonotoneBezier(ImageOnlyTransform):
    def __init__(self, lut_size=1000):
        super().__init__()
        t = np.linspace(0, 1, lut_size)
        # 与参考实现保持相同参数方向；x/y 分别排序后得到单调曲线。
        self.basis = np.array([math.comb(3, i) * t ** (3-i) * (1-t) ** i for i in range(4)], dtype=np.float32)

    def _apply_to_image(self, img, **params):
        data = img.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(data).all():
            raise ValueError("增强输入包含 NaN 或 Inf")
        result = np.empty_like(data)
        for c, channel in enumerate(data):
            low, high = float(channel.min()), float(channel.max())
            if high - low < 1e-6:
                result[c] = channel
                continue
            x = np.array([0, np.random.uniform(0, 1), np.random.uniform(0, 1), 1], dtype=np.float32)
            y = np.array([0, np.random.uniform(0, 1), np.random.uniform(0, 1), 1], dtype=np.float32)
            x, y = np.sort(x @ self.basis), np.sort(y @ self.basis)
            normalized = (channel - low) / (high - low)
            result[c] = np.interp(normalized.ravel(), x, y).reshape(channel.shape) * (high - low) + low
        return torch.from_numpy(result).to(device=img.device, dtype=img.dtype)
