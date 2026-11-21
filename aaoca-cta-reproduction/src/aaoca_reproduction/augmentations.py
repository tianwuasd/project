import random

import numpy as np
import torch
import torchio as tio
from monai.transforms import CastToTyped, Compose, MapTransform, OneOf
from monai.transforms import RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd


def scale_range(image, low, high):
    span = float(image.max() - image.min())
    if span == 0 or not np.isfinite(span):
        raise ValueError("作者增强的逐例归一化遇到常量或非有限输入")
    return (image - image.min()) / span * (high - low) + low


def author_resize(image, shape):
    ratios = [image.shape[i] / shape[i] for i in range(3)]
    return tio.Resample(target=ratios, image_interpolation="bspline")(image[None])[0]


def step_core(image, cutoff, intensity):
    result = image.copy()
    if random.random() < 0.5:
        result[:, :, cutoff:] += intensity
    else:
        result[:, :, :cutoff] += intensity
    return result


def motion_core(image, cutoff, move, original_shape):
    result = image.copy()
    if random.random() < 0.5:
        result[:, :result.shape[1] - move, cutoff:] = result[:, move:, cutoff:]
        result = result[:, :result.shape[1] - move, :]
    else:
        result[:, move:, cutoff:] = result[:, :result.shape[1] - move, cutoff:]
        result = result[:, move:, :]
    return author_resize(result, original_shape)


class AuthorArtifact(MapTransform):
    def __init__(self, kind, probability):
        super().__init__(keys=["img"])
        self.kind, self.probability = kind, probability

    def __call__(self, data, lazy=None):
        if random.random() >= self.probability:
            return data
        image = data["img"].copy().astype(np.float32)
        low, high = image.min(), image.max()
        shape = image.shape
        if self.kind in ("step", "combined"):
            cutoff = random.randint(0, shape[2] - 1)
            mean = image.mean()
            if self.kind == "combined":
                move = random.randint(int(shape[1] * 0.02), int(shape[1] * 0.1) - 1)
            intensity = random.uniform(0.2 * mean, 0.6 * mean)
            intensity = intensity if random.random() < 0.5 else -intensity
            image = step_core(image, cutoff, intensity)
            if self.kind == "combined":
                image = motion_core(image, cutoff, move, shape)
        elif self.kind == "motion":
            move = random.randint(int(shape[1] * 0.01), int(shape[1] * 0.05) - 1)
            cutoff = random.randint(int(shape[2] * 0.05), int(shape[2] * 0.2) - 1)
            cutoff = cutoff if random.random() < 0.5 else shape[2] - cutoff
            image = motion_core(image, cutoff, move, shape)
        else:
            raise ValueError("未知作者伪影类型")
        data["img"] = scale_range(image, low, high)
        return data


class AuthorAugmentation:
    def __init__(self, probability=0.15):
        self.monai = Compose([OneOf([
            RandGaussianNoised(keys=["img"], std=0.01, prob=probability),
            RandGaussianSmoothd(keys=["img"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15),
                                sigma_z=(0.5, 1.15), prob=probability),
            RandAdjustContrastd(keys=["img"], prob=probability),
            AuthorArtifact("step", probability), AuthorArtifact("motion", probability),
            AuthorArtifact("combined", probability)]), CastToTyped(keys=["img"], dtype=(np.float32,))])
        self.torchio = tio.OneOf([tio.RandomNoise(mean=0, std=0.02), tio.RandomGamma(log_gamma=0.3),
                                  tio.RandomBlur(std=(2, 2, 2))], p=probability)

    def __call__(self, array):
        if random.random() < 0.5:
            # 每样本从已保存的NumPy状态派生MONAI种子，避免依赖未保存的内部随机状态。
            self.monai.set_random_state(seed=int(np.random.randint(0, 2**32, dtype=np.uint32)))
            image = self.monai([{"img": array, "label": -1}])[0]["img"]
            return torch.tensor(np.asarray(image), dtype=torch.float32)
        image = self.torchio(torch.tensor(array, dtype=torch.float32)[None]).squeeze(0)
        span = image.max() - image.min()
        if not torch.isfinite(span) or span.item() == 0:
            raise ValueError("TorchIO作者分支归一化遇到常量或非有限输入")
        return (image - image.min()) / span
