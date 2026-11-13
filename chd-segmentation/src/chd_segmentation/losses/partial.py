import torch
from torch.nn import functional as F


def _target(logits, target):
    if target.ndim == logits.ndim:
        if target.shape[1] != 1:
            raise ValueError("标签必须只有一个通道")
        target = target[:, 0]
    if target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("标签与模型输出尺寸不一致")
    if target.is_floating_point() and not torch.equal(target, target.round()):
        raise ValueError("标签包含非整数")
    if not ((target >= -1) & (target < logits.shape[1])).all():
        raise ValueError("标签超出类别范围")
    return target.long()


def marginal_loss(logits, target, unknown_classes, smooth=1e-5):
    target = _target(logits, target)
    if len(unknown_classes) != len(logits):
        raise ValueError("逐病例监督信息数量不一致")
    results = []
    for values, label, missing in zip(logits, target, unknown_classes):
        unknown = sorted(set(missing))
        if not set(unknown) <= set(range(1, logits.shape[1])):
            raise ValueError("未知类别编号无效")
        valid = label != -1
        if not valid.any():
            continue
        label = label[valid]
        if any((label == c).any() for c in unknown):
            raise ValueError("标签与未知类别声明矛盾")
        values = values.float().flatten(1)[:, valid.flatten()]
        numerator = values.gather(0, label[None])[0]
        merged = torch.logsumexp(values[[0, *unknown]], dim=0)
        ce = (torch.logsumexp(values, dim=0) - torch.where(label == 0, merged, numerator)).mean()
        probabilities = values.softmax(0)
        groups = [[0, *unknown]] + [[c] for c in range(1, len(values)) if c not in unknown]
        dice = []
        for group in groups:
            probability = probabilities[group].sum(0)
            truth = (label == group[0]).to(probability)
            dice.append((2 * (probability * truth).sum() + smooth) /
                        (probability.sum() + truth.sum() + smooth))
        results.append(ce + 1 - torch.stack(dice).mean())
    return torch.stack(results).mean() if results else logits.sum() * 0


def imagechd_loss(logits, target, unknown_classes, smooth=1e-5):
    target = _target(logits, target)
    if logits.shape[1] != 8 or len(unknown_classes) != len(logits):
        raise ValueError("ImageCHD 需要八类输出与逐病例监督信息")
    if any(set(missing) not in (set(), {5}) for missing in unknown_classes):
        raise ValueError("ImageCHD 配方仅支持完整标注或缺少 MYO")
    available = logits.new_ones((len(logits), 7), dtype=torch.bool)
    for i, missing in enumerate(unknown_classes):
        if missing:
            available[i, 4] = False
            if (target[i] == 5).any():
                raise ValueError("缺少 MYO 的病例出现 MYO 标签")
    values = logits if logits.dtype == torch.float64 else logits.float()
    valid = target != -1
    safe = target.clamp_min(0)
    logp = F.log_softmax(values, dim=1)
    selected = logp.gather(1, safe[:, None])[:, 0]
    missing_myo = (~available[:, 4]).view(-1, *([1] * (target.ndim - 1)))
    merged = torch.logsumexp(logp[:, [0, 5]], dim=1)
    ce = (-torch.where((target == 0) & missing_myo, merged, selected) * valid).sum() / valid.sum().clamp_min(1)
    truth = F.one_hot(safe, 8).movedim(-1, 1)[:, 1:]
    probability = logp[:, 1:].exp()
    axes = tuple(range(2, logits.ndim))
    intersect = (probability * truth * valid[:, None]).sum(axes) * available
    predicted = (probability * valid[:, None]).sum(axes) * available
    reference = (truth * valid[:, None]).sum(axes) * available
    active = (available & valid.flatten(1).any(1)[:, None]).any(0)
    dice = (2 * intersect.sum(0) + smooth) / (predicted.sum(0) + reference.sum(0) + smooth)
    return ce - (dice * active).sum() / active.sum().clamp_min(1)


def deep_supervision_loss(outputs, target, unknown, dataset):
    outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    weights = [1 / 2 ** index for index in range(len(outputs))]
    if len(weights) > 1:
        weights[-1] = 0
    loss_fn = marginal_loss if dataset == "chd1282" else imagechd_loss
    targets = [F.interpolate(target[:, None].float(), size=value.shape[2:], mode="nearest")[:, 0].long()
               for value in outputs]
    return sum(weight * loss_fn(value, label, unknown) for weight, value, label in
               zip(weights, outputs, targets) if weight) / sum(weights)
