import torch
from torch.nn import functional as F


def dice_ce_loss(logits, target, weights):
    weights = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    if weights.numel() != logits.shape[1] or (weights <= 0).any():
        raise ValueError("Need one positive weight per class")
    probs = logits.softmax(1)
    onehot = F.one_hot(target.long(), logits.shape[1]).movedim(-1, 1).to(probs)
    axes = (0,) + tuple(range(2, logits.ndim))
    intersection = (probs * onehot).sum(axes)
    dice = (2 * intersection + 1e-6) / (probs.sum(axes) + onehot.sum(axes) + 1e-6)
    return 1 - (weights * dice).sum() / weights.sum() + F.cross_entropy(logits, target.long(), weight=weights)
