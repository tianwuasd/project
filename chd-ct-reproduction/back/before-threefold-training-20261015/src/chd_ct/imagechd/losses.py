"""Weighted Dice plus cross entropy; both terms ignore label 255.

Section 4.3 assigns class weights to Dice. CE is unweighted and the terms
have equal coefficients because the paper does not specify other values.
"""

from torch.nn import functional as F


def masked_loss(logits, target, weights=None):
    valid = target != 255
    if not valid.any():
        raise ValueError("没有可监督的体素")
    weights = logits.new_tensor(weights or [1] + [2] * (logits.shape[1] - 1))
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe, logits.shape[1]).movedim(-1, 1).to(logits) * valid[:, None]
    probabilities = logits.softmax(1) * valid[:, None]
    axes = (0, *range(2, logits.ndim))
    dice = (2 * (probabilities * onehot).sum(axes) + 1e-6) / (
        probabilities.sum(axes) + onehot.sum(axes) + 1e-6
    )
    return 1 - (dice * weights).sum() / weights.sum() + F.cross_entropy(logits, target, ignore_index=255)
