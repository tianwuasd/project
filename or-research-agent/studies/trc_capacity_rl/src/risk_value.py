"""Risk-insertion opportunity features; exact feasibility is handled elsewhere."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from environment import Order
from pattern_value import PatternBank
from scenario_environment import patterns
from risk_environment import RISK_CONTEXT_DIM


def conditional_saturation(accepted, capacities, q, delta):
    """P(L_cell=capacity_cell | Z), with independent unit loads given Z.

    Uses truncated Poisson-binomial recursion, not a joint-success probability.
    Floating-point output is for features; it is not a certified risk bound.
    """
    cap = np.asarray(capacities)
    if np.any(cap < 0) or np.any(cap != np.floor(cap)):
        raise ValueError('Nonnegative integer capacities required')
    prior = np.asarray(q, dtype=float)
    transition = (1 - delta) * np.eye(2) + delta * prior[None, :]
    size, degree = cap.size, int(cap.max())
    pmf = np.zeros((2, size, degree + 1))
    pmf[:, :, 0] = 1.
    for usage in accepted:
        usage = np.asarray(usage)
        if usage.shape != (2, *cap.shape) or np.any((usage != 0) & (usage != 1)):
            raise ValueError('Feature recursion requires unit binary occupancy')
        p = (transition @ usage.reshape(2, -1))[:, :, None]
        previous = pmf.copy()
        pmf *= 1 - p
        pmf[:, :, 1:] += p * previous[:, :, :-1]
    return np.take_along_axis(pmf, cap.ravel()[None, :, None].astype(int), axis=2)[:, :, 0]


class RiskPatternBank(PatternBank):
    def __init__(self, config, variant='risk_insertion', delta=.01, distances=(1., 2., 3., 4., 5., 6.)):
        if variant not in ('pattern', 'risk_insertion'):
            raise ValueError('Unknown insertion feature variant')
        if len(config.probabilities) != 2 or not 0 <= delta <= 1:
            raise ValueError('Invalid two-outcome law')
        super().__init__(config, distances)
        self.variant, self.delta = variant, delta
        self.capacities = np.repeat(np.array([config.pad_capacity, config.fleet_capacity,
                                             config.charger_capacity])[:, None], config.slots, axis=1)
        self.prior = np.asarray(config.probabilities)
        transition = (1 - delta) * np.eye(2) + delta * self.prior[None, :]
        modes = [p['scenario_usage'] for slot in range(config.slots) for distance in distances
                 for p in patterns(Order('template', slot, distance), config)]
        if any(np.any((u != 0) & (u != 1)) for u in modes):
            raise ValueError('Insertion bound requires unit template occupancy')
        self.conditional_usage = np.asarray([transition @ u.reshape(2, -1) for u in modes])
        self.conditional_usage = self.conditional_usage.reshape(len(modes), 2 * self.capacities.size)

    def incremental_bounds(self, accepted):
        saturation = conditional_saturation(accepted, self.capacities, self.prior, self.delta)
        return self.conditional_usage @ (saturation * self.prior[:, None]).ravel()

    def values(self, capacity, slot, accepted, remaining):
        if self.variant == 'pattern':
            return super().values(capacity, slot)
        expected = (2, 3, self.config.slots)
        if capacity.shape != expected or np.any(capacity < 0) or remaining < 0:
            raise ValueError('Invalid afterstate')
        result = np.zeros(self.size, dtype=np.float32)
        if len(self.groups):
            ratios = np.where(self.mask, capacity.ravel()[self.indices] / self.units, np.inf)
            modes = np.minimum(1., ratios.min(axis=1))
            bounds = self.incremental_bounds(accepted)
            risk_fraction = np.divide(remaining, bounds, out=np.ones_like(bounds), where=bounds > 0)
            modes *= np.minimum(1., risk_fraction)
            np.maximum.at(result, self.groups, modes)
        result[self.release < slot] = 0.
        return result


class RiskCritic(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.weights = nn.Sequential(nn.Linear(RISK_CONTEXT_DIM, 32), nn.Tanh(), nn.Linear(32, dimension))
        nn.init.zeros_(self.weights[-1].weight)
        nn.init.constant_(self.weights[-1].bias, -4.)
        self.intercept = nn.Sequential(nn.Linear(RISK_CONTEXT_DIM, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x):
        context, features = x[:, :RISK_CONTEXT_DIM], x[:, RISK_CONTEXT_DIM:]
        return (F.softplus(self.weights(context)) * features).sum(dim=1) + self.intercept(context).squeeze(-1)
