import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from learning import ValueNetwork


class LearningTests(unittest.TestCase):
    def test_capacity_monotonicity_holds_by_construction(self):
        torch.manual_seed(2)
        net = ValueNetwork(60, monotone=True)
        low = torch.rand(100, 64)
        high = low.clone(); high[:,4:] += torch.rand(100,60)
        with torch.no_grad():
            self.assertTrue(torch.all(net(high) >= net(low)-1e-6))
