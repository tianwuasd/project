import unittest

import torch

from chd_segmentation.losses.partial import imagechd_loss, marginal_loss


class SupervisionTests(unittest.TestCase):
    def test_1282_unknown_probability_can_move_within_background_group(self):
        first = torch.tensor([0.3, 0.2, 0.5]).log()[None, :, None, None, None].requires_grad_()
        second = torch.tensor([0.6, 0.2, 0.2]).log()[None, :, None, None, None].requires_grad_()
        target = torch.zeros((1, 1, 1, 1), dtype=torch.long)
        left = marginal_loss(first, target, [[2]])
        right = marginal_loss(second, target, [[2]])
        self.assertAlmostEqual(left.item(), right.item(), places=6)
        left.backward()
        self.assertTrue(torch.isfinite(first.grad).all())

    def test_missing_myo_is_not_penalized_as_absent(self):
        p = torch.tensor([0.2, 0.05, 0.05, 0.05, 0.05, 0.5, 0.05, 0.05])
        q = torch.tensor([0.6, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05])
        target = torch.zeros((1, 1, 1, 1), dtype=torch.long)
        a = imagechd_loss(p.log()[None, :, None, None, None], target, [[5]])
        b = imagechd_loss(q.log()[None, :, None, None, None], target, [[5]])
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_all_ignored_produces_zero_loss_and_zero_gradient(self):
        logits = torch.randn((1, 11, 2, 2, 2), requires_grad=True)
        value = marginal_loss(logits, torch.full((1, 2, 2, 2), -1), [[]])
        value.backward()
        self.assertEqual(value.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
