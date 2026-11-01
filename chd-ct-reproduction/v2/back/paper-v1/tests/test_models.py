import torch

from chd_ct.models import BiConvLSTM, UNet
from chd_ct.training.losses import dice_ce_loss


def test_unet_odd_shape_and_loss_gradient():
    torch.set_num_threads(2)
    model = UNet(3, 11, base=2, levels=3, spatial_gate=True)
    x = torch.randn(1, 1, 17, 19, 21)
    y = model(x)
    assert y.shape == (1, 11, 17, 19, 21)
    loss = dice_ce_loss(y, torch.zeros((1, 17, 19, 21), dtype=torch.long), [1] + [2] * 10)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_biconvlstm_uses_future_and_past():
    torch.manual_seed(12)
    model = BiConvLSTM(channels=2, layers=2, classes=3)
    x = torch.randn(1, 7, 2, 8, 8)
    y = model(x)
    assert y.shape == (1, 3, 8, 8)
    for i in (0, 6):
        changed = x.clone()
        changed[:, i] += 4
        assert not torch.allclose(y, model(changed))
