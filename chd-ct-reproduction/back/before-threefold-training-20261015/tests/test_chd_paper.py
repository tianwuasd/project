"""Checks against Figure 4 / Table 3 of Xu et al., MedIA 90 (2023)."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from chd_ct.imagechd.config import build_model, load_config
from chd_ct.models.chd_unet import ARCHITECTURE, CHDUNet

ROOT = Path(__file__).resolve().parents[1]


def test_table3_defaults_and_installed_presets():
    config = load_config(ROOT / "configs/chd.yaml")
    assert config == load_config(ROOT / "src/chd_ct/presets/chd.yaml")
    assert config["architecture"] == ARCHITECTURE
    assert config["learning_rate"] == 0.0002
    for stage in ("crop64", "all64", "crop128", "all128"):
        spec = config["stages"][stage]
        size = 64 if stage.endswith("64") else 128
        assert (spec["size"], spec["base"], spec["levels"], spec["batch"], spec["epochs"]) == (
            size,
            64 if size == 64 else 48,
            5,
            4,
            15,
        )
        assert spec["spatial_gate"] is True
        assert spec["activation_checkpoint"] is True
    spec = config["stages"]["blood2d"]
    assert tuple(spec[k] for k in ("size", "base", "levels", "batch", "epochs")) == (512, 32, 8, 4, 12)
    assert spec["spatial_gate"] is False
    assert spec["activation_checkpoint"] is True
    spec = config["stages"]["blood_lstm"]
    assert tuple(spec[k] for k in ("size", "base", "layers", "sequence", "batch", "epochs")) == (
        512,
        32,
        2,
        7,
        1,
        3,
    )


@pytest.mark.parametrize(
    "stage,dim,base,levels,size",
    [("crop64", 3, 64, 5, 64), ("all128", 3, 48, 5, 128), ("blood2d", 2, 32, 8, 512)],
)
def test_full_paper_shapes_and_layers_on_meta(stage, dim, base, levels, size):
    config = load_config(ROOT / "configs/chd.yaml")
    with torch.device("meta"):
        model = build_model(stage, config["stages"][stage])
        x = torch.empty(1, 1, *((size,) * dim))
        result = model(x)
        features = model.features(x)
    classes = 3 if dim == 2 else 8
    assert isinstance(model, CHDUNet)
    assert result.shape == (1, classes, *((size,) * dim))
    assert features.shape == (1, base, *((size,) * dim))
    assert len(model.core.encoder) == levels
    assert len(model.core.decoder) == levels - 1
    norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
    conv = nn.Conv3d if dim == 3 else nn.Conv2d
    for i, block in enumerate(model.core.encoder):
        assert [m.out_channels for m in block if isinstance(m, conv)] == [base * 2**i] * 2
    for block in [*model.core.encoder, *model.core.decoder]:
        assert [type(m) for m in block] == [conv, norm, nn.LeakyReLU] * 2
        assert all(m.padding == (1,) * dim for m in block if isinstance(m, conv))
    assert not any(isinstance(m, nn.Dropout) for m in model.modules())
    if dim == 3:
        assert isinstance(model.gate.pool, nn.MaxPool3d)
        assert model.gate.pool.kernel_size == 4
        convolutions = [m for m in model.gate.network if isinstance(m, nn.Conv3d)]
        assert [(m.in_channels, m.out_channels) for m in convolutions] == [(8, 64), (64, 64), (64, 8)]
        assert all(m.kernel_size == (3, 3, 3) for m in convolutions)
        assert isinstance(model.gate.up, nn.ConvTranspose3d)
        assert model.gate.up.kernel_size == model.gate.up.stride == (4, 4, 4)


@pytest.mark.parametrize("dim,shape,gate", [(2, (17, 19), False), (3, (17, 19, 21), True)])
def test_paper_core_odd_grids_and_gate_gradients(dim, shape, gate):
    torch.set_num_threads(2)
    model = CHDUNet(dim, 3, base=2, levels=3, spatial_gate=gate)
    x = torch.randn(1, 1, *shape, requires_grad=True)
    logits = model(x)
    assert logits.shape == (1, 3, *shape)
    F.cross_entropy(logits, torch.ones((1, *shape), dtype=torch.long)).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if gate:
        assert all(p.grad.abs().sum() > 0 for p in model.gate.parameters())


def test_dice_weights_do_not_reweight_cross_entropy():
    from chd_ct.imagechd.losses import masked_loss

    logits = torch.tensor([[[[1.0, 2.0, -1.0]], [[2.0, -1.0, 1.0]], [[-1.0, 1.0, 2.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 2, 255]]])
    weights = torch.tensor([1.0, 2.0, 16.0])
    p = logits.softmax(1)[0, :, 0, :2]
    y = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    dice = (2 * (p * y).sum(1) + 1e-6) / (p.sum(1) + y.sum(1) + 1e-6)
    expected = 1 - (dice * weights).sum() / weights.sum() + F.cross_entropy(logits, target, ignore_index=255)
    loss = masked_loss(logits, target, weights.tolist())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[..., 2].abs().sum() == 0


def test_2d_and_sequence_augmentation_share_rotation_without_scaling(monkeypatch):
    from chd_ct.imagechd.augment import augment_batch

    captured = []
    original = F.affine_grid

    def capture(theta, *args, **kwargs):
        captured.append(theta.detach().clone())
        return original(theta, *args, **kwargs)

    monkeypatch.setattr(F, "affine_grid", capture)
    x = torch.arange(32 * 32).reshape(1, 1, 32, 32).float()
    y = (x[:, 0] > 500).long()
    torch.manual_seed(23)
    single, label = augment_batch(x, y)
    torch.manual_seed(23)
    sequence, seq_label = augment_batch(x[:, None].repeat(1, 7, 1, 1, 1), y)
    assert len(captured) == 2
    assert not torch.equal(single, x)  # Rotation occurs, not merely a flip.
    for frame in sequence.unbind(1):
        torch.testing.assert_close(frame, single)
    torch.testing.assert_close(label, seq_label)
    for theta in captured:
        rotation = theta[:, :, :2]
        torch.testing.assert_close(rotation.transpose(1, 2) @ rotation, torch.eye(2)[None])


def test_legacy_config_is_rejected():
    with pytest.raises(ValueError, match="architecture"):
        load_config(ROOT / "configs/chd-author-unet.yaml")


def test_position_branch_is_normalized_pointwise_product():
    from chd_ct.models.chd_unet import PositionCorrelation

    torch.set_num_threads(2)
    branch = PositionCorrelation(3)
    logits = torch.randn(1, 3, 8, 8, 8)
    captured = []
    handle = branch.up.register_forward_hook(lambda module, inputs, output: captured.append(output))
    with torch.no_grad():
        actual = branch(logits).softmax(1)
    handle.remove()
    product = logits.softmax(1) * captured[0].sigmoid()
    torch.testing.assert_close(actual, product / product.sum(1, keepdim=True))


@pytest.mark.parametrize("sequence", [False, True])
def test_rotated_borders_and_existing_unknown_labels_remain_ignored(sequence):
    from chd_ct.imagechd.augment import augment_batch

    x = torch.ones(1, 1, 32, 32)
    y = torch.ones(1, 32, 32, dtype=torch.long)
    y[:, 14:18, 14:18] = 255
    torch.manual_seed(23)
    _, result = augment_batch(x[:, None].repeat(1, 7, 1, 1, 1) if sequence else x, y)
    assert set(result.unique().tolist()) == {1, 255}
    assert (result[:, :2] == 255).any()
    assert (result[:, 12:20, 12:20] == 255).any()


@pytest.mark.parametrize("dim,gate", [(2, False), (3, True)])
def test_activation_checkpoint_preserves_batch_loss_and_parameter_gradients(dim, gate):
    from chd_ct.imagechd.losses import masked_loss

    torch.set_num_threads(2)
    torch.manual_seed(31)
    reference = CHDUNet(dim, 3, base=2, levels=3, spatial_gate=gate)
    efficient = CHDUNet(dim, 3, base=2, levels=3, spatial_gate=gate, activation_checkpoint=True)
    efficient.load_state_dict(reference.state_dict())
    x = torch.randn(2, 1, *((16,) * dim))
    target = torch.randint(0, 3, (2, *((16,) * dim)))
    actual, expected = efficient(x), reference(x)
    torch.testing.assert_close(actual, expected)
    a_loss = masked_loss(actual, target, [1, 2, 16])
    b_loss = masked_loss(expected, target, [1, 2, 16])
    torch.testing.assert_close(a_loss, b_loss)
    a_loss.backward()
    b_loss.backward()
    for a, b in zip(efficient.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
