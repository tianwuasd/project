"""Check the port against the authors' bundled Caffe graphs, not another U-Net."""

import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from chd_ct.models.unet import UNet

REF = Path(__file__).resolve().parents[1] / "src/chd_ct/models/reference"


def source_layers(dim):
    name = "2d-phseg_v5-train.prototxt" if dim == 2 else "3dUnet_miccai2016_no_BN.prototxt"
    seen = set()
    for line in (REF / name).read_text().splitlines():
        if "type:" not in line or "name:" not in line:
            continue
        name = re.search(r"name:\s*'([^']+)'", line)[1]
        kind = re.search(r"type:\s*'?([A-Za-z_]+)", line)[1].lower()
        if name in seen or kind not in {
            "convolution",
            "deconvolution",
            "relu",
            "pooling",
            "concat",
            "crop",
            "dropout",
        }:
            continue
        seen.add(name)
        yield kind, re.findall(r"bottom:\s*'([^']+)'", line), re.findall(r"top:\s*'([^']+)'", line), line


def convolutions(model):
    kinds = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    order = []
    handles = [
        m.register_forward_hook(lambda m, x, y: order.append(m))
        for m in model.modules()
        if isinstance(m, kinds)
    ]
    return order, handles


def test_reference_files_are_unmodified():
    manifest = json.loads((REF / "sources.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((REF / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("dim,shape,out", [(2, (572, 572), (388, 388)), (3, (116, 132, 132), (28, 44, 44))])
def test_official_widths_shapes_and_forward_layer_signatures(dim, shape, out):
    with torch.device("meta"):
        model = UNet(dim, 2 if dim == 2 else 3)
        modules, handles = convolutions(model)
        result = model(torch.empty(1, 1, *shape))
    assert tuple(result.shape[2:]) == out
    source = [
        (kind, line) for kind, _, _, line in source_layers(dim) if kind in {"convolution", "deconvolution"}
    ]
    assert len(modules) == len(source)
    for module, (kind, line) in zip(modules, source):
        assert module.out_channels == int(re.search(r"num_output:\s*(\d+)", line)[1])
        assert module.kernel_size == (int(re.search(r"kernel_size:\s*(\d+)", line)[1]),) * dim
        assert module.padding == (0,) * dim
        assert isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)) == (kind == "deconvolution")
    assert not any(
        isinstance(m, (nn.LeakyReLU, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.GroupNorm))
        for m in model.modules()
    )
    assert sum(isinstance(m, nn.Dropout) for m in model.modules()) == (2 if dim == 2 else 0)
    for h in handles:
        h.remove()


def crop_to(x, shape):
    return x[
        (
            slice(None),
            slice(None),
            *(slice((a - b) // 2, (a - b) // 2 + b) for a, b in zip(x.shape[2:], shape)),
        )
    ]


@pytest.mark.parametrize("dim,size", [(2, 188), (3, 92)])
def test_numerical_equivalence_to_independent_source_graph(dim, size):
    torch.set_num_threads(2)
    torch.manual_seed(2)
    model = UNet(dim, 3, base=1).eval()
    x = torch.randn(1, 1, *((size,) * dim))
    order, handles = convolutions(model)
    with torch.no_grad():
        actual = model(x)
    for handle in handles:
        handle.remove()
    convs = iter(order)
    blobs = {"data": x, "d0a": x}
    with torch.no_grad():
        for kind, bottom, top, line in source_layers(dim):
            if kind in {"convolution", "deconvolution"}:
                m = next(convs)
                op = (
                    (F.conv2d if dim == 2 else F.conv3d)
                    if kind == "convolution"
                    else (F.conv_transpose2d if dim == 2 else F.conv_transpose3d)
                )
                y = op(blobs[bottom[0]], m.weight, m.bias, stride=m.stride)
            elif kind == "relu":
                y = blobs[bottom[0]].clamp_min(0)
            elif kind == "pooling":
                y = (F.max_pool2d if dim == 2 else F.max_pool3d)(blobs[bottom[0]], 2, 2, ceil_mode=True)
            elif kind == "crop":
                y = crop_to(blobs[bottom[0]], blobs[bottom[1]].shape[2:])
            elif kind == "concat":
                y = torch.cat([crop_to(blobs[b], blobs[bottom[0]].shape[2:]) for b in bottom], dim=1)
            elif kind == "dropout":
                y = blobs[bottom[0]]  # eval mode
            blobs[top[0]] = y
    torch.testing.assert_close(actual, blobs["score"], rtol=0, atol=0)


@pytest.mark.parametrize("dim,shape", [(2, (1, 7)), (2, (17, 19)), (3, (7, 9, 11))])
def test_grid_adapter_and_gate_preserve_shape_and_gradients(dim, shape):
    from chd_ct.models.grid import GridUNet, mirror_pad

    torch.set_num_threads(2)
    model = GridUNet(dim, 3, base=1, levels=3, spatial_gate=True)
    x = torch.randn(1, 1, *shape, requires_grad=True)
    raw_shape = model.core.input_shape_for_output(shape)
    extended = mirror_pad(x, raw_shape)
    # 2D dropout requires evaluation for deterministic feature comparison.
    model.eval()
    with torch.no_grad():
        expected = crop_to(model.core.features(extended), shape)
        torch.testing.assert_close(model.features(x), expected)
    y = model(x)
    assert tuple(y.shape) == (1, 3, *shape)
    F.cross_entropy(y, torch.ones((1, *shape), dtype=torch.long)).backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.gate.parameters())
    model.gate = None
    torch.testing.assert_close(model(x), model.core.head(model.features(x)))


def test_symmetric_mirror_keeps_original_coordinates():
    import numpy as np

    from chd_ct.models.grid import mirror_pad

    x = torch.arange(6.0).reshape(1, 1, 2, 3)
    actual = mirror_pad(x, (9, 10))[0, 0].numpy()
    expected = np.pad(x[0, 0].numpy(), ((3, 4), (3, 4)), mode="symmetric")
    np.testing.assert_array_equal(actual, expected)


def test_raw_core_rejects_non_lattice_input_and_gate():
    with pytest.raises(ValueError, match="gate"):
        UNet(3, 8, spatial_gate=True)
    model = UNet(3, 8, base=1)
    with pytest.raises(ValueError, match="input|输入"):
        model(torch.zeros(1, 1, 93, 92, 92))


def test_main_factory_uses_author_core_with_switchable_gate(tmp_path):
    import yaml

    from chd_ct.imagechd.config import build_model, load_config
    from chd_ct.models.unet import ARCHITECTURE

    root = REF.parents[3]
    config = load_config(root / "configs/chd.yaml")
    assert config["architecture"] == ARCHITECTURE
    spec = config["stages"]["crop64"]
    model = build_model("crop64", {**spec, "base": 1})
    assert isinstance(model.core, UNet) and model.gate is not None
    assert build_model("crop64", {**spec, "base": 1, "spatial_gate": False}).gate is None
    config["stages"]["crop64"]["spatial_gate"] = "false"
    file = tmp_path / "invalid.yaml"
    file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="spatial_gate"):
        load_config(file)


def test_old_collection_is_rejected_with_retraining_guidance(tmp_path):
    from chd_ct.imagechd.checkpoints import read_collection
    from chd_ct.imagechd.config import STAGES

    (tmp_path / "models.json").write_text(
        json.dumps(
            {"format": "imagechd7-multistage-v1", "status": "complete", "stages": dict.fromkeys(STAGES, {})}
        )
    )
    with pytest.raises(ValueError, match="重新训练"):
        read_collection(tmp_path, allow_smoke=True)
