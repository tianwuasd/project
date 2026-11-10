from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
import torch
from torch.nn import functional as F

from ..contracts import sha256_file


WEIGHT_SHA256 = '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
SPACING_MM = 1.5
WIDTH = 128


def load_model(path, device):
    import medim

    if version('medim') != '0.1.2' or sha256_file(path) != WEIGHT_SHA256:
        raise ValueError('需要 MedIM 0.1.2 和固定版本的官方 Turbo 权重')
    model = medim.create_model('SAM-Med3D', pretrained=False)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'], strict=True)
    return model.to(device).eval()


def validate_point(point, shape):
    if (not isinstance(point, list) or len(point) != 3
            or any(type(x) is not int or not 0 <= x < n for x,n in zip(point,shape))):
        raise ValueError('提示点必须是原 CT 网格内的零起始 XYZ 体素索引')
    return np.asarray(point, dtype=np.float64)


def prompt_patch(values, spacing, point):
    point = validate_point(point, values.shape)
    # Dataset1282 的 Excel 工作网格已固定为 XYZ 正对角矩阵。
    # 直接采样局部网格，避免先写一份完整的 1.5 mm CT。
    scale = SPACING_MM / np.asarray(spacing, dtype=np.float64)
    center = np.full(3, WIDTH // 2)
    offset = point - center * scale
    patch = affine_transform(values, np.diag(scale), offset=offset,
                             output_shape=(WIDTH,)*3, order=1, prefilter=False)
    positive = patch[patch > 0]
    if positive.size < 2:
        raise ValueError('提示点周围没有足够的正灰度体素')
    std = positive.std(ddof=1, dtype=np.float64)
    if not np.isfinite(std) or std <= 0:
        raise ValueError('提示块的归一化方差无效')
    patch -= positive.mean(dtype=np.float64)
    patch /= std
    return patch, center, scale, offset


def restore_mask(mask, shape, scale, offset):
    return affine_transform(mask.astype(np.uint8, copy=False), np.diag(1 / scale),
                            offset=-offset / scale, output_shape=shape,
                            order=0, mode='grid-constant', prefilter=False)


def load_ct(row):
    path = Path(row['image'])
    if sha256_file(path) != row['sha256']['image']:
        raise ValueError('原 CT 摘要发生变化')
    image = nib.load(path)
    if tuple(image.shape) != tuple(row['shape']):
        raise ValueError('原 CT 尺寸发生变化')
    values = image.get_fdata(dtype=np.float32, caching='unchanged')
    if not np.isfinite(values).all():
        raise ValueError('原 CT 含非有限数值')
    return values


@torch.inference_mode()
def predict(model, values, spacing, point, device):
    patch, center, scale, offset = prompt_patch(values, spacing, point)
    image = torch.from_numpy(patch)[None, None].to(device)
    coords = torch.as_tensor(center, device=device, dtype=torch.float32)[None,None]
    labels = torch.ones((1, 1), device=device, dtype=torch.long)
    previous = torch.zeros((1, 1, *((WIDTH // 4,)*3)), device=device)
    embedding = model.image_encoder(image)
    sparse, dense = model.prompt_encoder(points=(coords,labels), boxes=None, masks=previous)
    low_res, _ = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
    logits = F.interpolate(low_res, size=(WIDTH,)*3, mode='trilinear', align_corners=False)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('SAM-Med3D 输出非有限数值')
    mask = (logits[0,0] > 0).cpu().numpy()
    touches_edge = any(np.take(mask, [0, WIDTH-1], axis=axis).any() for axis in range(3))
    result = restore_mask(mask, values.shape, scale, offset)
    return result, dict(prompt_xyz=point, patch_prompt_xyz=center.tolist(),
                        patch_origin_mm=(offset*np.asarray(spacing)).tolist(),
                        spacing_mm=SPACING_MM, patch_voxels=WIDTH, touches_patch_edge=touches_edge)
