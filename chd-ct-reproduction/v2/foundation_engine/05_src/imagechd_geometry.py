"""Author-spacing derived copies; original pixels and headers remain intact."""
from pathlib import Path
import hashlib
import os
import numpy as np
import nibabel as nib


def physical_affine(image, spacing):
    spacing = np.asarray(spacing,dtype=float)
    affine = image.affine.copy()
    norms = np.linalg.norm(affine[:3,:3],axis=0)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or np.any(spacing<=0) or np.any(norms<=0):
        raise ValueError('无效作者间距/源方向')
    affine[:3,:3] = affine[:3,:3]/norms*spacing
    # Retain source directions/origin; these are not independently authenticated
    # scanner coordinates. Only the author's voxel spacings are newly asserted.
    return affine


def save_derived(source, destination, spacing, labels=False):
    image = nib.load(source)
    if image.ndim!=3:
        raise ValueError('只接受3D源文件')
    array = np.asanyarray(image.dataobj)
    if not np.isfinite(array).all():
        raise ValueError('源体积包含非有限值')
    if labels and not np.isin(np.unique(array),np.arange(8)).all():
        raise ValueError('ImageCHD标签只能为0..7')
    affine = physical_affine(image,spacing)
    header = image.header.copy()
    header.set_xyzt_units('mm')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name.replace('.nii.gz','.partial.nii.gz'))
    converted=nib.Nifti1Image(array,affine,header)
    converted.set_qform(affine,code=1);converted.set_sform(affine,code=1)
    nib.save(converted,temporary)
    restored=nib.load(temporary)
    if not np.array_equal(np.asanyarray(restored.dataobj),array) or not np.allclose(restored.affine,affine,rtol=0,atol=1e-5):
        raise ValueError('派生影像写后像素或间距核验失败')
    os.replace(temporary,destination)
    return destination


def derived_image(root, case):
    """Image-only input for held-out inference: no label is opened here."""
    from server_data import digest,read_json,write_json
    out=Path(root)/'04_data/derived/imagechd_inference'/case['case_id']
    signature=dict(source_sha256=digest(case['image']),spacing_mm=case['spacing_mm'],
        implementation_sha256=digest(Path(__file__)))
    if signature['source_sha256']!=case['image_sha256']:
        raise ValueError('推理源影像已变化')
    image=out/'image.nii.gz';marker=out/'geometry.json'
    if marker.exists():
        previous=read_json(marker)
        if previous['signature']!=signature or digest(image)!=previous['derived_sha256']:
            raise ValueError('已有派生输入身份或文件变化')
    else:
        save_derived(case['image'],image,case['spacing_mm'])
        write_json(marker,dict(signature=signature,derived_sha256=digest(image)))
    return image
