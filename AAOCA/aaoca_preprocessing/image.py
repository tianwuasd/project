"""Geometry adapted from tools/research_ct/research_preprocess.py's validated grid.

The CPU slab kernel implements the same linear, align_corners=False sampling.
No nnU-Net planning, foreground statistics, training or GPU initialization.
"""
import gzip
import hashlib
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from .config import Config
from .control import Control


def normalize_hu(array):
    data = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(data).all():
        raise ValueError('CT contains nonfinite pixels (NaN/Inf)')
    np.clip(data, -1000, 1000, out=data)
    data /= np.float32(1000)
    return data


def output_grid(shape, affine, target_spacing=None):
    shape = np.asarray(shape, dtype=np.int64)
    affine = np.asarray(affine, dtype=float)
    if shape.shape != (3,) or (shape <= 0).any():
        raise ValueError('expected a nonempty 3D volume')
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError('invalid or singular spatial affine')
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError('invalid affine bottom row')
    if target_spacing is None:
        size = shape.copy()
    else:
        target = np.asarray(target_spacing, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError('target spacing must contain three positive finite values')
        rounded = np.rint(shape * nib.affines.voxel_sizes(affine) / target)
        if not np.isfinite(rounded).all() or (rounded < 1).any() or (rounded > 32767).any():
            raise ValueError('requested grid has empty or unsupported NIfTI-1 dimensions')
        size = rounded.astype(np.int64)
    ratio = shape / size
    mapping = np.eye(4)
    mapping[:3, :3] = np.diag(ratio)
    mapping[:3, 3] = (ratio - 1) / 2
    return tuple(map(int, size)), affine @ mapping, mapping


def inspect_image(path, config: Config):
    image = nib.load(str(path))
    if len(image.shape) != 3 or any(n < 1 for n in image.shape):
        raise ValueError('expected one nonempty 3D CT; split phases explicitly before preprocessing')
    if image.get_data_dtype().kind not in 'iuf':
        raise ValueError('CT pixels must be real-valued scalar numbers')
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('spatial units must explicitly be mm; unknown units are not inferred')
    sform, scode = image.get_sform(coded=True)
    qform, qcode = image.get_qform(coded=True)
    if not scode and not qcode:
        raise ValueError('missing coded spatial transform')
    if scode and qcode and not np.allclose(sform, qform, rtol=0, atol=1e-3):
        raise ValueError('sform and qform disagree; review source geometry')
    if b'aaoca_preprocessing' in bytes(image.header['descrip']):
        raise ValueError('input is an already normalized AAOCA output, not native HU')
    output_grid(image.shape, image.affine)
    orientation = nib.orientations.io_orientation(image.affine)
    if not np.isfinite(orientation).all():
        raise ValueError('cannot determine orientation')
    ras = nib.orientations.ornt_transform(orientation, nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    canonical_shape = tuple(np.asarray(image.shape)[np.argsort(ras[:, 0].astype(int))].tolist())
    canonical_affine = image.affine @ nib.orientations.inv_ornt_aff(ras, image.shape)
    shape, affine, mapping = output_grid(canonical_shape, canonical_affine, config.target_spacing)
    # Conservative array working-set estimate; filesystem cache and other processes are separate.
    source_bytes = math.prod(image.shape) * max(8, image.get_data_dtype().itemsize)
    slab_bytes = shape[0] * shape[1] * min(config.chunk_depth, shape[2]) * 16 * config.compute_threads
    return {
        'original_shape': list(image.shape), 'original_affine': image.affine.tolist(),
        'canonical_shape': list(canonical_shape), 'canonical_affine': canonical_affine.tolist(),
        'output_shape': list(shape), 'output_affine': affine.tolist(),
        'actual_spacing_mm': nib.affines.voxel_sizes(affine).tolist(),
        'output_index_to_canonical_index': mapping.tolist(),
        'output_index_to_original_index': (np.linalg.inv(image.affine) @ affine).tolist(),
        'original_index_to_output_index': (np.linalg.inv(affine) @ image.affine).tolist(),
        'canonical_spacing_mm': nib.affines.voxel_sizes(canonical_affine).tolist(),
        'downsample_axes_RAS': [axis for axis, old, new in zip('RAS', canonical_shape, shape) if new < old],
        'axis_size_factors_RAS': (np.array(shape) / canonical_shape).tolist(),
        'antialias': False,
        'input_scaling': {'slope': float(image.dataobj.slope), 'intercept': float(image.dataobj.inter)},
        'output_bytes_uncompressed': math.prod(shape) * 4,
        'estimated_working_bytes': source_bytes * 3 + slab_bytes + math.prod(shape) * 4,
        'scratch_bytes': math.prod(shape) * 4 + (int(image.dataobj.offset) +
                         math.prod(image.shape) * image.get_data_dtype().itemsize if str(path).lower().endswith('.gz') else 0),
    }


def check_gzip(path):
    if str(path).lower().endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            while stream.read(8 * 1024 * 1024):
                pass  # Read to EOF to validate gzip CRC/trailer, not just NIfTI voxel count.


def preprocess_image(source: Path, destination: Path, scratch: Path, config: Config, info: dict,
                     control=None, case_id=''):
    control = control or Control()
    timings = {}
    control.stage(case_id, 'decode_load')
    started = time.monotonic()
    with control.io():
        local = source
        if str(source).lower().endswith('.gz'):
            local = scratch / 'source.nii'
            # One sequential decompression, including CRC/trailer. Bound expansion
            # to the header-declared payload to avoid unbounded scratch usage.
            source_header = nib.load(source)
            expected = int(source_header.dataobj.offset) + math.prod(source_header.shape) * source_header.get_data_dtype().itemsize
            written = 0
            with gzip.open(source, 'rb') as reader, local.open('xb') as writer:
                while block := reader.read(8 * 1024 * 1024):
                    control.check()
                    written += len(block)
                    if written > expected:
                        raise ValueError('unexpected bytes after NIfTI payload')
                    writer.write(block)
            if written != expected:
                raise ValueError('truncated source NIfTI payload')
        # mmap=False releases scratch files reliably on Windows after loading.
        canonical = nib.as_closest_canonical(nib.load(local, mmap=False))
        data = canonical.get_fdata(dtype=np.float32, caching='unchanged')
        if not np.isfinite(data).all():
            raise ValueError('CT contains nonfinite pixels (NaN/Inf)')
    timings['decode_load'] = time.monotonic() - started
    shape = tuple(info['output_shape'])
    mapping = np.array(info['output_index_to_canonical_index'])
    output = np.memmap(scratch / 'normalized.float32', dtype=np.float32, mode='w+', shape=shape, order='F')
    lo, hi, below, above, total, squares = float('inf'), -float('inf'), 0, 0, 0.0, 0.0
    try:
        control.stage(case_id, 'compute', total_slices=shape[2])
        started = time.monotonic()
        def slab(z):
            control.check()
            depth = min(config.chunk_depth, shape[2] - z)
            if config.target_spacing is None:
                block = np.array(data[:, :, z:z+depth], dtype=np.float32, copy=True)
            else:
                offset = mapping[:3, 3] + mapping[:3, :3] @ [0, 0, z]
                block = affine_transform(data, mapping[:3, :3], offset=offset,
                                         output_shape=(shape[0], shape[1], depth), output=np.float32,
                                         order=1, mode='nearest', prefilter=False)
            return z, block

        # Keep at most compute_threads futures; consume in z order for stable
        # statistics and bounded memory even when an early slab is slow.
        positions = iter(range(0, shape[2], config.chunk_depth))
        with ThreadPoolExecutor(max_workers=config.compute_threads) as pool:
            pending = deque(pool.submit(slab, z) for z in [next(positions, None) for _ in range(config.compute_threads)] if z is not None)
            while pending:
                z, block = pending.popleft().result()
                control.check()
                depth = block.shape[2]
                below += int(np.count_nonzero(block < -1000))
                above += int(np.count_nonzero(block > 1000))
                np.clip(block, -1000, 1000, out=block)
                block /= np.float32(1000)
                lo, hi = min(lo, float(block.min())), max(hi, float(block.max()))
                total += float(block.sum(dtype=np.float64))
                squares += float(np.square(block, dtype=np.float64).sum())
                output[:, :, z:z+depth] = block
                control.stage(case_id, 'compute', completed_slices=z+depth, total_slices=shape[2])
                next_z = next(positions, None)
                if next_z is not None:
                    pending.append(pool.submit(slab, next_z))
        output.flush()
        timings['compute'] = time.monotonic() - started
        affine = np.array(info['output_affine'])
        result = nib.Nifti1Image(output, affine)  # fresh header: no source names/free-text/extensions
        result.set_sform(affine, code=1)
        direction = affine[:3, :3] / nib.affines.voxel_sizes(affine)
        orthogonal = np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-6)
        result.set_qform(affine, code=1 if orthogonal else 0)
        result.header.set_xyzt_units('mm')
        result.header.set_slope_inter(1, 0)
        result.header['descrip'] = b'aaoca_preprocessing v1; normalized [-1,1]; NOT HU'
        control.stage(case_id, 'save')
        started = time.monotonic()
        with control.io(), destination.open('xb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=config.compression_level, mtime=0) as stream:
                result.to_file_map({'image': nib.FileHolder(fileobj=stream)})
        timings['save'] = time.monotonic() - started
        control.stage(case_id, 'verify')
        started = time.monotonic()
        with control.io():
            verification = validate_written(destination, output, affine, config.chunk_depth, control)
        timings['verify'] = time.monotonic() - started
        count = math.prod(shape)
        return {
            'input_statistics_hu': {'min': float(data.min()), 'max': float(data.max())},
            'output_statistics': {'min': lo, 'max': hi, 'mean': total / count,
                                  'std': math.sqrt(max(0.0, squares / count - (total / count)**2)),
                                  'voxels': count, 'clipped_below': below, 'clipped_above': above},
            'verification': verification,
            'timings_seconds': timings, 'compression_level': config.compression_level,
        }
    finally:
        output._mmap.close()


def validate_written(path, expected, affine, chunk_depth=8, control=None):
    image = nib.load(path)
    if (image.shape != expected.shape or image.get_data_dtype() != np.dtype('float32')
            or image.header.get_xyzt_units()[0] != 'mm'
            or not np.allclose(image.affine, affine, rtol=0, atol=1e-4)
            or nib.aff2axcodes(image.affine) != ('R', 'A', 'S')
            or image.dataobj.slope != 1 or image.dataobj.inter != 0):
        raise ValueError('saved NIfTI geometry or dtype validation failed')
    digest = hashlib.sha256()
    # Sequential gzip stream avoids repeated decompression of large volumes for slab comparisons.
    with gzip.open(path, 'rb') as stream:
        offset = int(image.dataobj.offset)
        if len(stream.read(offset)) != offset:
            raise ValueError('truncated output header')
        for z in range(0, expected.shape[2], chunk_depth):
            if control:
                control.check()
            block = np.asarray(expected[:, :, z:z+chunk_depth], dtype='<f4').tobytes(order='F')
            saved = stream.read(len(block))
            if saved != block:
                raise ValueError('saved pixels differ from computed output')
            digest.update(saved)
        if stream.read(1):
            raise ValueError('unexpected trailing NIfTI payload')
    return {'all_voxels_roundtrip_equal': True, 'gzip_crc_verified': True,
            'voxel_payload_sha256': digest.hexdigest(),
            'affine_max_error_mm': float(np.max(np.abs(image.affine - affine)))}
