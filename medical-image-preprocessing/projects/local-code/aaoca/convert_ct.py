"""Convert conventional single-frame CT series to 3-D NIfTI, preserving HU and geometry.

No missing-slice interpolation, resampling, denoising, anonymization or labels.
"""
from collections import defaultdict
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from pydicom.pixels import apply_modality_lut
from pydicom.uid import CTImageStorage

from common import (ConversionError, finish, parser, pixels, prepare_output, read_dicom,
                    run_report, scan, source_info, staged_output, token, write_json)


def vector(ds, key, length):
    try:
        a = np.asarray(getattr(ds, key), dtype=float)
    except Exception as exc:
        raise ConversionError(f'Missing or invalid {key}') from exc
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ConversionError(f'Invalid {key}')
    return a


def geometry(headers):
    if len(headers) < 2:
        raise ConversionError('At least two slices are needed; single-slice CT is not converted by this script')
    first = headers[0]
    iop = vector(first, 'ImageOrientationPatient', 6)
    row, col = iop[:3], iop[3:]
    if not (np.isclose(np.linalg.norm(row), 1, atol=1e-4) and np.isclose(np.linalg.norm(col), 1, atol=1e-4) and abs(np.dot(row, col)) < 1e-4):
        raise ConversionError('Orientation vectors must be unit and orthogonal')
    spacing = vector(first, 'PixelSpacing', 2)
    if (spacing <= 0).any():
        raise ConversionError('PixelSpacing must be positive')
    identity = (str(getattr(first, 'StudyInstanceUID', '')), str(getattr(first, 'SeriesInstanceUID', '')))
    if not all(identity):
        raise ConversionError('StudyInstanceUID and SeriesInstanceUID are required')
    seen_sop, positions = set(), []
    for ds in headers:
        if str(getattr(ds, 'SOPClassUID', '')) != str(CTImageStorage) or int(getattr(ds, 'NumberOfFrames', 1)) != 1:
            raise ConversionError('Only conventional single-frame CT Image Storage is supported; enhanced/multiframe CT needs a separate reader')
        if str(getattr(ds, 'Modality', '')) != 'CT' or int(ds.SamplesPerPixel) != 1 or ds.PhotometricInterpretation != 'MONOCHROME2':
            raise ConversionError('Expected single-channel MONOCHROME2 CT')
        if (str(getattr(ds, 'StudyInstanceUID', '')), str(getattr(ds, 'SeriesInstanceUID', ''))) != identity:
            raise ConversionError('Mixed studies or series')
        sop = str(getattr(ds, 'SOPInstanceUID', ''))
        if not sop or sop in seen_sop:
            raise ConversionError('Missing or duplicate SOPInstanceUID')
        seen_sop.add(sop)
        if (int(ds.Rows), int(ds.Columns)) != (int(first.Rows), int(first.Columns)):
            raise ConversionError('Inconsistent image dimensions')
        if not np.allclose(vector(ds, 'ImageOrientationPatient', 6), iop, atol=1e-5, rtol=0):
            raise ConversionError('Inconsistent image orientation')
        if not np.allclose(vector(ds, 'PixelSpacing', 2), spacing, atol=1e-5, rtol=0):
            raise ConversionError('Inconsistent PixelSpacing')
        for key in ['FrameOfReferenceUID', 'TemporalPositionIdentifier', 'TriggerTime']:
            if str(getattr(ds, key, '')) != str(getattr(first, key, '')):
                raise ConversionError(f'Inconsistent {key}; possible mixed geometry or cardiac phases')
        positions.append(vector(ds, 'ImagePositionPatient', 3))
    normal = np.cross(row, col)
    pos = np.asarray(positions)
    order = np.argsort(pos @ normal)
    pos = pos[order]
    gaps = np.diff(pos @ normal)
    if (gaps <= 1e-4).any():
        raise ConversionError('Duplicate or non-increasing slice positions')
    step = float(np.median(gaps))
    if not np.allclose(gaps, step, atol=max(.001, step*.001), rtol=0):
        raise ConversionError('Irregular slice spacing / possible missing slices', [{'gaps_mm': gaps.tolist()}])
    for ds in headers:
        if 'SpacingBetweenSlices' in ds:
            declared = float(ds.SpacingBetweenSlices)
            if not np.isfinite(declared) or declared <= 0 or not np.isclose(declared,step,atol=max(.001,step*.001),rtol=0):
                raise ConversionError('Declared SpacingBetweenSlices disagrees with physical positions; manual review required')
    # Retain a constant in-plane shift too (e.g. gantry tilt) in the sform.
    delta = (pos[-1] - pos[0]) / (len(pos)-1)
    if not np.allclose(pos, pos[0] + np.arange(len(pos))[:,None]*delta, atol=.01, rtol=0):
        raise ConversionError('Slice positions do not fit a single regular affine grid')
    affine_lps = np.eye(4)
    affine_lps[:3,0] = row * spacing[1]  # output X index = DICOM column index
    affine_lps[:3,1] = col * spacing[0]  # output Y index = DICOM row index
    affine_lps[:3,2] = delta
    affine_lps[:3,3] = pos[0]
    affine_ras = np.diag([-1.,-1.,1.,1.]) @ affine_lps
    return order, affine_ras, pos, step


def make_preview(volume, out):
    canvas = Image.new('RGB', (960, 350), (24,24,24))
    draw = ImageDraw.Draw(canvas)
    for i, z in enumerate([volume.shape[2]//4, volume.shape[2]//2, volume.shape[2]*3//4]):
        a = volume[:,:,z].T
        im = Image.fromarray((np.clip((a+200)/800, 0, 1)*255).astype(np.uint8)).convert('RGB')
        im.thumbnail((312,312))
        canvas.paste(im,(i*320+4,30))
        draw.text((i*320+5,7),f'Slice {z+1}/{volume.shape[2]}; W800 L200',fill='white')
    canvas.save(out/'preview.png')


def convert_series(paths, destination, quality_callback=None, checkpoint=None):
    paths = [Path(p) for p in paths]
    out = Path(destination)
    if out.exists():
        raise FileExistsError(out)
    headers = [read_dicom(p, header=True)[0] for p in paths]
    order, affine, positions, step = geometry(headers)
    paths = [paths[i] for i in order]
    sources = [source_info(p) for p in paths]
    first = headers[int(order[0])]
    volume = np.empty((int(first.Columns), int(first.Rows), len(paths)), dtype=np.float32)
    errors, warnings_out, scalings = [], [], []
    for z, path in enumerate(paths):
        if checkpoint:checkpoint(z,len(paths))
        try:
            ds, warning_types = read_dicom(path)
            a = pixels(ds)
            if a.shape != (int(first.Rows), int(first.Columns)):
                raise ConversionError('Decoded array shape does not match declared slice dimensions')
            if 'ModalityLUTSequence' in ds:
                raise ConversionError('CT ModalityLUTSequence not supported; HU semantics need manual review')
            if 'RescaleSlope' not in ds or 'RescaleIntercept' not in ds:
                raise ConversionError('Missing rescale parameters; cannot guarantee HU conversion')
            slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
            if not np.isfinite([slope,intercept]).all() or slope == 0:
                raise ConversionError('Invalid rescale parameters')
            if str(getattr(ds,'RescaleType','HU')).strip().upper() != 'HU':
                raise ConversionError('RescaleType is not HU')
            hu = apply_modality_lut(a, ds)
            values = hu.astype(np.float32)
            if not np.isfinite(values).all() or not np.allclose(values, hu, atol=.001, rtol=0):
                raise ConversionError('Non-finite HU or float32 conversion loses more than 0.001 HU')
            volume[:,:,z] = values.T
            if quality_callback is not None and z in {0,len(paths)//2,len(paths)-1}:
                quality_callback(values,z,len(paths))
            scalings.append({'slope':slope,'intercept':intercept})
            if warning_types:
                warnings_out.append({'file':path.name,'warning_types':warning_types})
        except Exception as exc:
            errors.append({'path':str(path),'error':str(exc)})
    if errors:
        raise ConversionError(f'{len(errors)} slice(s) failed; whole series rejected, no volume written', errors)
    meta = {'status':'converted','modality':'CT','array_axes':'DICOM columns, DICOM rows, spatially sorted slices',
            'shape':list(volume.shape),'dtype':'float32','intensity_units':'HU',
            'coordinate_system':'RAS+ millimetres','affine_ras_mm':affine.tolist(),
            'slice_increment_along_normal_mm':step,'positions_lps_mm':positions.tolist(),
            'slice_thickness_mm':float(first.SliceThickness) if 'SliceThickness' in first else None,
            'source_rescale_in_slice_order':scalings,'sources_in_slice_order':sources,
            'series_key':token(str(first.StudyInstanceUID)+'|'+str(first.SeriesInstanceUID)),
            'warnings':warnings_out,
            'operations':['sort_by_physical_position','apply_rescale_once','transpose_row_column','LPS_to_RAS_affine'],
            'not_performed':['resampling','missing_slice_filling','denoising','deidentification','segmentation'],
            'completeness_note':'Uniform spacing cannot prove that every original slice or full scan coverage was exported.'}
    with staged_output(out) as stage:
        if checkpoint:checkpoint(len(paths),len(paths))
        image = nib.Nifti1Image(volume, affine)
        image.header.set_xyzt_units('mm')
        image.set_sform(affine, code=1)
        image.set_qform(affine, code=0)  # sform is authoritative; qform cannot express shear
        image.header.set_slope_inter(1, 0)
        nifti_path = stage/'image.nii.gz'
        nib.save(image,nifti_path)
        restored = nib.load(nifti_path)
        if not np.allclose(restored.affine,affine,atol=1e-4,rtol=0):
            raise ConversionError('NIfTI affine round-trip verification failed')
        if not np.array_equal(np.asanyarray(restored.dataobj),volume):
            raise ConversionError('NIfTI voxel round-trip verification failed')
        meta['output_sha256'] = source_info(nifti_path)['sha256']
        meta['roundtrip_verified'] = True
        write_json(stage/'metadata.json',meta)
        make_preview(volume,stage)
    return meta


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    try:
        roots, out = prepare_output(args.input,args.output)
    except Exception as exc:
        print(f'Cannot start: {exc}',file=sys.stderr)
        return 1
    report = run_report('CT', roots)
    matches, report['scan_errors'], report['modalities_seen'] = scan(roots,'CT')
    if report['scan_errors']:
        report['aborted_reason'] = 'Unreadable DICOM header(s) cannot be safely assigned to a series; this CT run is blocked. Repair or explicitly narrow input directories.'
        print(report['aborted_reason'],flush=True)
        return finish(out,report)
    groups = defaultdict(list)
    for path, ds, _ in matches:
        study, series = str(getattr(ds,'StudyInstanceUID','')), str(getattr(ds,'SeriesInstanceUID',''))
        if not study or not series:
            report['failed'].append({'path':str(path),'error':'Missing Study/Series UID'})
        else:
            groups[(study,series)].append(path)
    if report['failed']:
        report['aborted_reason'] = 'CT headers missing grouping UIDs cannot be safely assigned; this run is blocked.'
        print(report['aborted_reason'],flush=True)
        return finish(out,report)
    for (study,series), paths in groups.items():
        folder = f'study_{token(study)}/series_{token(series)}'
        print(f'CT {folder}: {len(paths)} slices',flush=True)
        try:
            meta = convert_series(paths,out/folder)
            report['converted'].append({'directory':folder,'slices':len(paths),'shape':meta['shape'],'roundtrip_verified':True})
        except Exception as exc:
            report['failed'].append({'directory':folder,'source_files':[str(p) for p in paths],
                                     'error':str(exc),'details':getattr(exc,'details',[])})
            print(f'Rejected: {exc}',flush=True)
    return finish(out,report)


if __name__ == '__main__':
    sys.exit(main())
