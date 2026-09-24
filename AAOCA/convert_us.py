"""Convert US DICOM stills to lossless PNG and cine clips to lossless NPZ.

Retains temporal sampling and ultrasound region calibration. No cropping or de-identification.
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from pydicom.pixels import apply_color_lut
from common import (ConversionError, finish, json_value, parser, pixels, prepare_output,
                    read_dicom, run_report, scan, source_info, staged_output, token, write_json)


def finite_number(ds,key):
    v = getattr(ds,key,None)
    if v is None or str(v).strip() == '':
        return None
    x = float(v)
    if not np.isfinite(x):
        raise ConversionError(f'Non-finite {key}')
    return x


def frame_timing(ds,n):
    ft = finite_number(ds,'FrameTime')
    rate = finite_number(ds,'CineRate')
    display = finite_number(ds,'RecommendedDisplayFrameRate')
    delay = finite_number(ds,'FrameDelay')
    pointer = getattr(ds,'FrameIncrementPointer',[])
    if isinstance(pointer,int):
        pointer=[pointer]
    pointer=[int(v) for v in pointer]
    vector = getattr(ds,'FrameTimeVector',None)
    times, source = None, 'unknown'
    if vector is not None:
        increments = np.atleast_1d(np.asarray(vector,dtype=float))
        if len(increments) != n or not np.isfinite(increments).all() or abs(increments[0])>1e-6 or (increments[1:]<=0).any():
            raise ConversionError('Invalid FrameTimeVector: expected one finite increment per frame, first=0, others>0')
        if 0x00181063 in pointer and 0x00181065 not in pointer:
            if ft is None or ft<=0:
                raise ConversionError('FrameIncrementPointer points to invalid FrameTime')
            times = np.arange(n)*ft
            source = 'FrameTime'
        else:
            times = np.cumsum(increments)
            source = 'FrameTimeVector'
    elif 0x00181065 in pointer:
        raise ConversionError('FrameIncrementPointer requires missing FrameTimeVector')
    elif ft is not None and ft>0:
        times = np.arange(n)*ft
        source = 'FrameTime'
    elif 0x00181063 in pointer and n>1:
        raise ConversionError('FrameIncrementPointer requires valid FrameTime')
    elif rate is not None and rate>0:
        times = np.arange(n)*(1000/rate)
        source = 'CineRate_nominal'
    return {'source':source,'timestamps_ms':times.tolist() if times is not None else None,
            'timestamp_origin':'relative to first stored frame; add frame_delay_ms for offset from ContentTime',
            'frame_delay_ms':delay,'frame_time_ms':ft,'cine_rate_fps':rate,
            'recommended_display_fps':display,'frame_increment_pointer':pointer,
            'frame_time_vector_ms':json_value(vector),
            'first_to_last_frame_ms':float(times[-1]) if times is not None else None}


REGION_FIELDS = ['RegionSpatialFormat','RegionDataType','RegionFlags','RegionLocationMinX0',
                 'RegionLocationMinY0','RegionLocationMaxX1','RegionLocationMaxY1',
                 'ReferencePixelX0','ReferencePixelY0','PhysicalUnitsXDirection','PhysicalUnitsYDirection',
                 'ReferencePixelPhysicalValueX','ReferencePixelPhysicalValueY','PhysicalDeltaX','PhysicalDeltaY']
META_FIELDS = ['Rows','Columns','NumberOfFrames','SamplesPerPixel','BitsAllocated','BitsStored',
               'PixelRepresentation','PhotometricInterpretation','ImageType','LossyImageCompression',
               'LossyImageCompressionRatio','LossyImageCompressionMethod','BurnedInAnnotation',
               'PixelSpacing','ImagerPixelSpacing','RescaleSlope','RescaleIntercept','RescaleType',
               'WindowCenter','WindowWidth','VOILUTFunction','HeartRate','StartTrim','StopTrim','EffectiveDuration',
               'PreferredPlaybackSequencing']


def preview(frame, monochrome1=False):
    a = frame
    if a.ndim==3 and a.dtype==np.uint8:
        im = Image.fromarray(a)
    else:
        lo,hi = float(a.min()),float(a.max())
        a = (np.clip((a.astype(float)-lo)/max(hi-lo,1),0,1)*255).astype(np.uint8)
        if monochrome1:
            a=255-a
        im=Image.fromarray(a)
    im.thumbnail((800,800))
    return im


def convert_file(path, destination, quality_callback=None, checkpoint=None):
    path, out = Path(path), Path(destination)
    if out.exists():
        raise FileExistsError(out)
    src = source_info(path)
    if checkpoint:checkpoint(0,1)
    ds, warning_types = read_dicom(path)
    if str(getattr(ds,'Modality',''))!='US':
        raise ConversionError('Expected US modality')
    if str(getattr(ds,'SOPClassUID','')) not in {
            '1.2.840.10008.5.1.4.1.1.6','1.2.840.10008.5.1.4.1.1.6.1',
            '1.2.840.10008.5.1.4.1.1.3','1.2.840.10008.5.1.4.1.1.3.1'}:
        raise ConversionError('Only conventional 2-D ultrasound still/cine SOP classes are supported; volume or other objects need a dedicated reader')
    a = pixels(ds)
    if checkpoint:checkpoint(0,1)
    photo = str(ds.PhotometricInterpretation)
    if photo not in ['MONOCHROME1','MONOCHROME2','RGB','YBR_FULL','YBR_FULL_422','PALETTE COLOR']:
        raise ConversionError(f'Unsupported photometric interpretation: {photo}')
    if 'ModalityLUTSequence' in ds or 'VOILUTSequence' in ds or 'RealWorldValueMappingSequence' in ds:
        raise ConversionError('US contains lookup/mapping sequences that require a dedicated exporter; refusing to discard them')
    nf = int(getattr(ds,'NumberOfFrames',1))
    multi = 'NumberOfFrames' in ds
    samples = int(ds.SamplesPerPixel)
    expected = (int(ds.Rows),int(ds.Columns)) + ((samples,) if samples>1 else ())
    if a.shape != ((nf,)+expected if nf>1 else expected):
        raise ConversionError('Decoded shape does not match declared rows, columns, samples and frames')
    transformations = []
    if photo=='PALETTE COLOR':
        a=apply_color_lut(a,ds)
        transformations.append('palette_lookup_to_RGB')
    elif photo.startswith('YBR'):
        transformations.append('pydicom_YBR_to_RGB')
    saved_photo = 'RGB' if photo in ['RGB','YBR_FULL','YBR_FULL_422','PALETTE COLOR'] else photo
    frames = a if nf>1 else a[None,...]
    if quality_callback is not None:
        for index in sorted({0,nf//2,nf-1}):
            quality_callback(frames[index],index,nf)
    timing = frame_timing(ds,nf)
    # PNG supports uint8 RGB and unsigned 8/16-bit grayscale without loss.
    png_ok = (a.ndim==2 and a.dtype in [np.dtype('uint8'),np.dtype('uint16')]) or (a.ndim==3 and a.shape[-1]==3 and a.dtype==np.uint8)
    use_npz = multi or not png_ok
    meta = {'status':'converted','kind':'video' if multi else 'still',
            'storage':'NPZ' if use_npz else 'PNG','array_shape':list(frames.shape if use_npz else a.shape),
            'array_axes':'T,H,W,C' if use_npz and frames.ndim==4 else ('T,H,W' if use_npz else ('H,W,C' if a.ndim==3 else 'H,W')),
            'dtype':str(a.dtype),'number_of_frames':nf,'saved_photometric_interpretation':saved_photo,
            'source':src,'timing':timing,'source_tags':{k:json_value(getattr(ds,k,None)) for k in META_FIELDS},
            'ultrasound_regions':[{k:json_value(getattr(r,k,None)) for k in REGION_FIELDS} for r in getattr(ds,'SequenceOfUltrasoundRegions',[])],
            'transfer_syntax':str(ds.file_meta.TransferSyntaxUID),'dicom_warning_types':warning_types,
            'transformations':transformations,'deidentified':False,
            'privacy_note':'Image pixels retain original text and measurements. This output is not anonymized.',
            'preview_note':'Preview is resized; non-uint8 values min/max scaled, MONOCHROME1 inverted for display only. Numerical output is unchanged.'}
    with staged_output(out) as stage:
        if checkpoint:checkpoint(0,1)
        if use_npz:
            data_path=stage/'frames.npz'
            # Unknown timestamps are represented by an empty numeric vector, never object arrays.
            np.savez_compressed(data_path,frames=frames,
                                timestamps_ms=np.array(timing['timestamps_ms'] if timing['timestamps_ms'] is not None else [],dtype=np.float64))
            with np.load(data_path,allow_pickle=False) as restored:
                if not np.array_equal(restored['frames'],frames):
                    raise ConversionError('NPZ frame round-trip failed')
                expected_times=np.array(timing['timestamps_ms'] if timing['timestamps_ms'] is not None else [],dtype=float)
                if not np.array_equal(restored['timestamps_ms'],expected_times):
                    raise ConversionError('NPZ timestamps round-trip failed')
        else:
            data_path=stage/'image.png'
            Image.fromarray(a).save(data_path)
            with Image.open(data_path) as restored:
                if not np.array_equal(np.array(restored),a):
                    raise ConversionError('PNG pixel round-trip failed')
        preview(frames[nf//2],photo=='MONOCHROME1').save(stage/'preview.png')
        meta['output_sha256']=source_info(data_path)['sha256']
        meta['roundtrip_verified']=True
        write_json(stage/'metadata.json',meta)
    return meta


def main(argv=None):
    args=parser(__doc__).parse_args(argv)
    try:
        roots,out=prepare_output(args.input,args.output)
    except Exception as exc:
        print(f'Cannot start: {exc}',file=sys.stderr)
        return 1
    report=run_report('US',roots)
    matches,report['scan_errors'],report['modalities_seen']=scan(roots,'US')
    for index,(path,ds,_) in enumerate(matches):
        study=str(getattr(ds,'StudyInstanceUID','')) or str(path.parent)
        folder=f'study_{token(study)}/item_{token(str(path))}'
        try:
            meta=convert_file(path,out/folder)
            report['converted'].append({'directory':folder,'source':str(path),'kind':meta['kind'],
                                       'frames':meta['number_of_frames'],'storage':meta['storage'],'roundtrip_verified':True})
        except Exception as exc:
            report['failed'].append({'source':str(path),'error':str(exc),'details':getattr(exc,'details',[])})
        print(f'US {index+1}/{len(matches)}: {path.name}',flush=True)
    return finish(out,report)


if __name__=='__main__':
    sys.exit(main())
