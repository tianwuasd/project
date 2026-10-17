"""Conservative, opt-in repairs: select real copies or separate explicit phase tags."""
from collections import defaultdict
import copy

import numpy as np
from common import ConversionError, pixels, read_dicom, sha256
from convert_ct import geometry
from patient_pipeline.imaging import BAD_PIXELS, dataset_from_tags

PROOF_TAGS=['PatientID','IssuerOfPatientID','SOPClassUID','SOPInstanceUID','StudyInstanceUID','SeriesInstanceUID',
 'Modality','Rows','Columns','SamplesPerPixel','PhotometricInterpretation','BitsAllocated','BitsStored',
 'PixelRepresentation','NumberOfFrames','ImagePositionPatient','ImageOrientationPatient','PixelSpacing',
 'SpacingBetweenSlices','FrameOfReferenceUID','RescaleSlope','RescaleIntercept','RescaleType',
 'TemporalPositionIdentifier','TriggerTime','FrameTime','FrameTimeVector','FrameIncrementPointer',
 'NominalPercentageOfCardiacPhase']


def select_copies(records, enabled, checkpoint):
    groups=defaultdict(list);kept=[];repairs=[]
    for r in records:
        if r.get('read_error'):raise ConversionError('源文件读取失败：'+r['read_error'])
        t=r['header']['tags'];groups[t.get('SOPInstanceUID') or r['key']].append(r)
    for sop,items in groups.items():
        checkpoint()
        if len(items)==1:kept.extend(items);continue
        hashes={}
        for r in items:
            checkpoint();hashes.setdefault(sha256(r['path']),[]).append(r)
        if len(hashes)==1:
            kept.append(items[0]);repairs.append({'type':'exact_duplicate','kept':items[0]['path'],
                                                'excluded':[i['path'] for i in items[1:]],'sop':sop})
            continue
        authorized=enabled is True or (isinstance(enabled,set) and sop in enabled)
        if not authorized:raise ConversionError('SOP_CONTENT_CONFLICT：同一影像唯一编号对应不同文件内容')
        proof=lambda r:([r['header']['tags'].get(k) for k in PROOF_TAGS],r['header'].get('ultrasound_regions'))
        if any(proof(r)!=proof(items[0]) for r in items[1:]):
            raise ConversionError('SOP_CONTENT_CONFLICT：同一编号的几何、身份或像素解释参数不一致，不能选择副本')
        valid=[r for r in items if r['header']['pixel_status'] not in BAD_PIXELS]
        if not valid:raise ConversionError('PIXEL_MISSING：同一编号的所有副本均无完整像素')
        first=pixels(read_dicom(valid[0]['path'])[0])
        for r in valid[1:]:
            checkpoint()
            if not np.array_equal(first,pixels(read_dicom(r['path'])[0])):
                raise ConversionError('SOP_CONTENT_CONFLICT：完整副本的实际像素不同，不自动选取')
        kept.append(valid[0]);repairs.append({'type':'complete_copy','kept':valid[0]['path'],
            'kept_sha256':sha256(valid[0]['path']),'excluded':[r['path'] for r in items if r is not valid[0]],
            'proof':'同SOP、身份与几何一致；选用完整且解码通过的像素；所有完整副本像素逐点一致','sop':sop})
    return kept,repairs


def phase_groups(records):
    """No phase inference: require explicit repeated tags and equal, regular 3-D grids."""
    chosen=None
    for key in ['NominalPercentageOfCardiacPhase','TemporalPositionIdentifier','TriggerTime']:
        values=[r['header']['tags'].get(key) for r in records]
        if all(v is not None and str(v)!='' for v in values) and len(set(map(str,values)))>1:
            chosen=key;break
    if not chosen:raise ConversionError('没有可用于可靠分组的明确期相标签')
    groups=defaultdict(list)
    for r in records:groups[str(r['header']['tags'][chosen])].append(r)
    if len({len(g) for g in groups.values()})!=1 or min(map(len,groups.values()))<3:
        raise ConversionError('各期相切片数不一致或不足3张，拒绝自动拆分')
    evidence=[];grids=[];reference=None
    for label,items in sorted(groups.items()):
        headers=[dataset_from_tags(r['header']['tags']) for r in items]
        shape_ref=(int(headers[0].Rows),int(headers[0].Columns),str(getattr(headers[0],'FrameOfReferenceUID','')))
        if reference is None:reference=shape_ref
        elif shape_ref!=reference:raise ConversionError('不同期相的图像宽高或参考坐标系不同，不能确认同一空间网格')
        override=False;declared=[]
        try:g=geometry(headers)
        except ConversionError as exc:
            if 'Declared SpacingBetweenSlices disagrees' not in str(exc):raise
            declared=[float(h.SpacingBetweenSlices) for h in headers if 'SpacingBetweenSlices' in h]
            if len(declared)!=len(headers) or len(set(declared))!=1:raise
            copied=copy.deepcopy(headers)
            for h in copied:del h.SpacingBetweenSlices
            g=geometry(copied);override=True
        grids.append(g)
        evidence.append({'tag':chosen,'value':label,'slices':len(items),'actual_spacing_mm':g[3],
                         'original_spacing_values':sorted(set(declared)), 'spacing_override':override})
    for g in grids[1:]:
        if not np.allclose(grids[0][1],g[1],atol=1e-4,rtol=0) or not np.allclose(grids[0][2],g[2],atol=1e-4,rtol=0):
            raise ConversionError('不同期相未覆盖同一空间网格，拒绝自动拆分')
    return [(label,groups[label],proof) for label,proof in zip(sorted(groups),evidence)]


def phase_geometry(headers):
    copies=copy.deepcopy(headers)
    for h in copies:
        if 'SpacingBetweenSlices' in h:del h.SpacingBetweenSlices
    return geometry(copies)
