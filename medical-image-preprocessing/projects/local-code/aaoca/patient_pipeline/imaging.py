"""Metadata-only inventory and conservative image checks."""
from collections import defaultdict
import math
from pathlib import Path
import re
import warnings

import numpy as np
from pydicom.dataset import Dataset
from pydicom.datadict import tag_for_keyword
from pydicom.filereader import read_partial
from pydicom.uid import CTImageStorage, EnhancedCTImageStorage, LegacyConvertedEnhancedCTImageStorage

from common import json_value, sha256
from convert_us import REGION_FIELDS, frame_timing
from patient_pipeline.clinical import normal_id

FIELDS=['PatientID','IssuerOfPatientID','PatientName','PatientSex','PatientBirthDate','PatientAge',
 'Modality','SOPClassUID','SOPInstanceUID','StudyInstanceUID','SeriesInstanceUID','StudyDate','StudyTime',
 'SeriesNumber','InstanceNumber','StudyDescription','SeriesDescription','ProtocolName','ImageType',
 'ContrastBolusAgent','ContrastBolusVolume','ContrastBolusRoute','Manufacturer','ManufacturerModelName',
 'Rows','Columns','SamplesPerPixel','PhotometricInterpretation','NumberOfFrames','BitsAllocated','BitsStored',
 'PixelRepresentation','PixelSpacing','SliceThickness','SpacingBetweenSlices','ImagePositionPatient',
 'ImageOrientationPatient','FrameOfReferenceUID','RescaleSlope','RescaleIntercept','RescaleType',
 'KVP','XRayTubeCurrent','Exposure','ConvolutionKernel','ReconstructionDiameter','CTDIvol',
 'TemporalPositionIdentifier','TriggerTime','NominalInterval','HeartRate','NumberOfSeriesRelatedInstances',
 'NumberOfStudyRelatedInstances','FrameTime','FrameTimeVector','FrameIncrementPointer','FrameDelay',
 'CineRate','RecommendedDisplayFrameRate','LossyImageCompression','LossyImageCompressionRatio',
 'BurnedInAnnotation','SequenceOfUltrasoundRegions','SpecificCharacterSet']
TAGS=[tag_for_keyword(k) for k in FIELDS]


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    return value


def folder_id(path):
    for part in Path(path).parents:
        if re.fullmatch(r'\d{4,20}',part.name):return part.name
    return ''


def read_header(path):
    path=Path(path);pixel={};size=path.stat().st_size
    with path.open('rb') as stream, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        def stop(tag,vr,length):
            if int(tag) in (0x7FE00010,0x7FE00008,0x7FE00009):
                pixel.update(tag=int(tag),length=length,offset=stream.tell());return True
            return False
        ds=read_partial(stream,stop_when=stop,specific_tags=TAGS)
        ct_classes={str(CTImageStorage),str(EnhancedCTImageStorage),str(LegacyConvertedEnhancedCTImageStorage)}
        if str(getattr(ds,'SOPClassUID','')) in ct_classes and str(getattr(ds,'Modality',''))!='CT':
            raise ValueError('CT SOPClassUID与Modality不一致，拒绝按其他模态解释像素')
        nf=int(getattr(ds,'NumberOfFrames',1))
        if nf<1:raise ValueError('NumberOfFrames必须为正整数')
        tags={k:finite_json(json_value(getattr(ds,k,None))) for k in FIELDS if k!='SequenceOfUltrasoundRegions'}
        regions=[{k:finite_json(json_value(getattr(r,k,None))) for k in REGION_FIELDS} for r in getattr(ds,'SequenceOfUltrasoundRegions',[])]
        ts=getattr(ds.file_meta,'TransferSyntaxUID',None)
        compressed=bool(ts and ts.is_compressed)
        timing=frame_timing(ds,int(getattr(ds,'NumberOfFrames',1))) if str(getattr(ds,'Modality',''))=='US' else None
    ident=normal_id(tags.get('PatientID'));fallback=folder_id(path)
    status='missing';expected=None
    if pixel:
        length=pixel['length']
        if length==0:status='empty'
        elif compressed or length==0xFFFFFFFF:status='compressed_needs_decode'
        else:
            status='native_length_ok'
            try:
                samples=2 if tags['PhotometricInterpretation']=='YBR_FULL_422' else int(tags['SamplesPerPixel'])
                expected=math.ceil(int(tags['Rows'])*int(tags['Columns'])*samples*int(tags.get('NumberOfFrames') or 1)*int(tags['BitsAllocated'])/8)
                if length not in (expected,expected+expected%2):status='length_mismatch'
            except (TypeError,ValueError,KeyError):status='invalid_pixel_parameters'
            if pixel['offset']+length>size:status='truncated'
    if pixel and pixel['tag']!=0x7FE00010:status='unsupported_float_pixel_data'
    return {'path':str(path.resolve()),'bytes':size,'mtime_ns':path.stat().st_mtime_ns,
            'patient_id':ident or fallback,'identity_basis':'PatientID' if ident else ('folder_unverified' if fallback else 'unknown'),
            'folder_patient_id':fallback,'issuer':str(tags.get('IssuerOfPatientID') or ''),
            'tags':tags,'pixel_status':status,'pixel_length':pixel.get('length'),'expected_pixel_bytes':expected,
            'transfer_syntax':str(ts or ''),'timing':timing,'ultrasound_regions':regions,
            'warning_types':sorted({w.category.__name__ for w in caught})}


def dataset_from_tags(tags):
    ds=Dataset()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k,v in tags.items():
            if v is not None:setattr(ds,k,v)
    return ds


def classify_ct(tags):
    fields={k:str(tags.get(k) or '') for k in ['StudyDescription','SeriesDescription','ProtocolName']}
    text=' '.join(fields.values())
    evidence=[f'{k}={v}' for k,v in fields.items() if v]
    if re.search(r'calcium|non[- ]?contrast|without contrast|钙化评分|平扫',text,re.I):return 'CT_未确认CTA',evidence
    if re.search(r'\bCTA\b|CT\s*angiogra|coronary\s*angiogra|冠(?:脉|状动脉)(?:CT)?(?:造影|血管成像)',text,re.I):return 'CTA_描述明确',evidence
    if tags.get('ContrastBolusAgent') and re.search(r'cardiac|heart|coronary|冠脉|冠状动脉|心脏|心臟',text,re.I):
        return '增强心脏CT_疑似CTA',evidence+['ContrastBolusAgent='+str(tags['ContrastBolusAgent'])]
    return 'CT_未确认CTA',evidence


def deduplicate(records):
    groups=defaultdict(list);kept=[];removed=[];conflicts=[]
    for r in records:
        sop=r['tags'].get('SOPInstanceUID')
        if sop:groups[sop].append(r)
        else:kept.append(r)
    for sop,items in groups.items():
        if len(items)==1:kept.extend(items);continue
        fingerprints=defaultdict(list)
        for r in items:fingerprints[sha256(r['path'])].append(r)
        if len(fingerprints)>1:
            conflicts.extend(items);kept.extend(items)
        else:
            items=sorted(items,key=lambda r:r['path']);kept.append(items[0])
            for r in items[1:]:removed.append({'source':r['path'],'kept_source':items[0]['path'],'sop':sop,'sha256':next(iter(fingerprints))})
    return kept,removed,conflicts


def quality_metrics(array,index,total):
    a=np.asarray(array);finite=np.isfinite(a);values=a[finite]
    return {'index_zero_based':index,'total':total,'shape':list(a.shape),'dtype':str(a.dtype),
            'min':float(values.min()) if values.size else None,'max':float(values.max()) if values.size else None,
            'mean':float(values.mean()) if values.size else None,'std':float(values.std()) if values.size else None,
            'finite_fraction':float(finite.mean()),'zero_fraction':float((a==0).mean()),
            'constant':bool(values.size and values.min()==values.max())}


def unique_values(records,key):
    import json
    values={json.dumps(r['tags'][key],ensure_ascii=False,sort_keys=True) for r in records if r['tags'].get(key) is not None}
    return [json.loads(v) for v in sorted(values)]


BAD_PIXELS={'missing','empty','length_mismatch','truncated','invalid_pixel_parameters','unsupported_float_pixel_data'}
