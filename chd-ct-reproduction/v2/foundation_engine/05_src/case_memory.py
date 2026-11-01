"""Conservative natural-case memory evidence, never a private/incremental claim."""
import math
from pathlib import Path
import time
from server_data import read_json

MAX_AGE_SECONDS=24*3600
PEAK_SCOPE='process lifetime high-water at case boundary'


def compatibility(image,spacing,stage):
    from preparation_identity import current
    return dict(stage=stage,geometry=dict(shape=list(image.shape),spacing=[float(x) for x in image.header.get_zooms()],
        target_spacing=[float(x) for x in spacing] if spacing is not None else None),implementation=current())


def corrected_estimate(work,prior,keys,*,now=None):
    now=time.time() if now is None else now
    floor=prior;used=[]
    for path in sorted((Path(work)/'00_admin/case-observations').glob('*.json')):
        try:row=read_json(path)
        except (OSError,ValueError):continue
        if not isinstance(row,dict):continue
        peak=row.get('peak_rss_bytes');observed=row.get('observed_at',0)
        if not isinstance(observed,(int,float)) or not math.isfinite(observed):continue
        age=now-observed
        if (row.get('status')!='completed' or row.get('compatibility') not in keys or
            row.get('peak_scope')!=PEAK_SCOPE or not 0<=age<=MAX_AGE_SECONDS or
            not isinstance(peak,(int,float)) or not math.isfinite(peak) or peak<=0):continue
        # Entire process high-water (including imports/shared pages) plus margin.
        # Do not subtract shared pages or claim this is an incremental case peak.
        corrected=math.ceil(peak*1.1)
        floor=max(floor,corrected)
        used.append(dict(path=str(path),peak_rss_bytes=peak,conservative_floor=corrected,
            output_bytes=row.get('output_bytes'),peak_scope=PEAK_SCOPE,age_seconds=age))
    return floor,used
