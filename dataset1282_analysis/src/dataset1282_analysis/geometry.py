"""Header evidence does not certify anatomical correspondence or physical calibration."""
import math
import numpy as np
import nibabel as nib


def header_info(path):
    image = nib.load(str(path))
    h, p = image.header, image.dataobj
    affine = image.affine
    finite = bool(np.isfinite(affine).all())
    singular = not finite or abs(float(np.linalg.det(affine[:3,:3]))) < 1e-12
    def matrix(value):
        return [[float(v) if math.isfinite(float(v)) else None for v in row] for row in value]
    q, qc = image.get_qform(coded=True); s, sc = image.get_sform(coded=True)
    return {'shape':list(image.shape), 'dtype':str(image.get_data_dtype()),
            'spacing':[float(x) for x in h.get_zooms()], 'units':list(h.get_xyzt_units()),
            'affine':matrix(affine), 'affine_singular':singular,
            'orientation':list(nib.aff2axcodes(affine)) if not singular else [None]*3,
            'qform_code':int(qc),'sform_code':int(sc),'qform':matrix(q) if q is not None else None,
            'sform':matrix(s) if s is not None else None,
            'qform_sform_disagree':bool(q is not None and s is not None and not np.allclose(q,s,rtol=0,atol=1e-5)),
            'proxy_slope':float(p.slope),'proxy_intercept':float(p.inter),
            'intensity_units':'unverified; proxy-scaled values are not assumed HU'}


def inspect_geometry(image_path, label_path, metadata=None):
    image, label = header_info(image_path), header_info(label_path)
    a,b = np.array(image['affine'],dtype=float),np.array(label['affine'],dtype=float)
    same = bool(np.allclose(a,b,rtol=0,atol=1e-5))
    known = image['units'][0] != 'unknown' and image['units'][0] == label['units'][0]
    physical = known and same and not image['affine_singular'] and not label['affine_singular']
    conflicts = []
    if metadata:
        for key, dim in [('Rows',0),('Cols',1),('valid_z',2)]:
            value = metadata.get(key)
            if value is not None and (len(image['shape'])<=dim or value != image['shape'][dim]):
                conflicts.append({'field':key,'excel_value':value,'image_value':image['shape'][dim] if len(image['shape'])>dim else None})
    return {'image':image,'label':label,
            'array_geometry_status':'match' if image['shape']==label['shape'] else 'mismatch',
            'header_affine_match':same,
            'origin_delta_header_units':[float(v) if math.isfinite(v) else None for v in (b[:3,3]-a[:3,3])],
            'physical_geometry_status':'header_consistent_unvalidated' if physical else 'unverified',
            'physical_volume_supported':False,
            'label_semantics_status':'unverified_numeric_labels',
            'metadata_shape_conflicts':conflicts,
            'physical_note':'Header consistency alone does not validate physical calibration or segmentation alignment.'}
