"""Private spreadsheet records; strict source-key linkage, no patient merging."""
from collections import Counter, defaultdict
from datetime import date, datetime
import math

FIELDS = ['idx','PatientName','PatientID','PatientBirthDate','PatientSex','AcquisitionDate',
          'ManufacturerModelName','PixelSpacingX','PixelSpacingY','SliceThickness','Rows','Cols',
          'all_file_z','all_read_file_z','valid_z','calc_zSpacing','dicom_folder']
NUMERIC = FIELDS[7:16]


def as_date(value):
    if isinstance(value, datetime): return value.date()
    if isinstance(value, date): return value
    if value is None or value == '': return None
    text = str(value).strip()
    for pattern in ('%Y%m%d','%Y-%m-%d','%Y-%m-%dT%H:%M:%S'):
        try: return datetime.strptime(text, pattern).date()
        except ValueError: pass
    raise ValueError('Invalid date')


def read_metadata(path, sheet='dicom2nii_info'):
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in workbook.sheetnames: raise ValueError('Metadata sheet is missing')
        iterator = workbook[sheet].iter_rows(values_only=True)
        headers = list(next(iterator))
        if headers != FIELDS: raise ValueError('Unexpected metadata columns or order')
        rows = []
        for index, values in enumerate(iterator, start=2):
            if all(v is None for v in values): continue
            row = dict(zip(headers, values)); issues = []
            row['_excel_row'] = index
            for key in ('idx','PatientID'):
                if row[key] is None: issues.append(key + '_missing')
                elif not isinstance(row[key], str): issues.append(key + '_was_numeric_or_nontext')
                row[key] = '' if row[key] is None else str(row[key])
            for key in ('PatientBirthDate','AcquisitionDate'):
                try:
                    value = as_date(row[key]); row[key] = value.isoformat() if value else None
                except ValueError: issues.append(key + '_invalid')
            row['age_at_acquisition_years'] = None
            try:
                birth, acquisition = as_date(row['PatientBirthDate']), as_date(row['AcquisitionDate'])
                if birth and acquisition:
                    days = (acquisition-birth).days
                    if days < 0: issues.append('negative_acquisition_age')
                    else: row['age_at_acquisition_years'] = days / 365.2425
            except ValueError: pass
            for key in NUMERIC:
                value = row[key]
                if value is None: continue
                if not isinstance(value, (int,float)) or not math.isfinite(value): issues.append(key + '_invalid_numeric')
            row['_issues'] = issues
            rows.append(row)
        return rows
    finally: workbook.close()


def link_metadata(cases, rows):
    by_key = defaultdict(list)
    by_source = Counter(c.source_key for c in cases)
    for i, row in enumerate(rows):
        if row.get('idx'): by_key[row['idx']].append(i)
    links, consumed, involved = [], set(), set()
    for case in cases:
        candidates = by_key.get(case.source_key, [])
        status = 'matched' if len(candidates)==1 and by_source[case.source_key]==1 else ('ambiguous' if candidates else 'unmatched')
        links.append({'case_id':case.case_id, 'source_key':case.source_key, 'status':status,
                      'excel_index': candidates[0] if status=='matched' else None,
                      'candidate_excel_indices':candidates})
        involved.update(candidates)
        if status == 'matched': consumed.update(candidates)
    return {'case_rows':links, 'matched_count':len(consumed),
            'unmatched_excel_indices':[i for i in range(len(rows)) if i not in involved],
            'ambiguous_excel_indices': sorted(involved-consumed),
            'method':'exact idx == source_key only; no fuzzy names or PatientID aliasing'}


def describe(values):
    import numpy as np
    present = [float(x) for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
    if not present: return {'valid_count':0,'missing_or_invalid':len(values),'min':None,'median':None,'max':None,'mean':None}
    return {'valid_count':len(present),'missing_or_invalid':len(values)-len(present),
            'min':min(present),'median':float(np.median(present)),'max':max(present),'mean':sum(present)/len(present)}


def summarize_metadata(rows):
    ids = Counter(r.get('PatientID') for r in rows if r.get('PatientID'))
    return {'record_count':len(rows), 'distinct_patient_ids':len(ids),
            'distinct_ids_are_not_verified_independent_patients':True,
            'duplicate_id_groups':{k:v for k,v in ids.items() if v>1},
            'missing_patient_id_records':sum(not r.get('PatientID') for r in rows),
            'categories':{key:dict(Counter(str(r.get(key)) if r.get(key) is not None else '(missing)' for r in rows))
                          for key in ('PatientSex','ManufacturerModelName')},
            'numeric':{key:describe([r.get(key) for r in rows]) for key in NUMERIC+['age_at_acquisition_years']},
            'date_range':{key:{'min':min([r[key] for r in rows if r.get(key)], default=None),
                               'max':max([r[key] for r in rows if r.get(key)], default=None)}
                          for key in ('AcquisitionDate',)},
            'records_with_issues':sum(bool(r.get('_issues')) for r in rows)}
