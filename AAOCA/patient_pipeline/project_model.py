"""Versioned project options and small, serializable records."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import re

VERSION = 'project-v1'


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    default=str, allow_nan=False).encode('utf-8')).hexdigest()


def identity_evidence(record):
    """Bind an association to identity-bearing content, not to a reused pathname."""
    if record.get('kind')=='PDF':
        from patient_pipeline.clinical import filename_name
        name=record.get('document_name') or filename_name(record.get('path',''))
        return digest(['PDF',record.get('sha256'),name.casefold()])
    tags=(record.get('header') or record.get('previous_header') or {}).get('tags',{})
    return digest(['DICOM',{k:tags.get(k) for k in ['PatientID','PatientName','IssuerOfPatientID',
        'StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID']}])


def patient_folder(ident):
    # Preserve ordinary identifiers exactly, including leading zeros.
    if (not ident or ident in {'.','..'} or re.search(r'[<>:"/\\|?*\x00-\x1f]',ident)
            or ident.endswith((' ','.')) or re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',ident)):
        return '编号_'+digest(ident)[:16]
    return ident


@dataclass
class ProjectOptions:
    action: str = 'append'  # append, retry, report, audit
    match_names: bool = False
    recover_copies: bool = False
    split_phases: bool = False
    patients: tuple = ()
    policy_path: str = ''

    def __post_init__(self):
        if self.action not in {'append','retry','report','audit'}:
            raise ValueError('未知操作模式')

    def as_dict(self):
        return asdict(self)


def issue(code, message, pid='', source='', severity='error', **extra):
    return dict(code=code, message=str(message), patient_id=pid, source=str(source),
                severity=severity, **extra)
