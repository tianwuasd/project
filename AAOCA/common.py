"""Shared, non-destructive DICOM conversion utilities."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import uuid
import warnings

import numpy as np
import pydicom
from pydicom.uid import CTImageStorage, EnhancedCTImageStorage, LegacyConvertedEnhancedCTImageStorage


class ConversionError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or []


def json_value(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): json_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return [json_value(x) for x in v]
    return str(v)


def write_json(path, value):
    path=Path(path)
    content=json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            stream.write(content);stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:16]


def source_info(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def read_dicom(path, header=False):
    # Avoid printing identifiable free-text DICOM values in decoder warnings.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ds = pydicom.dcmread(path, stop_before_pixels=header)
        # pydicom converts raw elements lazily. Force parsing inside this warning
        # boundary so invalid UIDs cannot leak to the console on later access.
        for element in ds.iterall():
            _ = element.value
    return ds, [w.category.__name__ for w in caught]


def pixels(ds):
    if 'PixelData' not in ds or not ds.PixelData:
        raise ConversionError('PixelData missing or empty')
    if int(getattr(ds, 'NumberOfFrames', 1)) < 1:
        raise ConversionError('NumberOfFrames must be positive')
    ts = ds.file_meta.TransferSyntaxUID
    if not ts.is_compressed:
        samples = 2 if ds.PhotometricInterpretation == 'YBR_FULL_422' else int(ds.SamplesPerPixel)
        bits = int(ds.Rows) * int(ds.Columns) * samples * int(getattr(ds, 'NumberOfFrames', 1)) * int(ds.BitsAllocated)
        expected = (bits + 7) // 8
        if len(ds.PixelData) not in (expected, expected + expected % 2):
            raise ConversionError(f'PixelData length mismatch: expected {expected}, found {len(ds.PixelData)}')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            array = ds.pixel_array
        if caught:
            raise ConversionError('Pixel decoder emitted warnings; manual review required',
                                  [{'warning_type': w.category.__name__} for w in caught])
        return array
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f'Pixel decode failed: {exc}') from exc


def prepare_output(inputs, output):
    roots = [Path(p).resolve() for p in inputs]
    if not roots or any(not p.is_dir() for p in roots):
        raise ValueError('Every input must be an existing directory')
    out = Path(output).resolve()
    if any(out == root or out.is_relative_to(root) for root in roots):
        raise ValueError('Output must be outside all input directories')
    out.mkdir(parents=True, exist_ok=False)
    return roots, out


@contextmanager
def staged_output(destination):
    """Publish a directory only after all files and verification have succeeded."""
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Inherit the output parent's access rules. Python 3.14 mkdtemp uses a
    # Windows owner-only ACL, which would survive publication by rename.
    temporary = dest.parent / ('.incomplete-' + uuid.uuid4().hex)
    temporary.mkdir(exist_ok=False)
    try:
        yield temporary
        if dest.exists():
            raise FileExistsError(dest)
        temporary.rename(dest)
    except BaseException:
        # Delete only the unique staging directory created by this invocation.
        if temporary.exists():
            if temporary.parent != dest.parent or not temporary.name.startswith('.incomplete-'):
                raise RuntimeError('Refusing unsafe staging cleanup')
            shutil.rmtree(temporary)
        raise


def scan(inputs, modality):
    selected, failures = [], []
    counts = {}
    seen = set()
    for root in inputs:
        for path in sorted(Path(root).rglob('*')):
            if not path.is_file() or path.suffix.lower() != '.dcm':
                continue
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                ds, warn = read_dicom(path, header=True)
                mod = str(getattr(ds, 'Modality', 'UNKNOWN'))
                sop = str(getattr(ds, 'SOPClassUID', ''))
                if sop in [str(CTImageStorage),str(EnhancedCTImageStorage),str(LegacyConvertedEnhancedCTImageStorage)] and mod != 'CT':
                    raise ConversionError('CT SOP class has missing or inconsistent Modality')
                if not mod.strip() or mod == 'UNKNOWN':
                    raise ConversionError('Missing Modality: cannot safely assign this DICOM')
                counts[mod] = counts.get(mod, 0) + 1
                if mod == modality:
                    selected.append((path, ds, warn))
            except Exception as exc:
                failures.append({'path': str(path), 'error': str(exc)})
    return selected, failures, counts


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--input', nargs='+', required=True, type=Path, help='One or more input directories, scanned recursively for .dcm')
    p.add_argument('--output', required=True, type=Path, help='NEW output directory outside all input directories')
    return p


def run_report(kind, roots):
    return {'converter': kind, 'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'input_directories': [str(r) for r in roots],
            'versions': {'python': platform.python_version(), **{k: importlib.metadata.version(k) for k in ['pydicom', 'numpy', 'nibabel', 'Pillow']}},
            'privacy': 'LOCAL RESEARCH COPY, NOT DE-IDENTIFIED: pixel text, source paths and filenames may identify a patient.',
            'converted': [], 'failed': [], 'scan_errors': []}


def finish(out, report):
    good, bad = len(report['converted']), len(report['failed']) + len(report['scan_errors'])
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    report['status'] = 'success' if good and not bad else ('partial_failure' if good else 'failed_or_no_matching_data')
    report['converted_count'] = good
    report['failed_count'] = bad
    write_json(out / 'manifest.json', report)
    print(f'{report["converter"]}: converted={good}, failed={bad}, status={report["status"]}', flush=True)
    print(f'Manifest: {out / "manifest.json"}', flush=True)
    return 0 if report['status'] == 'success' else 2
