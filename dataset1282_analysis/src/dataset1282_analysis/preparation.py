"""Scheme A patient allocation and admission to source-only nnU-Net preparation."""
from collections import Counter, defaultdict
from datetime import date
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from .contracts import atomic_json, digest, read_json, sha256_file
from .inputs import load_cases
from .metadata import link_metadata, read_metadata
from .observability.status import operation_lock, register_workers
from .resources import budget, preprocessing_memory, snapshot, thread_environment

SOURCES = ('LightSpeed VCT', 'SOMATOM Definition Flash')
TARGETS = {'iCT 256': 'target_primary', 'Brilliance iCT': 'target_secondary'}
SEED = 20261104
PROTOCOL = 'dataset1282-scheme-a-preprocess-1'


def age_band(row):
    try:
        birth = date.fromisoformat(row['PatientBirthDate'])
        exam = date.fromisoformat(row['AcquisitionDate'])
    except (ValueError, TypeError, KeyError):
        return 'unknown'
    if exam < birth:
        return 'unknown'
    age = exam.year - birth.year - ((exam.month, exam.day) < (birth.month, birth.day))
    return 'under1' if age < 1 else '1to5' if age < 6 else '6to17' if age < 18 else 'adult'


def allocate(cases, rows):
    """Use all workbook rows for patient embargoes, even unmatched and adult rows."""
    owners = defaultdict(set)
    for row in rows:
        owners[row['PatientID'].strip()].add(row['ManufacturerModelName'])
    strata, candidates = defaultdict(list), defaultdict(list)
    records = {}
    links = link_metadata(cases, rows)['case_rows']
    for link in links:
        name = link['case_id']
        if link['status'] != 'matched':
            records[name] = {'role': 'unlinked'}
            continue
        row = rows[link['excel_index']]
        patient = row['PatientID'].strip()
        scanner = row['ManufacturerModelName']
        band = age_band(row)
        devices = owners[patient]
        record = {'patient': patient, 'device': scanner, 'age_band': band,
                  'date': row['AcquisitionDate']}
        records[name] = record
        if not patient or 'PatientID_was_numeric_or_nontext' in row.get('_issues', []):
            role = 'identity_review'
        elif band in ('unknown', 'adult'):
            role = band
        elif len(devices.intersection(TARGETS)) > 1:
            role = 'dual_target_review'
        elif scanner in TARGETS:
            role = TARGETS[scanner]
        elif devices.intersection(TARGETS):
            role = 'target_patient_other_scan'
        elif scanner not in SOURCES or devices.difference(SOURCES):
            role = 'other_device'
        else:
            role = 'candidate'
            candidates[patient].append(name)
        record['role'] = role

    for patient, names in candidates.items():
        index = min(names, key=lambda name: (records[name]['date'], name))
        strata[records[index]['device'], records[index]['age_band']].append(patient)
    for patients in strata.values():
        patients.sort(key=lambda key: digest([SEED, key]))
        n = len(patients)
        # Largest-remainder rounding gives each stratum an exact integer partition.
        quotas = [n * f for f in (.70, .15, .15)]
        counts = [math.floor(v) for v in quotas]
        for i in sorted(range(3), key=lambda i: quotas[i] - counts[i], reverse=True)[:n-sum(counts)]:
            counts[i] += 1
        for i, patient in enumerate(patients):
            role = 'train' if i < counts[0] else 'development' if i < sum(counts[:2]) else 'source_test'
            for name in candidates[patient]:
                records[name]['role'] = role
    counts = Counter(r['role'] for r in records.values())
    groups = defaultdict(set)
    for record in records.values():
        if record.get('patient'):
            groups[record['role']].add(record['patient'])
    return {'protocol': PROTOCOL, 'seed': SEED, 'status': 'metadata_candidates',
            'scans': dict(counts), 'patient_groups': {k: len(v) for k, v in groups.items()},
            'cases': records}


def select_cases(cases, cohort, review):
    """Only disclose train/development inputs to the isolated numerical worker."""
    records = cohort['cases']
    source = {c.case_id: c for c in cases if records[c.case_id]['role'] in ('train', 'development')}
    if review is None:
        names = []
        for role, limit in [('train', 2), ('development', 1)]:
            patients = set()
            for name, case in sorted(source.items()):
                r = records[name]
                if r['role'] == role and case.collection == 'native' and r['patient'] not in patients:
                    names.append(name)
                    patients.add(r['patient'])
                    if len(patients) == limit:
                        break
    else:
        if review.get('cohort_digest') != digest(cohort):
            raise ValueError('Review does not match frozen cohort')
        if review.get('intensity', 'body_percentile') not in ('body_percentile', 'verified_hu'):
            raise ValueError('Unknown intensity protocol')
        names = list(review['cases'])
        if set(names) - source.keys():
            raise ValueError('Review includes an ineligible or frozen case')
        for name in names:
            item = review['cases'][name]
            if not isinstance(item.get('evidence'), str) or not item['evidence'].strip() or not all(
                item.get(k) is True for k in ('preoperative_verified', 'patient_group_verified',
                                              'index_alignment_verified', 'full_supervision_verified')
            ):
                raise ValueError('Incomplete physical/clinical review for ' + name)
            if item.get('source_sha256') != source[name].expected_hashes:
                raise ValueError('Review is not bound to the current original pair')
            if review.get('intensity') == 'verified_hu' and item.get('hu_verified') is not True:
                raise ValueError('HU evidence is required for every selected case')
        # Main evaluation allows one eligible index examination per development patient.
        chosen_patients = set()
        selected = []
        for name in sorted(names, key=lambda n: (records[n]['date'], n)):
            if records[name]['role'] == 'development':
                patient = records[name]['patient']
                if patient in chosen_patients:
                    continue
                chosen_patients.add(patient)
            selected.append(name)
        names = selected
    result = {}
    for name in names:
        c = source[name]
        if set(c.expected_hashes) != {'image', 'label'}:
            raise ValueError('Both original file hashes are required')
        result[name] = {'role': records[name]['role'], 'image': c.image, 'label': c.label,
                        'sha256': c.expected_hashes, 'review': review['cases'][name] if review else None}
    roles = {r['role'] for r in result.values()}
    if 'train' not in roles or (review is not None and 'development' not in roles):
        raise ValueError('Both source training and development need eligible cases')
    return result


def excel_spacing(row, image_shape):
    """Convert only a unique series' spacing fields, without a layer-thickness fallback."""
    if tuple(image_shape) != tuple(row[k] for k in ('Cols', 'Rows', 'valid_z')):
        raise ValueError('NIfTI dimensions differ from Excel Cols/Rows/valid_z')
    spacing = [row[k] for k in ('PixelSpacingX', 'PixelSpacingY', 'calc_zSpacing')]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError('Invalid Excel pixel spacing or slice increment')
    return list(map(float, spacing))


def apply_excel_geometry(selected, cases, rows):
    """Fill reviewed geometry using the strictly matched acquisition sampling."""
    import nibabel as nib
    import numpy as np

    links = {r['case_id']: r for r in link_metadata(cases, rows)['case_rows']}
    for name, record in selected.items():
        review = record['review']
        source = review.get('spacing_source', 'reviewed_affine')
        if source == 'reviewed_affine':
            continue
        if source != 'excel' or review.get('excel_grid_verified') is not True or 'affine_ras_mm' in review:
            raise ValueError('Excel geometry requires one verified source grid for ' + name)
        link = links[name]
        if link['status'] != 'matched':
            raise ValueError('No unique Excel series for ' + name)
        row = rows[link['excel_index']]
        spacing = excel_spacing(row, nib.load(record['image']).shape)
        axes = np.asarray(review.get('directions_ras'), dtype=float)
        origin = np.asarray(review.get('origin_ras_mm'), dtype=float)
        if (axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5)
                or origin.shape != (3,) or not np.isfinite(origin).all()):
            raise ValueError('Verified unit axis directions and RAS-mm origin required for ' + name)
        affine = np.eye(4)
        affine[:3, :3] = axes * spacing
        affine[:3, 3] = origin
        record['review'] = dict(review, affine_ras_mm=affine.tolist(), excel_row=row['_excel_row'],
                                source_spacing_xyz_mm=spacing)


def select_excel_pilot(cases, cohort, rows, memory_budget=None):
    """Inspect training headers only. Sample independent patients before reading pixels."""
    import nibabel as nib
    import numpy as np
    links = {r['case_id']: r for r in link_metadata(cases, rows)['case_rows']}
    eligible = defaultdict(list)
    inspected = {}
    for case in cases:
        name = case.case_id
        meta = cohort['cases'][name]
        if meta['role'] != 'train':
            continue
        issues = []
        inspected[name] = {'device': meta['device'], 'issues': issues}
        link = links[name]
        if link['status'] != 'matched':
            issues.append('no_unique_excel_row')
            continue
        row = rows[link['excel_index']]
        try:
            image = nib.load(case.image)
            label = nib.load(case.label)
            if (case.collection != 'native' or label.shape != image.shape
                    or not np.allclose(label.affine, image.affine, rtol=0, atol=1e-5)):
                issues.append('pair_grid_review')
            spacing = excel_spacing(row, image.shape)
            inspected[name].update(source_spacing_xyz_mm=spacing, shape_xyz=list(image.shape),
                                   excel_row=row['_excel_row'])
        except (ValueError, OSError, nib.filebasedimages.ImageFileError) as exc:
            issues.append(str(exc))
        if set(case.expected_hashes) != {'image', 'label'}:
            issues.append('source_hash_missing')
        if not issues:
            eligible[meta['device']].append(case)
    selected, patients = {}, set()

    def pick(case, role):
        name = case.case_id
        patients.add(cohort['cases'][name]['patient'])
        selected[name] = {'role': role, 'image': case.image, 'label': case.label,
                          'sha256': case.expected_hashes, 'excel_geometry': inspected[name], 'review': None}

    for device in SOURCES:
        picked = 0
        for case in sorted(eligible[device], key=lambda c: c.case_id):
            if cohort['cases'][case.case_id]['patient'] not in patients:
                pick(case, 'train')
                picked += 1
                if picked == 2:
                    break
    screen = {'scope': 'source_train_candidates_only', 'checked': len(inspected),
              'eligible': sum(len(v) for v in eligible.values()), 'cases': inspected,
              'available_budget_bytes': memory_budget}
    if selected:
        # A header-only estimate for local selection. The official planner still determines the grid.
        target = np.median([inspected[n]['source_spacing_xyz_mm'] for n in selected], axis=0)
        screen['estimated_planning_spacing_xyz_mm'] = target.tolist()
        for device in SOURCES:
            probes = []
            for case in eligible[device]:
                name = case.case_id
                info = inspected[name]
                shape = np.array(info['shape_xyz'])
                projected = np.ceil(shape * info['source_spacing_xyz_mm'] / target).astype(int)
                need = preprocessing_memory(math.prod(info['shape_xyz']), math.prod(map(int, projected)))
                info['estimated_peak_bytes'] = need
                if memory_budget is not None and need > memory_budget:
                    info['capacity'] = 'exceeds_local_reservation'
                    continue
                if cohort['cases'][name]['patient'] not in patients:
                    probes.append((need, name, case))
            if probes:
                pick(min(probes)[2], 'pilot_probe')
    return selected, screen


def prepare(config, root, home, args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.run):
        raise ValueError('Invalid preparation run name')
    base = Path(config['work_root']) / 'prepared' / args.run
    if base.resolve() != base:
        raise ValueError('Linked preparation output')
    with operation_lock(home, 'prepare', base) as owner:
        manifest = Path(config['data_root']) / 'manifest.json'
        excel = Path(config['metadata_excel'])
        identity = {'manifest': sha256_file(manifest), 'excel': sha256_file(excel)}
        cases = load_cases(config)
        rows = read_metadata(excel, config['metadata_sheet'])
        cohort = allocate(cases, rows)
        if identity != {'manifest': sha256_file(manifest), 'excel': sha256_file(excel)}:
            raise ValueError('Inputs changed during allocation')
        cohort['inputs'] = identity
        path = base / 'cohort.json'
        if path.exists() and read_json(path) != cohort:
            raise ValueError('Frozen cohort differs. Use a new run name.')
        if not path.exists():
            atomic_json(path, cohort)
        result = {k: cohort[k] for k in ('protocol', 'status', 'scans', 'patient_groups')}
        result.update(directory=str(base), cohort_digest=digest(cohort))
        if not args.voxel_smoke and not args.excel_pilot and not args.review:
            return result
        review = read_json(args.review) if args.review else None
        if args.review and not isinstance(review, dict):
            raise ValueError('Review must be an object')
        mode = 'excel-pilot' if args.excel_pilot else 'voxel-smoke' if review is None else 'physical'
        output = base / mode
        if output.exists():
            raise FileExistsError('Output already exists. Use a fresh run name: ' + str(output))
        if args.excel_pilot:
            memory = budget(snapshot(), config)['memory_bytes']
            if memory is None:
                raise RuntimeError('Cannot establish local memory capacity')
            selected, screening = select_excel_pilot(cases, cohort, rows, memory)
            atomic_json(base / 'excel-screening.json', screening)
            if not selected:
                raise ValueError('No source training cases passed Excel grid screening')
            if any(screening['cases'][n].get('capacity') for n in selected):
                raise RuntimeError('Pilot fitting set exceeds local memory. See excel-screening.json.')
        else:
            selected = select_cases(cases, cohort, review)
            if review is not None:
                apply_excel_geometry(selected, cases, rows)
        output.mkdir()
        request = {'mode': mode, 'cases': selected, 'cohort_digest': digest(cohort),
                   'intensity': review.get('intensity', 'body_percentile') if review else 'body_percentile',
                   'review_sha256': sha256_file(args.review) if args.review else None,
                   'policy': {k: config[k] for k in ('cpu_fraction', 'ram_fraction', 'reserve_gib',
                                                      'read_concurrency', 'max_workers')}}
        atomic_json(output / 'request.json', request)
        env = dict(os.environ, **thread_environment(), PYTHONPATH=str(root / 'src'))
        for key, sub in [('nnUNet_raw', 'raw'), ('nnUNet_preprocessed', 'preprocessed'),
                         ('nnUNet_results', 'results')]:
            env[key] = str(output / sub)
        cmd = [args.research_python or sys.executable, '-B', '-m',
               'dataset1282_analysis.nnunet_prepare', str(output)]
        with (output / 'preparation.log').open('wb') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            register_workers(home, owner, [proc.pid])
            if proc.wait():
                raise RuntimeError('Preparation failed. See ' + str(output / 'preparation.log'))
        return dict(result, numerical=read_json(output / 'complete.json'))
