"""Private policy and cohort admission, independent of the numerical environment."""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import os
from pathlib import Path
import re
import shutil
import sys

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..inputs import load_cases
from ..metadata import link_metadata, read_metadata
from ..preparation import allocate, age_band, excel_spacing, TARGETS
from ..engine.runner import code_identity
from ..observability.logging import event
from ..observability.status import operation_lock, write_status
from ..resources import preprocessing_memory

DATASET = 'Dataset282_CHDSource'


def derive_cohort(cases, rows, frozen, policy, identities):
    """Never reshuffle existing roles when removing scans or resolving an alias."""
    expected = allocate(cases, rows)
    expected['inputs'] = identities
    if frozen != expected:
        raise ValueError('Frozen cohort does not match the current manifest/workbook/allocation')
    result = deepcopy(frozen)
    records = result['cases']
    by_id = {c.case_id: c for c in cases}
    links = {r['case_id']: r['excel_index'] for r in link_metadata(cases, rows)['case_rows']}
    seen = set()
    for alias in policy.get('patient_aliases', []):
        name = alias['case_id']
        if name in seen or name not in by_id or records[name]['role'] != 'unlinked':
            raise ValueError('Alias must resolve exactly one previously unlinked case')
        seen.add(name)
        c = by_id[name]
        matches = [i for i, r in enumerate(rows) if r['idx'] == alias['excel_idx']]
        if (len(matches) != 1 or alias['source_key'] != c.source_key
                or alias['source_sha256'] != c.expected_hashes
                or alias['workbook_sha256'] != identities['excel']):
            raise ValueError('Alias evidence is not bound to these inputs')
        index = matches[0]
        row = rows[index]
        patient = row['PatientID'].strip()
        if (patient != alias['patient_id'] or patient not in c.source_key.split('_')
                or sum(r['PatientID'].strip() == patient for r in rows) != 1
                or index in links.values() or row['_excel_row'] != alias['excel_row']):
            raise ValueError('Alias requires a unique exact patient identifier and unused Excel row')
        band, device = age_band(row), row['ManufacturerModelName']
        # This explicit workflow alias is a held-out scanner; no refitting of source roles.
        if band in ('unknown', 'adult') or device not in TARGETS:
            raise ValueError('Alias needs a separately reviewed role')
        records[name] = dict(patient=patient, device=device, age_band=band,
                             date=row['AcquisitionDate'], role=TARGETS[device], alias=True)
        links[name] = index
    excluded = set()
    for item in policy.get('excluded_cases', []):
        name = item['case_id']
        if name in excluded or name not in by_id or item['source_sha256'] != by_id[name].expected_hashes:
            raise ValueError('Excluded case evidence changed')
        excluded.add(name)
        records[name]['excluded_reason'] = item['reason']
    owners = defaultdict(set)
    for name, r in records.items():
        if r.get('patient') and name not in excluded:
            group = r['role']
            if group == 'target_patient_other_scan':
                group = 'target'
            if group.startswith('target_'):
                group = 'target'
            owners[r['patient']].add(group)
    if any(len(roles) > 1 and roles.intersection({'train', 'development', 'source_test'})
           for roles in owners.values()):
        raise ValueError('Patient leakage across frozen roles')
    groups = defaultdict(list)
    for name, r in records.items():
        r['included'] = name not in excluded
        if r.get('patient') and r['included']:
            groups[(r['patient'], r['role'])].append(name)
    for names in groups.values():
        first = min(names, key=lambda n: (records[n]['date'], n))
        for name in names:
            records[name]['index_scan'] = name == first
    result.update(parent_digest=digest(frozen), policy_digest=digest(policy),
                  status='derived_frozen_roles', included_scans=sum(r['included'] for r in records.values()),
                  main_index_scans=sum(r['included'] and r.get('index_scan', False) for r in records.values()))
    return result, links


def shared_geometry(image, label, spacing):
    """Only the documented unit placeholder is repairable by this policy."""
    import numpy as np
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError('Image and label require identical 3-D array dimensions')
    unit = np.eye(4)
    shifted = unit.copy()
    shifted[:3, 3] = 1
    if (not np.allclose(image.affine, unit, rtol=0, atol=1e-5)
            or not (np.allclose(label.affine, unit, rtol=0, atol=1e-5)
                    or np.allclose(label.affine, shifted, rtol=0, atol=1e-5))):
        raise ValueError('Unexpected geometry: only reviewed identity/+1 placeholder grids are allowed')
    if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError('Invalid millimetre spacing')
    return np.diag([*spacing, 1.])


def prepare_workflow(config, root, home, args):
    import nibabel as nib
    from .queue import run_jobs
    from .runtime import numerical_runtime
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.run):
        raise ValueError('Invalid run identifier')
    output = Path(config['work_root']) / 'prepared' / args.run / 'hu-marginal'
    if output.resolve() != output:
        raise ValueError('Linked preparation output')
    with operation_lock(home, 'prepare-hu', output) as owner:
        cases = load_cases(config)
        rows = read_metadata(config['metadata_excel'], config['metadata_sheet'])
        identities = dict(manifest=sha256_file(Path(config['data_root']) / 'manifest.json'),
                          excel=sha256_file(config['metadata_excel']))
        policy = read_json(args.workflow)
        hu = policy['HU_policy']
        if hu['additional_offset'] != 0 or hu['additional_scale'] != 1:
            raise ValueError('This workflow accepts direct reader-scaled HU only')
        if (not policy['partial_supervision']['include_training_partition_missing_class_cases']
                or policy['label_mapping'].get('renumber_voxels') is not False
                or set(policy['label_mapping']['values']) != {str(i) for i in range(11)}
                or len(set(policy['label_mapping']['values'].values())) != 11
                or policy['label_mapping']['values']['0'] != 'background'):
            raise ValueError('Expected fixed 0..10 labels and partial supervision')
        cohort, links = derive_cohort(cases, rows, read_json(args.frozen_cohort), policy, identities)
        records = cohort['cases']
        eligible = {c.case_id: c for c in cases if records[c.case_id]['role'] in ('train', 'development')
                    and records[c.case_id]['included'] and records[c.case_id].get('index_scan')}
        names = sorted(set(args.cases)) if args.cases else sorted(eligible)
        if set(names) - eligible.keys():
            raise ValueError('Selection contains frozen, excluded or non-index cases')
        selected = {}
        for name in names:
            c = eligible[name]
            if links[name] is None or set(c.expected_hashes) != {'image', 'label'}:
                raise ValueError('Missing exact metadata or source hashes: ' + name)
            im, lab = nib.load(c.image), nib.load(c.label)
            spacing = excel_spacing(rows[links[name]], im.shape)
            shared_geometry(im, lab, spacing)
            selected[name] = dict(role=records[name]['role'], image=c.image, label=c.label,
                                  sha256=c.expected_hashes, shape=list(im.shape), spacing=spacing,
                                  known_absent=policy.get('known_absent_classes', {}).get(name, []))
        if not any(r['role'] == 'train' for r in selected.values()):
            raise ValueError('At least one source training case is required')
        python = args.research_python or sys.executable
        runtime_identity = numerical_runtime(python)
        code = code_identity(root)
        request = dict(schema=1, method='HU-marginal-v1', cases=selected, cohort_digest=digest(cohort),
                       policy_digest=digest(policy), labels=policy['label_mapping']['values'],
                       scope='technical_subset' if args.cases else 'all_eligible_source_index_scans',
                       code=code, numerical_runtime=runtime_identity, nnunetv2='2.8.1', staged_extension=args.staged_extension,
                       fingerprint_samples=min(100_000_000, 100_000 * sum(r['role']=='train' for r in selected.values()))
                       if args.cases else 100_000_000)
        if output.exists():
            if not args.resume or read_json(output/'request.json') != request:
                raise ValueError('Existing run requires --resume and an unchanged code/input/policy contract')
            if code_identity(output/'code_snapshot') != code:
                raise ValueError('Frozen numerical code changed')
        else:
            if args.resume:
                raise ValueError('Cannot resume a missing run')
            output.mkdir(parents=True)
            atomic_json(output/'request.json', request)
            atomic_json(output/'cohort.json', cohort)
            for relative in code:
                target = output/'code_snapshot'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root/relative, target)
        event(output, 'workflow_start', scope=request['scope'], cases=len(selected), resume=args.resume)
        # Re-hash even on resume: source identity cannot be inferred from a stale manifest.
        for name, r in selected.items():
            for kind in ('image', 'label'):
                if sha256_file(r[kind]) != r['sha256'][kind]:
                    raise ValueError('Source changed: ' + name + '/' + kind)
        if identities != dict(manifest=sha256_file(Path(config['data_root'])/'manifest.json'),
                              excel=sha256_file(config['metadata_excel'])):
            raise ValueError('Input metadata changed during admission')
        runtime = {k:config[k] for k in ('cpu_fraction','ram_fraction','reserve_gib','max_workers',
                                       'read_concurrency','resource_wait_seconds')}
        if args.workers:
            runtime['max_workers'] = args.workers
        if runtime['max_workers'] < 1:
            raise ValueError('Worker count must be positive')
        try:
            required_disk=sum(math.prod(r['shape'])*5 for r in selected.values())
            if not args.resume and shutil.disk_usage(output).free < required_disk*1.2+2**30:
                raise OSError('Insufficient disk headroom for staged working copies')
            jobs = [dict(id='stage-'+n, kind='stage', case=n,
                         memory=700*2**20+24*math.prod(r['shape'])) for n,r in selected.items()]
            stage_stats = run_jobs(output, jobs, python, runtime, home, owner)
            fit_memory = max(preprocessing_memory(math.prod(r['shape'])) for r in selected.values() if r['role']=='train')
            fit_stats = run_jobs(output, [dict(id='fit',kind='fit',memory=fit_memory)], python, runtime, home, owner)
            fit = read_json(output/'fit.json')
            jobs = [dict(id='process-'+n,kind='process',case=n,
                         memory=preprocessing_memory(math.prod(r['shape']),
                         math.prod(max(1,math.ceil(a*b/c)) for a,b,c in zip(r['shape'],r['spacing'],fit['spacing_xyz']))))
                    for n,r in selected.items()]
            process_stats = run_jobs(output, jobs, python, runtime, home, owner)
            train = [n for n,r in selected.items() if r['role']=='train']
            val = [n for n,r in selected.items() if r['role']=='development']
            dest = output/'preprocessed'/DATASET
            atomic_json(dest/'splits_final.json', [dict(train=train,val=val)])
            supervision = {n:read_json(output/'staged'/n/'metadata.json')['supervision'] for n in selected}
            atomic_json(dest/'partial_supervision.json', supervision)
            contract_files=['partial_supervision.json','splits_final.json','nnUNetPlans.json','dataset.json']
            atomic_json(dest/'training_contract.json',dict(request_digest=digest(request),
                        artifacts={p:sha256_file(dest/p) for p in contract_files}))
            result = dict(state='completed', run_dir=str(output), scope=request['scope'],
                          train_cases=len(train), development_cases=len(val),
                          partial_cases=sum(bool(v['unknown_classes']) for v in supervision.values()),
                          fit=fit, stages=[stage_stats,fit_stats,process_stats],
                          request_digest=digest(request), training_executed=False,
                          hu_calibration_verified=False, coordinate_frame='local_array_excel_mm')
            atomic_json(output/'complete.json', result)
            write_status(output, **result)
            event(output, 'workflow_complete', **result)
            return result
        except BaseException as exc:
            write_status(output,state='interrupted',error=type(exc).__name__+': '+str(exc))
            event(output,'workflow_interrupted',error=str(exc))
            raise
