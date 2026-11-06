"""Produce prediction-only ROI candidates, then certify structure coverage before downstream use."""
from pathlib import Path
import re

import numpy as np

from .contracts import atomic_json, digest, read_json
from .xu_roi import MAX_BOX_FRACTION, MIN_CLASS_P05, RULE, coverage, expand_box, measure_box, vote_box


def _roles(cache):
    return {name: role for role in ('train', 'development') for name in cache.case_ids(role)}


def _identity(cache, args):
    from .xu_workflow import PROTOCOL, input_id
    return dict(protocol=PROTOCOL, request=cache.request_digest, locator_input=input_id(cache,args.supervision),
                profile=args.profile, supervision=args.supervision, rule=RULE,
                max_fraction=MAX_BOX_FRACTION, minimum_class_p05=MIN_CLASS_P05)


def _check_identity(payload, cache, args):
    if any(payload[key] != value for key, value in _identity(cache, args).items()):
        raise ValueError('ROI identity differs from this experiment')
    if type(payload['padding_percent']) is not int or payload['padding_percent'] not in (0, 10):
        raise ValueError('Unrecognized ROI margin')
    checkpoints = payload['checkpoints']
    if (set(checkpoints) != {'crop64','crop128'} or
            not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in checkpoints.values())):
        raise ValueError('Invalid locator identities')


def localize(cache, args, out):
    from .xu_workflow import frozen_predictor, side
    destination = out / 'candidates.json'
    if destination.exists() or (out/'boxes.json').exists():
        raise ValueError('Use a fresh output directory for localization')
    identity = _identity(cache, args)
    models = {key: frozen_predictor(path, branch=key, profile=args.profile, supervision=args.supervision,
                                     data_identity=identity['locator_input'], device=args.device)
              for key, path in [('crop64',args.crop64), ('crop128',args.crop128)]}
    cases = {}
    roles = _roles(cache)
    for i, (name, role) in enumerate(roles.items(), 1):
        predictions = [predict(cache.read_image(name, (side(key),)*3))
                       for key, (predict, _) in models.items()]
        item = vote_box(*predictions, cache.geometry(name), args.padding_percent)
        cases[name] = dict(role=role, **item)
        print(f"Localize {i}/{len(roles)} {role} {name} "
              f"fraction={item['quality']['native_fraction']} outcome={item['quality']['reason'] or 'candidate'}",
              flush=True)
    payload = dict(**identity, padding_percent=args.padding_percent,
                   checkpoints={k:h for k,(_,h) in models.items()}, cases=cases)
    atomic_json(destination, dict(payload=payload, sha256=digest(payload)))
    return dict(command='localize', cases=len(cases), candidate_file=str(destination),
                rejected=sum(not row['quality']['accepted'] for row in cases.values()),
                next_step='audit-roi')


def audit(cache, args, out):
    if (out/'boxes.json').exists() or (out/'localization_qc.json').exists():
        raise ValueError('Use a fresh output directory for the ROI audit')
    envelope = read_json(args.candidates)
    payload = envelope['payload']
    if digest(payload) != envelope['sha256']:
        raise ValueError('ROI candidates changed since localization')
    _check_identity(payload, cache, args)
    cases, roles = payload['cases'], _roles(cache)
    if set(cases) != set(roles):
        raise ValueError('Incomplete ROI candidate cohort')
    rows, failures = {}, []
    for i, (name, role) in enumerate(roles.items(), 1):
        item, geo = cases[name], cache.geometry(name)
        if item['role'] != role:
            raise ValueError('Candidate role differs from frozen split')
        if digest(item['box']) != digest(expand_box(item['strict_box'], geo.cache_shape, payload['padding_percent'])):
            raise ValueError('Candidate ROI differs from the frozen vote and requested margin')
        quality = measure_box(item['box'], geo)
        covers = coverage(cache.read_native_label(name),
                          {key:item[key] for key in ('box','strict_box','union_box','legacy_box')},
                          geo, cache.supervision[name]['present_classes'])
        reasons = []
        if not quality['accepted']:
            reasons.append(quality['reason'])
        if any(value == 0 for value in covers['box'].values()):
            reasons.append('missed_structure')
        if reasons:
            failures.append(name)
        rows[name] = dict(role=role, quality=quality, coverage=covers, reasons=reasons,
                          strict_fraction=measure_box(item['strict_box'], geo)['native_fraction'],
                          legacy_fraction=measure_box(item['legacy_box'], geo)['native_fraction'],
                          union_fraction=measure_box(item['union_box'], geo)['native_fraction'],
                          disputed_membership=item['disputed_membership'])
        print(f"Audit ROI {i}/{len(roles)} {role} {name} {reasons or 'ok'}", flush=True)
    by_role, low = {}, []
    for role in ('train','development'):
        rates = {}
        for label in range(1,11):
            values = [row['coverage']['box'][label] for row in rows.values()
                      if row['role'] == role and label in row['coverage']['box']]
            rates[label] = dict(n=len(values), p05=float(np.percentile(values,5)) if values else None)
            if values and rates[label]['p05'] < MIN_CLASS_P05:
                low.append([role, label])
        by_role[role] = rates
    accepted = not failures and not low
    book = {key:value for key,value in payload.items() if key != 'cases'}
    book.update(boxes={name:item['box'] for name,item in cases.items()},
                candidates_sha256=envelope['sha256'], convention='half-open cache edges')
    qc = dict(accepted=accepted, cases=rows, by_role=by_role, failed_cases=failures,
              low_class_coverage=low, book_digest=digest(book))
    atomic_json(out/'localization_qc.json', qc)
    if not accepted:
        raise ValueError(f'ROI audit rejected {len(failures)} cases and {len(low)} class groups; '
                         'see localization_qc.json. No training box book released.')
    # Publish last. A partially completed or failed audit cannot authorize training.
    atomic_json(out/'boxes.json', book)
    return dict(command='audit-roi', cases=len(cases), accepted=True, file=str(out/'boxes.json'))


def verified_book(path, cache, args):
    if path is None:
        raise ValueError('Pass the boxes.json released by audit-roi')
    path = Path(path)
    book = read_json(path)
    qc = read_json(path.parent/'localization_qc.json')
    if ('rule' not in book or qc.get('accepted') is not True or qc.get('failed_cases') != []
            or qc.get('low_class_coverage') != [] or qc.get('book_digest') != digest(book)):
        raise ValueError('An accepted, matching non-degenerate ROI audit is required')
    _check_identity(book, cache, args)
    names = set(_roles(cache))
    if set(book['boxes']) != names or set(qc['cases']) != names:
        raise ValueError('Incomplete ROI book')
    for name, box in book['boxes'].items():
        q = measure_box(box, cache.geometry(name))
        if not q['accepted'] or qc['cases'][name]['reasons']:
            raise ValueError(f'Invalid ROI book for {name}: {q["reason"]}')
    return book
