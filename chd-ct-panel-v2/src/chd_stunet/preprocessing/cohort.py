"""Method-specific ImageCHD quality and historic case-level split rules."""
import random
import numpy as np
from .annotations import make_annotation

TARGETS = dict(background=0, LV=1, RV=2, LA=3, RA=4, MYO=5, AO=6, PA=7, ignore=8)


def frozen_roles(ids, seed=42):
    """Reproduce the previous 97-case split BEFORE filtering the new exclusion.

    These are file-pair identities. Patient-level independence is unverified.
    Never shuffle the remaining 96 again: that would change held-out subjects.
    """
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    roles = {cid: {} for cid in order}
    for fold in range(3):
        test = set(order[fold::3])
        remainder = [cid for cid in order if cid not in test]
        random.Random(seed + 1009 * (fold + 1)).shuffle(remainder)
        count = max(1, round(len(order) * .1))
        if count >= len(remainder):
            raise ValueError('Not enough complete cases for the frozen split')
        val = set(remainder[:count])
        for cid in order:
            roles[cid][str(fold)] = 'test' if cid in test else 'val' if cid in val else 'train'
    return roles


def map_labels(source, allowed=(8, 10, 12, 13, 14, 15)):
    """Map only documented auxiliary classes to ignore=8 in a NEW array.

    Unknown numbers stop this method. Case-level missing annotations are
    described separately; mapping must never guess where unannotated MYO is.
    """
    values = np.unique(source)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError('Non-integer or non-finite segmentation')
    unknown = set(map(int, values)) - set(range(8)) - set(allowed)
    if unknown:
        raise ValueError('Unknown labels: ' + str(sorted(unknown)))
    result = source.astype(np.int8, copy=True)
    result[result > 7] = 8
    return result


def select_cohort(rows, method):
    """Keep the historical main roles; admit only the named auxiliary policy.

    Auxiliary cases have no historical fold role. Their explicit aux_train
    role prevents them entering validation/test or main-train fitting.
    """
    included, excluded = [], []
    policy = method.get('partial_annotation_policy')
    if policy not in (None, 'missing_myo_only'):
        raise ValueError('Unknown partial annotation policy: ' + str(policy))
    for row in rows:
        reasons = []
        missing = row.get('missing_label_ids', [])
        auxiliary = policy == 'missing_myo_only' and set(missing) == {5}
        if missing and not auxiliary:
            reasons.append('missing_target_annotations:' + str(missing))
        if row['case_id'] in method['exclude_cases']:
            reasons.append('confirmed_source_intensity_quality_exclusion')
        group = 'excluded' if reasons else 'auxiliary' if auxiliary else 'main'
        copy = dict(row, exclusion_reasons=reasons, eligible=not reasons,
                    cohort_group=group, annotation=make_annotation(missing))
        if group == 'auxiliary':
            copy['roles'] = {str(fold): 'aux_train' for fold in range(3)}
        (excluded if reasons else included).append(copy)
    if len(included) != method['expected_included']:
        raise ValueError(f"Cohort changed: expected {method['expected_included']}, got {len(included)}")
    for group, key in (('main', 'expected_main'), ('auxiliary', 'expected_auxiliary')):
        count = sum(r['cohort_group'] == group for r in included)
        if key in method and count != method[key]:
            raise ValueError(f"Cohort changed: expected {method[key]} {group}, got {count}")
    return included, excluded
