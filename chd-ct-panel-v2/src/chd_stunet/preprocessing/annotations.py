"""Describe observed label availability without manufacturing segmentation truth.

Source label 0 in a partial case can contain unannotated myocardium. It stays
0 in the stored array, but a future loader/loss MUST consume this contract.
Presence checks establish which labels exist, not expert boundary accuracy.
"""
TARGET_CLASS_IDS = frozenset(range(1, 8))


def make_annotation(missing_label_ids):
    """Return a JSON/pickle-safe case contract based on native label presence."""
    missing = set(missing_label_ids)
    if not missing <= TARGET_CLASS_IDS:
        raise ValueError('Unknown missing target label IDs: ' + str(sorted(missing)))
    partial = bool(missing)
    return dict(schema=1, status='partial' if partial else 'complete',
                annotated_class_ids=sorted(TARGET_CLASS_IDS - missing),
                unannotated_class_ids=sorted(missing),
                background_semantics='background_or_unannotated' if partial else 'background',
                requires_partial_loss=partial,
                recommended_loss='marginal_ce_dice' if partial else 'standard_ce_dice',
                expert_boundary_review_verified=False)


def validate_resampled_labels(native_ids, output_ids, annotation):
    """Accept native omissions, reject lost known classes or invented classes.

    This check is intentionally stricter than merely allowing six output IDs:
    an originally annotated small PA disappearing during resampling still fails.
    Ignore=8 and crop padding=-1 are storage semantics, not target structures.
    """
    native_ids, output_ids = set(native_ids), set(output_ids)
    expected = make_annotation(TARGET_CLASS_IDS - native_ids)
    if annotation != expected:
        raise ValueError('Native labels differ from audited annotation contract')
    if not output_ids <= set(range(-1, 9)):
        raise ValueError('Non-discrete/unknown output labels')
    lost = set(annotation['annotated_class_ids']) - output_ids
    if lost:
        raise ValueError('Resampling removed target classes: ' + str(sorted(lost)))
    invented = (output_ids & TARGET_CLASS_IDS) - native_ids
    if invented:
        raise ValueError('Resampling invented unannotated target classes: ' + str(sorted(invented)))


def validate_case_roles(annotation, group, roles):
    """A partial case must never be emitted as validation/test or main train."""
    if annotation['requires_partial_loss']:
        if group != 'auxiliary' or roles != {str(f): 'aux_train' for f in range(3)}:
            raise ValueError('Partial annotation contract requires auxiliary aux_train roles in all folds')
    elif group != 'main' or any(role not in ('train', 'val', 'test') for role in roles.values()):
        raise ValueError('Complete main annotation contract has invalid cohort roles')
