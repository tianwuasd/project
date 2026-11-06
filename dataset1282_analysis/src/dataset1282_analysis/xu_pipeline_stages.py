"""Ordered Xu-model stages. Each runnable step calls the existing single-stage CLI."""
from dataclasses import dataclass
from pathlib import Path

from .xu_blood import blood_cohort
from .xu_targets import TargetStore, blood_ids


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    required: tuple[str, ...]
    resume_marker: str | None = None
    skip_reason: dict | None = None


def auxiliary_counts(cache, targets, supervision):
    """Count supervision under the same inclusion rules as the two auxiliary trainers."""
    store = TargetStore(targets, cache)
    blood = blood_cohort(cache,blood_ids(cache),supervision,store)
    counts = {'blood': {role:len(names) for role,names in blood.items()}, 'init': {}}
    for role in ('train', 'development'):
        init_names = cache.case_ids(role, complete_only=supervision == 'complete')
        counts['init'][role] = sum(store.has(name, 'init') for name in init_names)
    return counts


def _shortfall(kind, counts, minimum_train):
    if counts['train'] >= minimum_train and counts['development'] >= 1:
        return None
    return dict(reason='insufficient_auxiliary_supervision', target=kind,
                eligible=dict(counts), required=dict(train=minimum_train, development=1))


def iter_stages(cache, args, out):
    """Consume lazily: the targets stage commits on disk before eligibility is inspected."""
    out = Path(out)
    prepared = ['--prepared', str(args.prepared)]
    if args.technical_subset:
        prepared.append('--technical-subset')
    model = ['--profile', args.profile, '--device', args.device]
    arm = [*model, '--supervision', args.supervision]
    seed = ['--seed', str(args.seed)]
    training_files = ('best.pt', 'training.json')
    fusion_files = ('job.json', 'complete.json', 'evaluation.json')

    def stage(name, command, options, files, marker=None, skip=None):
        argv = (command, *prepared, '--output', str(out/name), *map(str, options))
        return Stage(name, argv, files, marker, skip)

    if bool(args.crop64) != bool(args.crop128):
        raise ValueError('External locator checkpoints must be supplied as a pair')
    locators = []
    for branch in ('crop64', 'crop128'):
        existing = getattr(args, branch)
        skip = dict(reason='supplied_frozen_locator', branch=branch) if existing else None
        yield stage(branch, 'train', [*arm, *seed, '--branch', branch],
                    training_files, 'last.pt', skip)
        locators += ['--'+branch, existing or out/branch/'best.pt']

    yield stage('localize', 'localize', [*arm, *locators, '--padding-percent', '20'],
                ('candidates.json',))
    yield stage('audit-roi', 'audit-roi', [*arm, '--candidates', out/'localize'/'candidates.json'],
                ('boxes.json', 'localization_qc.json'))
    boxes = ['--boxes', out/'audit-roi'/'boxes.json']
    four = [*arm, *boxes, *locators, '--evaluate']
    for branch in ('all64', 'all128'):
        yield stage(branch, 'train', [*arm, *seed, *boxes, '--branch', branch],
                    training_files, 'last.pt')
        four += ['--'+branch, out/branch/'best.pt']
    for branch in ('all64', 'all128'):
        evaluation = [*arm, *boxes, '--branch', branch, '--checkpoint', out/branch/'best.pt']
        yield stage('evaluate-'+branch, 'evaluate', evaluation, ('evaluation.json',))
    yield stage('four-way', 'fuse', four, fusion_files, 'job.json')

    reviews = ['--reviews', args.reviews] if args.reviews else []
    yield stage('targets', 'build-targets', reviews, ('target-request.json', 'targets.json'),
                'target-request.json')
    targets = ['--targets', out/'targets']
    counts = auxiliary_counts(cache, out/'targets', args.supervision)
    # Formal init training requires a full patient batch; the blood encoder batches slices.
    init_skip = _shortfall('init', counts['init'], 4 if args.profile == 'paper' else 1)
    blood_skip = _shortfall('blood', counts['blood'], 1)
    six = list(four)
    for branch in ('init64', 'init128'):
        yield stage(branch, 'train-init', [*arm, *seed, *boxes, *targets, '--branch', branch],
                    training_files, 'last.pt', init_skip)
        six += ['--'+branch, out/branch/'best.pt']
    yield stage('six-way', 'fuse', six, fusion_files, 'job.json', init_skip)

    yield stage('blood', 'train-blood', [*arm, *seed, *targets], ('best.pt', 'blood-training.json'),
                'blood-contract.json', blood_skip)
    fused = 'four-way' if init_skip else 'six-way'
    blood_fusion = [*(four if init_skip else six), '--blood', out/'blood'/'best.pt']
    yield stage('with-blood', 'fuse', blood_fusion, fusion_files, 'job.json', blood_skip)
    if blood_skip is None:
        fused = 'with-blood'
    yield stage('vessels', 'refine-vessels', ['--fusion', out/fused, '--evaluate'],
                fusion_files, 'job.json')
