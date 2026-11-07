"""Weighted fusion and boundary-constrained growing on a single frozen split."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import json
import os
import platform
import shutil
import tempfile
import uuid

import numpy as np
from scipy import ndimage

from .contracts import atomic_json, digest, read_json, sha256_file
from .observability.status import operation_lock
from .xu_geometry import _box_in, _resize
from .xu_inference_pool import case_plan, map_cases, record_plan, shard_identity


FINE_WEIGHTS = {"crop64": 1, "crop128": 1, "all64": 2, "all128": 2}
INIT = ("init64", "init128")
FUSION_PROTOCOL = "xu-fusion-1"


def grid_prediction(values, full_shape, source_box, target_box, shape):
    """Nearest sampling at target voxel centres in crop-edge coordinates, without first rounding the ROI."""
    values = np.asarray(values)
    source = _box_in(source_box, full_shape)
    target = _box_in(target_box, full_shape)
    if values.ndim != 3 or len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError("Invalid prediction geometry")
    indices = []
    for (sl, sr), (tl, tr), count, n in zip(source, target, values.shape, shape):
        if tl < sl or tr > sr:
            raise ValueError("Target must be enclosed by source ROI")
        centres = 2 * n * (tl - sl) + (2 * np.arange(n, dtype=np.int64) + 1) * (tr - tl)
        indices.append((count * centres) // (2 * n * (sr - sl)))
    return values[np.ix_(*indices)]


def _labels(arr, shape, maximum):
    arr = np.asarray(arr)
    if (
        arr.shape != shape
        or arr.dtype.kind not in "iu"
        or arr.size == 0
        or arr.min() < 0
        or arr.max() > maximum
    ):
        raise ValueError("Invalid prediction shape, dtype or label set")
    return arr


def vote(predictions, anatomy=None):
    """Six heterogeneous hard-label votes. A proximal vote never supports two fine classes."""
    fine = set(FINE_WEIGHTS)
    if set(predictions) not in (fine, fine | set(INIT)):
        raise ValueError(
            "Expected the four fine outputs, optionally both proximal outputs"
        )
    shape = predictions["all128"].shape
    if len(shape) != 3:
        raise ValueError("Expected 3D predictions")
    votes = np.zeros((11, *shape), dtype=np.uint8)
    for branch, weight in FINE_WEIGHTS.items():
        pred = _labels(predictions[branch], shape, 10)
        for label in range(11):
            votes[label] += (pred == label) * np.uint8(weight)
    proximal_abstentions = 0
    if "init64" in predictions:
        ao, pa = anatomy["AO"], anatomy["PA"]
        # Establish within-group evidence before either init output contributes.
        ao_score, pa_score = votes[ao].copy(), votes[pa].copy()
        decision = np.where(
            ao_score > pa_score, ao, np.where(pa_score > ao_score, pa, 0)
        )
        prior = predictions["all128"]
        tied_supported = (ao_score == pa_score) & (ao_score > 0)
        for label in (ao, pa):
            decision[tied_supported & (prior == label)] = label
        for branch in INIT:
            coarse = _labels(predictions[branch], shape, 6)
            for i, k in enumerate(("LV", "RV", "LA", "RA", "MYO"), 1):
                votes[anatomy[k]] += (coarse == i) * np.uint8(2)
            proximal = coarse == 6
            for label in (ao, pa):
                votes[label] += (proximal & (decision == label)) * np.uint8(2)
            proximal_abstentions += int(np.count_nonzero(proximal & (decision == 0)))
            # Coarse background includes all distal vessel classes: no fine-class vote there.
    best = votes.max(axis=0)
    result = votes.argmax(axis=0).astype(np.uint8)
    ties = (votes == best).sum(axis=0) > 1
    # If tied, prefer the finest all-structure predictor when among the winners.
    preferred = predictions["all128"]
    supports = np.take_along_axis(votes, preferred[None], axis=0)[0]
    result[ties & (supports == best)] = preferred[ties & (supports == best)]
    return result, dict(
        tied_voxels=int(ties.sum()), proximal_abstaining_votes=proximal_abstentions
    )


def blood_grow(labels, blood, myocardium):
    """Grow nonmyocardial seeds through blood. Predicted boundary is an explicit barrier."""
    from skimage.segmentation import watershed

    blood = _labels(blood, labels.shape, 2)
    labels = _labels(labels, labels.shape, 10)
    counts = np.bincount(blood.ravel(), minlength=3)
    if not counts.all():
        return labels.copy(), dict(
            status="abstained_missing_blood_class",
            changed=0,
            blood_class_counts=counts.tolist(),
        )
    region = (blood == 1) & (labels != myocardium)
    seeds = np.where(region, labels, 0).astype(np.uint8)
    if not seeds.any():
        return labels.copy(), dict(status="abstained_no_supported_seed", changed=0)
    grown = watershed(
        np.zeros(labels.shape, np.uint8), markers=seeds, connectivity=1, mask=region
    )
    result = labels.copy()
    result[region] = grown[region]
    # Remove boundary only beside supported blood regions; leave unsupported components alone.
    barrier = (
        (blood == 2) & (result != myocardium) & ndimage.binary_dilation(grown != 0)
    )
    result[barrier] = 0
    result, islands = _enclosed_islands(result, blood, myocardium)
    return result, dict(
        status="applied",
        changed=int(np.count_nonzero(result != labels)),
        removed_boundary=int(np.count_nonzero(barrier & (labels != 0))),
        unseeded_blood=int(np.count_nonzero(region & (grown == 0))),
        islands=islands,
        blood_class_counts=counts.tolist(),
    )


def _enclosed_islands(labels, blood, myocardium):
    """Reassign only a small secondary island completely surrounded by one blood class."""
    result = labels.copy()
    counts = dict(reassigned=0, conflicting_contacts=0, skipped_fragmented_classes=0)
    for cls in range(1, 11):
        if cls == myocardium:
            continue
        cc, n = ndimage.label(labels == cls)
        if n < 2:
            continue
        if n > 4096:
            counts["skipped_fragmented_classes"] += 1
            continue
        sizes = np.bincount(cc.ravel())[1:]
        threshold = 0.01 * int(sizes.max())
        bounds = ndimage.find_objects(cc)
        for idx in np.flatnonzero(sizes <= threshold):
            box = bounds[idx]
            if any(s.start == 0 or s.stop == dim for s, dim in zip(box, labels.shape)):
                continue
            cut = tuple(slice(s.start - 1, s.stop + 1) for s in box)
            island = cc[cut] == idx + 1
            neighbors = ndimage.binary_dilation(island) & ~island
            if np.any(blood[cut][neighbors | island] != 1):
                continue
            touching = np.unique(labels[cut][neighbors])
            if len(touching) == 1 and touching[0] not in (0, myocardium):
                result[cut][island] = touching[0]
                counts["reassigned"] += 1
            elif len(touching) > 1:
                counts["conflicting_contacts"] += 1
    return result, counts


def _refine_shape(box, limit=128**3 * 6):
    # Use cache spacing wherever possible, capping the high-resolution refinement working set.
    ext = np.array([hi - lo for lo, hi in box], dtype=np.int64)
    scale = min(1.0, (limit / int(ext.prod())) ** (1 / 3))
    return tuple(np.maximum(1, np.floor(ext * scale)).astype(int))


def _atomic_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + uuid.uuid4().hex + ".npy")
    try:
        with tmp.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_array(path, expected):
    if sha256_file(path) != expected:
        raise ValueError("Changed prediction artifact: " + str(path.name))
    return np.load(path, allow_pickle=False, mmap_mode="r")


def _open_job(out, identity, resume):
    manifest = out / "job.json"
    if manifest.exists():
        if not resume or read_json(manifest) != identity:
            raise ValueError("Use fresh output or resume the identical fusion job")
    else:
        if any(p.name not in (".runtime",) for p in out.iterdir()):
            raise ValueError("Unrecognized files in experiment output directory")
        atomic_json(manifest, identity)


def _ensure_branch(cache, args, book, branch, checkpoint, expected, names, dest, *, plan):
    from .xu_workflow import frozen_predictor, input_id, side

    marker = dest / "complete.json"
    if marker.exists():
        record = read_json(marker)
        if record["checkpoint"] != expected or set(record["files"]) != set(names):
            raise ValueError("Invalid completed prediction cache")
        return record["files"]
    dest.mkdir(parents=True, exist_ok=True)
    def make_worker(device):
        if branch in INIT:
            from .xu_init import load_init_predictor
            predict = load_init_predictor(checkpoint,cache=cache,branch=branch,profile=args.profile,
                                          supervision=args.supervision,book=book,device=device)
            consumed = sha256_file(checkpoint)
        else:
            identity = input_id(cache,args.supervision,None if branch.startswith("crop") else book)
            predict, consumed = frozen_predictor(checkpoint,branch=branch,profile=args.profile,
                              supervision=args.supervision,device=device,data_identity=identity)
        if consumed != expected:
            raise ValueError('Branch checkpoint changed during loading')
        def process(name):
            roi = book["boxes"][name]
            source = None if branch.startswith("crop") else roi
            n = side(branch)
            pred = _labels(predict(cache.read_image(name,(n,n,n),source)), (n,n,n), 6 if branch in INIT else 10)
            result = grid_prediction(pred,cache.geometry(name).cache_shape,source,roi,(128,128,128))
            path = dest/(name+".npy")
            _atomic_array(path,result)
            print(f"{branch} predicted {name} on {device}",flush=True)
            return sha256_file(path)
        return process
    outputs = map_cases(plan,make_worker)
    atomic_json(marker, dict(checkpoint=expected, files=outputs))
    return outputs


def _prediction_context(cache, args, names, job):
    """Bind the actual CT, geometry and numeric implementation before trusting a previous forward."""
    import torch

    device = torch.device(args.device)
    runtime = dict(torch=str(torch.__version__), numpy=np.__version__, scipy=version('scipy'),
                   blosc2=version('blosc2'), threads=torch.get_num_threads(), device=device.type,
                   deterministic=torch.are_deterministic_algorithms_enabled(),
                   matmul_precision=torch.get_float32_matmul_precision(), machine=platform.machine(),
                   host=platform.node(), build=digest(torch.__config__.show()),
                   cpu_capability=torch.backends.cpu.get_cpu_capability())
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        physical = getattr(props, 'uuid', None)
        if not physical:
            raise ValueError('Shared GPU predictions require a verified device UUID')
        runtime.update(uuid=str(physical), gpu=props.name, capability=[props.major, props.minor],
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       cudnn_tf32=torch.backends.cudnn.allow_tf32,
                       cudnn_deterministic=torch.backends.cudnn.deterministic,
                       cudnn_benchmark=torch.backends.cudnn.benchmark,
                       precision_environment={k:os.environ.get(k) for k in
                           ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','CUBLAS_WORKSPACE_CONFIG')})
    geometry = {}
    for name in names:
        # A cache hit skips read_image. Verify the committed CT bytes on that path too.
        cache._artifact(f'process-{name}', f'{cache._folder}/{name}.b2nd')
        geometry[name] = asdict(cache.geometry(name))
    sources = ('xu_fusion.py', 'xu_training.py', 'xu_models.py', 'xu_init.py',
               'xu_geometry.py', 'xu_workflow.py', 'xu_roi.py', 'xu_roi_workflow.py', 'xu_inference_pool.py')
    if 'inference_shards' in job:
        runtime['inference_shards'] = job['inference_shards']
    return dict(protocol='xu-frozen-branch-prediction-1', prepared=job['request'],
                artifacts=job['cache_artifacts'], boxes=job['boxes'], cases=list(names),
                geometry=digest(geometry), profile=args.profile, supervision=args.supervision,
                target_grid=[128]*3, runtime=runtime,
                implementation={name:sha256_file(Path(__file__).with_name(name)) for name in sources})


def _shared_branch(root, identity, dest, names, produce):
    """Publish the complete branch as a directory; interrupted writers cannot publish partial sets."""
    key = digest(identity)
    entry = root/'entries'/key
    checkpoint = identity['checkpoint']
    with operation_lock(root/'locks'/key, 'frozen-branch-cache'):
        created = not entry.exists()
        if created:
            entry.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.building-', dir=entry.parent) as scratch:
                building = Path(scratch)/'entry'
                files = produce(building)
                record = dict(identity=identity, files=files)
                atomic_json(building/'manifest.json', dict(record=record, digest=digest(record)))
                os.replace(building, entry)
        manifest = read_json(entry/'manifest.json')
        record = manifest['record']
        files = record['files']
        if (entry.is_symlink() or manifest.get('digest') != digest(record)
                or record['identity'] != identity or set(files) != set(names)
                or read_json(entry/'complete.json') != dict(checkpoint=checkpoint, files=files)):
            raise ValueError('Invalid shared prediction manifest')
        # Stage exports keep their existing layout. On one filesystem hardlinks avoid a second copy.
        dest.mkdir(parents=True, exist_ok=True)
        for name, expected in files.items():
            source, target = entry/(name+'.npy'), dest/(name+'.npy')
            if source.is_symlink() or sha256_file(source) != expected:
                raise ValueError('Changed shared prediction artifact')
            if target.exists():
                if target.is_symlink() or sha256_file(target) != expected:
                    raise ValueError('Changed stage prediction artifact')
                continue
            try:
                os.link(source, target)
            except OSError:
                tmp = dest/('.' + uuid.uuid4().hex + '.tmp')
                try:
                    shutil.copyfile(source, tmp)
                    if sha256_file(tmp) != expected:
                        raise ValueError('Changed prediction while copying')
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
        marker = dest/'complete.json'
        if marker.exists():
            previous = read_json(marker)
            if (previous.get('checkpoint') != checkpoint or previous.get('files') != files
                    or previous.get('cache_key') != key or previous.get('outcome') not in ('hit','miss')):
                raise ValueError('Stage prediction marker differs from the shared cache')
            outcome = previous['outcome']
        else:
            outcome = 'miss' if created else 'hit'
            atomic_json(marker, dict(checkpoint=checkpoint, files=files, cache_key=key, outcome=outcome))
    return files, dict(key=key, outcome=outcome, cases=len(files))


def _save_case(out, name, geo, box, small, qc, cache, evaluation):
    import nibabel as nib

    target = out / "cases"
    path = target / (name + ".npy")
    _atomic_array(path, small)
    native = geo.restore_label(small, box)
    nii = target / (name + ".nii.gz")
    tmp = target / ("." + uuid.uuid4().hex + ".nii.gz")
    try:
        img = nib.Nifti1Image(native, np.diag([*geo.spacing_xyz, 1.0]))
        img.header.set_xyzt_units("mm")
        nib.save(img, tmp)
        os.replace(tmp, nii)
    finally:
        tmp.unlink(missing_ok=True)
    entry = dict(
        box=box,
        shape=list(small.shape),
        label_sha256=sha256_file(path),
        native_sha256=sha256_file(nii),
        qc=qc,
    )
    if evaluation:
        from .xu_evaluation import class_metrics

        entry["metrics"] = class_metrics(
            cache.read_native_label(name),
            native,
            geo.spacing_xyz,
            cache.supervision[name],
        )
    return entry


def _finish(out, cache, job, records, evaluate):
    if evaluate:
        from .xu_evaluation import summarize

        primary = set(cache.case_ids("development", complete_only=True))
        atomic_json(
            out / "evaluation.json",
            dict(
                full_annotation=summarize(
                    [v["metrics"] for n, v in records.items() if n in primary]
                ),
                observed_structure=summarize([v["metrics"] for v in records.values()]),
            ),
        )
    # Metric keys are integers in memory and strings on disk. Hash the stored JSON form.
    final = dict(
        job=digest(job),
        request=cache.request_digest,
        cases=records,
        cases_digest=digest(json.loads(json.dumps(records, allow_nan=False))),
        evaluation_sha256=sha256_file(out / "evaluation.json") if evaluate else None,
    )
    atomic_json(out / "complete.json", final)
    return dict(
        command=job["command"], cases=len(records), file=str(out / "complete.json")
    )


def _completed(out, job, names, verify_files):
    marker = read_json(out / "complete.json")
    if (
        marker["job"] != digest(job)
        or marker["request"] != job["request"]
        or set(marker["cases"]) != set(names)
        or marker.get("cases_digest") != digest(marker["cases"])
    ):
        raise ValueError("Invalid completed prediction set")
    if job["evaluate"] and marker["evaluation_sha256"] != sha256_file(
        out / "evaluation.json"
    ):
        raise ValueError("Evaluation report changed")
    if verify_files:
        for name, entry in marker["cases"].items():
            for suffix, field in (
                (".npy", "label_sha256"),
                (".nii.gz", "native_sha256"),
            ):
                if sha256_file(out / "cases" / (name + suffix)) != entry[field]:
                    raise ValueError("Completed prediction changed")
    return marker


def _fuse(cache, args, out):
    from .xu_workflow import roi_book, artifact_identity
    from .xu_targets import anatomy

    book = roi_book(args.boxes, cache, args)
    names = cache.case_ids("development")
    plan = case_plan(args.device,names,cache,evaluation=args.evaluate)
    paths = {k: getattr(args, k) for k in FINE_WEIGHTS}
    if args.init64 or args.init128:
        if not args.init64 or not args.init128:
            raise ValueError(
                "Both proximal checkpoints are required for six-way fusion"
            )
        paths.update({k: getattr(args, k) for k in INIT})
    hashes = {k: sha256_file(p) for k, p in paths.items()}
    if {k: hashes[k] for k in ("crop64", "crop128")} != book["checkpoints"]:
        raise ValueError("ROI book and supplied locators differ")
    if args.blood:
        hashes["blood"] = sha256_file(args.blood)
    mapping = anatomy(cache) if args.blood or "init64" in paths else None
    job = dict(
        command="fuse",
        protocol=FUSION_PROTOCOL,
        implementation=sha256_file(Path(__file__)),
        request=cache.request_digest,
        profile=args.profile,
        supervision=args.supervision,
        boxes=digest(book),
        checkpoints=hashes,
        evaluate=args.evaluate,
        cache_artifacts=artifact_identity(cache),
    )
    shared = getattr(args, 'prediction_cache', None)
    if len(plan.requested)>1:
        job['inference_shards'] = shard_identity(plan)
    if shared is not None:
        shared = shared.resolve()
        if any(shared == path or shared in path.parents or path in shared.parents
               for path in (out.resolve(), cache.run_dir)):
            raise ValueError('Choose a separate prediction cache outside stage and prepared directories')
        job['prediction_cache'] = str(shared)
    _open_job(out, job, args.resume)
    if (out / "complete.json").exists():
        _completed(out, job, names, True)
        return dict(
            command=job["command"],
            cases=len(names),
            resumed=True,
            file=str(out / "complete.json"),
        )
    dirs = out / "branches"
    record_plan(out,plan)
    files = {}
    if shared is None:
        for k, p in paths.items():
            files[k] = _ensure_branch(cache, args, book, k, p, hashes[k], names, dirs/k,plan=plan)
    else:
        from functools import partial
        from .xu_workflow import side
        context = _prediction_context(cache, args, names, job)
        reuse = {}
        for k, p in paths.items():
            identity = dict(context, branch=k, checkpoint=hashes[k], prediction_side=side(k))
            producer = partial(_ensure_branch, cache, args, book, k, p, hashes[k], names, plan=plan)
            files[k], reuse[k] = _shared_branch(shared, identity, dirs/k, names, producer)
        hits = sum(row['outcome'] == 'hit' for row in reuse.values())
        atomic_json(out/'prediction-cache.json', dict(branches=reuse, hits=hits, misses=len(reuse)-hits))
    def make_fusion_worker(device):
        blood_predict = None
        if args.blood:
            from .xu_blood import load_blood_predictor
            blood_predict = load_blood_predictor(cache,args.blood,device=device,profile=args.profile,
                                                 supervision=args.supervision)
            if sha256_file(args.blood) != hashes['blood']:
                raise ValueError('Blood checkpoint changed during loading')
        def process(name):
            geo, roi = cache.geometry(name), book['boxes'][name]
            pred = {k:_load_array(dirs/k/(name+'.npy'),files[k][name]) for k in paths}
            small, qc = vote(pred,mapping)
            if blood_predict is not None:
                shape = _refine_shape(roi)
                blood = blood_predict(name,roi,shape)
                small, details = blood_grow(_resize(small,shape,0),blood,mapping['MYO'])
                qc['blood'] = details
            entry = _save_case(out,name,geo,roi,small,qc,cache,args.evaluate)
            print(f"Fused {name} on {device}",flush=True)
            return entry
        return process
    records = map_cases(plan,make_fusion_worker)
    return _finish(out, cache, job, records, args.evaluate)


def _refine(cache, args, out):
    from .xu_targets import anatomy
    from .xu_vessels import refine_vessels

    if (
        args.fusion.resolve() == out
        or out in args.fusion.resolve().parents
        or args.fusion.resolve() in out.parents
    ):
        raise ValueError("Use separate fusion and vessel output directories")
    original = read_json(args.fusion / "job.json")
    names = cache.case_ids("development")
    markers = _completed(args.fusion, original, names, False)
    from .xu_workflow import artifact_identity

    if (
        original["command"] != "fuse"
        or original["protocol"] != FUSION_PROTOCOL
        or markers["job"] != digest(original)
        or markers["request"] != cache.request_digest
        or original["cache_artifacts"] != artifact_identity(cache)
        or set(markers["cases"]) != set(names)
    ):
        raise ValueError("Expected completed fusion on this frozen cache")
    job = dict(
        command="refine-vessels",
        protocol=FUSION_PROTOCOL,
        source=digest(markers),
        request=cache.request_digest,
        implementation=sha256_file(Path(__file__).with_name("xu_vessels.py")),
        evaluate=args.evaluate,
    )
    _open_job(out, job, args.resume)
    if (out / "complete.json").exists():
        _completed(out, job, names, True)
        return dict(
            command=job["command"],
            cases=len(names),
            resumed=True,
            file=str(out / "complete.json"),
        )
    mapping = anatomy(cache)
    records = {}
    for i, name in enumerate(names, 1):
        record = markers["cases"][name]
        source = _load_array(
            args.fusion / "cases" / (name + ".npy"), record["label_sha256"]
        )
        geo = cache.geometry(name)
        box = _box_in(record["box"], geo.cache_shape)
        from .xu_roi import measure_box

        if (
            list(source.shape) != record["shape"]
            or not measure_box(box, geo)["accepted"]
        ):
            raise ValueError("Invalid vessel ROI input")
        spacing = [
            geo.spacing_xyz[2 - axis] * (stop - start) / n * (hi - lo) / m
            for axis, (start, stop), n, (lo, hi), m in zip(
                geo.transpose_forward,
                geo.bbox_transposed,
                geo.cache_shape,
                box,
                source.shape,
            )
        ]
        result, qc = refine_vessels(source, spacing, mapping)
        records[name] = _save_case(
            out, name, geo, box, result, qc, cache, args.evaluate
        )
        print(f"Refined vessels {i}/{len(names)}", flush=True)
    return _finish(out, cache, job, records, args.evaluate)


def register(subparsers):
    for command in ("fuse", "refine-vessels"):
        p = subparsers.add_parser(command)
        p.add_argument("--prepared", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--technical-subset", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument(
            "--evaluate",
            action="store_true",
            help="Evaluate observed structures on the native mm grid",
        )
        if command == "refine-vessels":
            p.add_argument(
                "--fusion", required=True, type=Path, help="Completed fuse output"
            )
            continue
        p.set_defaults(branch="fusion")
        p.add_argument("--boxes", required=True, type=Path)
        p.add_argument("--profile", choices=("smoke", "paper"), default="paper")
        p.add_argument(
            "--supervision", choices=("complete", "marginal"), default="complete"
        )
        p.add_argument("--device", default="cpu")
        p.add_argument("--prediction-cache", type=Path,
                       help="Share verified frozen branch predictions between fusion stages")
        for branch in (*FINE_WEIGHTS, *INIT, "blood"):
            p.add_argument("--" + branch, required=branch in FINE_WEIGHTS, type=Path)


def run(cache, args, out):
    return (
        _fuse(cache, args, out) if args.command == "fuse" else _refine(cache, args, out)
    )
