"""First Xu replication stage: independent multi-resolution locators and all-structure branches."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import re

from .contracts import atomic_json, digest, read_json, sha256_file
from .observability.status import operation_lock

BRANCHES = ('crop64','crop128','all64','all128')
PROTOCOL = 'xu2023-source-seg-1'


def side(branch):
    return 128 if branch.endswith('128') else 64


@lru_cache(maxsize=2)
def artifact_identity(cache):
    """Bind actual committed cache outputs as well as their preparation request."""
    artifacts = {}
    for name in (*cache.case_ids('train'), *cache.case_ids('development')):
        marker = read_json(cache.run_dir/'commits'/('process-'+name+'.json'))
        if marker['contract'] != cache.request_digest or marker['job']['id'] != 'process-'+name:
            raise ValueError('Case artifact identity differs from the prepared run')
        artifacts[name] = marker['artifacts']
    contract = read_json(cache.run_dir/'preprocessed/Dataset282_CHDSource/training_contract.json')
    return digest(dict(artifacts=artifacts, training=contract, fit=read_json(cache.run_dir/'fit.json')))


def input_id(cache, supervision, book=None):
    return digest(dict(protocol=PROTOCOL, cache=cache.request_digest,
                       artifacts=artifact_identity(cache),
                       geometry=sha256_file(Path(__file__).with_name('xu_geometry.py')),
                       supervision=supervision, roi_book=digest(book) if book is not None else None))


def frozen_predictor(path, **options):
    from .xu_training import load_predictor
    fingerprint = sha256_file(path)
    predict = load_predictor(path, **options)
    if sha256_file(path) != fingerprint:
        raise ValueError('Checkpoint changed while loading the predictor')
    return predict, fingerprint


def roi_book(path, cache, args):
    if args.branch.startswith('crop'):
        if path is not None:
            raise ValueError('Localization takes whole volumes')
        return None
    if path is None:
        raise ValueError('Run localize first and pass its boxes.json to --boxes')
    book = read_json(path)
    qc = read_json(Path(path).parent/'localization_qc.json')
    if (qc['book_digest'] != digest(book) or set(book['checkpoints']) != {'crop64','crop128'}
            or not all(re.fullmatch('[0-9a-f]{64}', h) for h in book['checkpoints'].values())):
        raise ValueError('Predicted box book or locator identity changed')
    if (book['protocol'] != PROTOCOL or book['request'] != cache.request_digest or
            book['locator_input'] != input_id(cache,args.supervision) or
            book['profile'] != args.profile or book['supervision'] != args.supervision):
        raise ValueError('ROI book is from another experiment')
    if set(book['boxes']) != set(cache.case_ids('train')) | set(cache.case_ids('development')):
        raise ValueError('Incomplete predicted ROI book')
    return book


def train(cache, args, out):
    from .xu_training import train_branch
    book = roi_book(args.boxes, cache, args)
    strict = args.supervision=='complete'
    train_names = cache.case_ids('train', complete_only=strict)
    dev_names = cache.case_ids('development', complete_only=strict)
    # The real three-case acceptance fixture has no completely annotated dev case.
    if args.profile=='smoke' and not dev_names:
        dev_names = cache.case_ids('development')

    def load(name, branch):
        shape = (side(branch),)*3
        box = book['boxes'][name] if book is not None else None
        return cache.read_image(name,shape,box), cache.read_label(name,shape,box)

    def annotations(names):
        return {name:cache.supervision[name]['unknown_classes'] for name in names}

    result = train_branch(branch=args.branch, output_dir=out, train_cases=annotations(train_names),
                          validation_cases=annotations(dev_names), load_sample=load,
                          data_identity=input_id(cache,args.supervision,book), profile=args.profile,
                          supervision=args.supervision, device=args.device, resume=args.resume, seed=args.seed)
    return dict(command='train', branch=args.branch, train=len(train_names), validation=len(dev_names),
                profile=args.profile, supervision=args.supervision, result=result)


def localize(cache, args, out):
    from .xu_geometry import prediction_box
    destination = out/'boxes.json'
    if destination.exists():
        raise ValueError('Use a fresh output directory for a new ROI book')
    paths = {'crop64':args.crop64, 'crop128':args.crop128}
    models = {k:frozen_predictor(p,branch=k,profile=args.profile,supervision=args.supervision,
                                data_identity=input_id(cache,args.supervision),device=args.device)
              for k,p in paths.items()}
    boxes = {}
    full_volume = []
    names = (*cache.case_ids('train'), *cache.case_ids('development'))
    for idx,name in enumerate(names,1):
        predicted = [predict(cache.read_image(name,(side(k),)*3)) for k,(predict,_) in models.items()]
        shape = cache.geometry(name).cache_shape
        box = prediction_box(*predicted,shape)
        boxes[name] = [[int(lo),int(hi)] for lo,hi in box]
        if all(lo==0 and hi==n for (lo,hi),n in zip(box,shape)):
            full_volume.append(name)
        print(f'Localized {idx}/{len(names)}', flush=True)
    book = dict(protocol=PROTOCOL,profile=args.profile,supervision=args.supervision,
                locator_input=input_id(cache,args.supervision),request=cache.request_digest,
                checkpoints={k:sha for k,(_,sha) in models.items()},
                convention='cache axes; inclusive start, exclusive stop',boxes=boxes)
    atomic_json(out/'localization_qc.json',
                dict(cases=len(names),full_volume_cases=full_volume,book_digest=digest(book)))
    if args.profile=='paper' and len(full_volume)==len(names):
        raise ValueError('Every predicted crop covers the full volume; review localizers before ROI training')
    atomic_json(destination,book)
    return dict(command='localize', cases=len(names),full_volume=len(full_volume),file=str(destination))


def evaluate(cache, args, out):
    import nibabel as nib
    import numpy as np
    from .xu_evaluation import class_metrics, summarize
    if (out/'evaluation.json').exists() or (out/'cases').exists():
        raise ValueError('Evaluation requires a fresh output directory')
    book = roi_book(args.boxes, cache, args)
    predictor, checkpoint_hash = frozen_predictor(args.checkpoint, branch=args.branch, profile=args.profile, supervision=args.supervision,
                               data_identity=input_id(cache,args.supervision,book), device=args.device)
    (out/'cases').mkdir()
    by_case = {}
    names = cache.case_ids('development')
    for idx,name in enumerate(names,1):
        geo = cache.geometry(name)
        box = book['boxes'][name] if book is not None else None
        label = predictor(cache.read_image(name, (side(args.branch),)*3, box))
        native = geo.restore_label(label,box)
        # Access the reference only after making the prediction and determining the crop.
        metrics = class_metrics(cache.read_native_label(name),native,geo.spacing_xyz,cache.supervision[name])
        by_case[name] = metrics
        atomic_json(out/'cases'/(name+'.json'),metrics)
        if args.save_predictions:
            affine = np.diag([*geo.spacing_xyz,1.])
            image = nib.Nifti1Image(native,affine)
            image.header.set_xyzt_units('mm')
            nib.save(image,out/'cases'/(name+'.nii.gz'))
        print(f'Evaluated {idx}/{len(names)}',flush=True)
    primary = cache.case_ids('development',complete_only=True)
    report = dict(protocol=PROTOCOL, request=cache.request_digest, branch=args.branch, profile=args.profile,
                  supervision=args.supervision, checkpoint_sha256=checkpoint_hash,
                  evaluation='development', full_annotation=summarize([by_case[n] for n in primary]),
                  observed_structure=summarize(list(by_case.values())))
    atomic_json(out/'evaluation.json',report)
    return dict(command='evaluate', cases=len(names), fully_annotated=len(primary), file=str(out/'evaluation.json'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    for name in ('inspect','train','localize','evaluate'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--prepared',type=Path,required=True,help='Completed hu-marginal run directory')
        cmd.add_argument('--technical-subset',action='store_true',help='Explicitly admit technical acceptance data')
        if name=='inspect':
            continue
        cmd.add_argument('--output',type=Path,required=True)
        cmd.add_argument('--profile',choices=('paper','smoke'),default='paper')
        cmd.add_argument('--supervision',choices=('complete','marginal'),default='complete')
        cmd.add_argument('--device',default='cpu')
        if name in ('train','evaluate'):
            cmd.add_argument('--branch',choices=BRANCHES,required=True)
            cmd.add_argument('--boxes',type=Path)
        if name=='train':
            cmd.add_argument('--resume',action='store_true')
            cmd.add_argument('--seed',type=int,default=2023)
        if name=='localize':
            cmd.add_argument('--crop64',type=Path,required=True)
            cmd.add_argument('--crop128',type=Path,required=True)
        if name=='evaluate':
            cmd.add_argument('--checkpoint',type=Path,required=True)
            cmd.add_argument('--save-predictions',action='store_true')
    args=p.parse_args(argv)
    from .xu_geometry import PreparedCache
    cache=PreparedCache(args.prepared,allow_technical_subset=args.technical_subset)
    if args.command=='inspect':
        result=dict(protocol=PROTOCOL, scope=cache.scope, request=cache.request_digest,
                    cases={role:dict(all=len(cache.case_ids(role)), complete=len(cache.case_ids(role,complete_only=True)))
                           for role in ('train','development')},
                    auxiliary_status='init64/init128 require verified proximal-vessel supervision')
    else:
        out=args.output.resolve()
        root=args.prepared.resolve()
        if out==root or root in out.parents or out in root.parents:
            raise ValueError('Choose a separate experiment output outside the preprocessing cache')
        out.mkdir(parents=True,exist_ok=True)
        with operation_lock(out,'xu-'+args.command,out):
            result={'train':train,'localize':localize,'evaluate':evaluate}[args.command](cache,args,out)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
