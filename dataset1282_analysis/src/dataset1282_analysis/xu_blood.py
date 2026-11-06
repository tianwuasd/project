"""Train the separate 2D and temporal blood-pool networks on prepared axial slices."""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from .contracts import atomic_json, read_json, sha256_file
from .xu_blood_grid import AxialViews


# Reviewed at 13ceecc2. Unscoped v1 used every case with a known blood target.
_V1_SOURCE = {
    'xu_blood.py': 'e70d0f19b0c292b4417f13a8c6d6b97160396df99fb5380b2a1861b508eb8abc',
    'xu_blood_grid.py': '3d3f354441941584e14c54160bf89722f250d60a82a8f03fb40acd8f3cde8a99',
    'xu_blood_models.py': '3e8b2508bbc470e9355eaf89490e3ffb5213d99a5e9eaaddc42d2f4f62357f8f',
    'xu_geometry.py': '4500549b63cd6dc38551a177193137c3b2da64faa582d5d182ec84be9c805d79',
}


def register(commands):
    for name in ('train-blood', 'predict-blood'):
        p = commands.add_parser(name)
        p.add_argument('--prepared', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--technical-subset', action='store_true')
        p.add_argument('--profile', choices=('paper','smoke'), default='paper')
        p.add_argument('--device', default='cpu')
        if name == 'train-blood':
            p.add_argument('--targets', required=True, type=Path, help='Completed build-targets directory')
            p.add_argument('--supervision',choices=('marginal','complete'),default='marginal')
            p.add_argument('--resume', action='store_true')
            p.add_argument('--seed', type=int, default=2023)
            p.add_argument('--technical-unvalidated', action='store_true',
                           help='Smoke on a technical subset lacking any eligible validation case')
        else:
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--case', required=True, help='Prepared development case')


def _source():
    return {name:sha256_file(Path(__file__).with_name(name)) for name in
            ('xu_blood.py','xu_blood_grid.py','xu_blood_models.py','xu_geometry.py')}


def blood_cohort(cache, ids, supervision, store=None):
    from .xu_targets import blood_eligible
    if supervision not in ('complete','marginal'):
        raise ValueError('Unknown blood supervision arm')
    eligible = {}
    for role in ('train','development'):
        selected = []
        for name in cache.case_ids(role):
            info = cache.supervision[name]
            known = blood_eligible(info,ids)
            if store is not None and known != store.has(name,'blood'):
                raise ValueError('Blood eligibility disagrees with target manifest')
            if known and (supervision == 'marginal' or len(info['present_classes']) == 10):
                selected.append(name)
        eligible[role] = selected
    return eligible


def _contract(cache, targets, supervision, profile, seed, eligible):
    from .xu_workflow import artifact_identity
    return dict(method='xu-blood-v2', request=cache.request_digest, artifacts=artifact_identity(cache),
                targets=targets, source=_source(), supervision=supervision, profile=profile,
                seed=seed, **eligible)


def _matches(stored, expected):
    if stored == expected:
        return True
    if expected['supervision'] != 'marginal' or stored.get('method') != 'xu-blood-v1':
        return False
    if any(expected['source'][k] != v for k,v in _V1_SOURCE.items() if k != 'xu_blood.py'):
        return False
    legacy = dict(expected, method='xu-blood-v1', source=_V1_SOURCE)
    del legacy['supervision']
    return stored == legacy


def _objective(scores, target):
    import torch
    from torch.nn import functional as F
    valid = target >= 0
    probs = scores.float().softmax(1) * valid[:,None]
    dice = []
    for c in range(3):
        ref = target == c
        dice.append((2*(probs[:,c]*ref).sum()+1e-6)/(probs[:,c].sum()+ref.sum()+1e-6))
    weights = scores.new_tensor([1.,2.,16.])
    return F.cross_entropy(scores.float(),target,ignore_index=-1) + 1 - (weights*torch.stack(dice)).sum()/weights.sum()


def _batch(views, centres, temporal, device, augment):
    import torch
    from torch.nn import functional as F
    windows = [views.window(z) if temporal else [z] for z in centres]
    images = torch.from_numpy(np.stack([[views.image(i) for i in w] for w in windows])).to(device)
    labels = torch.from_numpy(np.stack([views.label(z) for z in centres])).long().to(device)
    if augment is not None:
        # Rotate/flip all slices and the target together; input padding is never supervised.
        matrices = []
        for _ in centres:
            angle = augment.uniform(-math.pi/12, math.pi/12)
            flip = augment.choice([-1, 1])
            matrices.append([[flip*math.cos(angle), -math.sin(angle), 0],
                             [flip*math.sin(angle), math.cos(angle), 0]])
        grid = F.affine_grid(images.new_tensor(matrices), images.shape, align_corners=False)
        images = F.grid_sample(images,grid,align_corners=False)
        labels = F.grid_sample((labels[:,None]+1).float(),grid,mode='nearest',align_corners=False)[:,0].long()-1
    return images, labels


def _epoch(cache, store, names, encoder, sequence, device, stage, profile, seed, optimizer=None):
    import torch
    training = optimizer is not None
    temporal = stage == 'sequence'
    encoder.train(training and not temporal)
    sequence.train(training and temporal)
    rng = np.random.default_rng(seed)
    order = list(names)
    if training:
        rng.shuffle(order)
    batch = 4 if profile=='paper' and not temporal else 1
    loss_sum, count = 0., 0
    for pos, name in enumerate(order,1):
        views = AxialViews(cache,name,store)
        centres = np.arange(views.zyx[0])
        if profile=='smoke':
            centres = np.array([len(centres)//2])
        if training:
            rng.shuffle(centres)
        for idx in range(0,len(centres),batch):
            images, labels = _batch(views,centres[idx:idx+batch],temporal,device,rng if training else None)
            if not (labels >= 0).any():
                continue
            with torch.set_grad_enabled(training):
                if temporal:
                    with torch.no_grad():
                        frames = [encoder.features(images[:,i:i+1]) for i in range(7)]
                    scores = sequence(frames)
                else:
                    scores = encoder(images)
                loss = _objective(scores,labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite blood-pool loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
            count += len(images)
        views.image.cache_clear()
        print(json.dumps(dict(stage=stage,phase='train' if training else 'development',
                              cases_completed=pos,total=len(order),slices=count)),flush=True)
    if not count:
        raise ValueError('No supervised blood-pool slices')
    return loss_sum/count


def _train_stage(cache, store, contract, out, encoder, sequence, stage, device):
    import torch
    from .xu_training import learning_rate, _save
    epochs = {'encoder':12,'sequence':3}[stage] if contract['profile']=='paper' else 1
    model = encoder if stage=='encoder' else sequence
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
    last = out/(stage+'-last.pt')
    best = out/(stage+'-selected.pt')
    start, history, selected = 0, [], 0
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=True)
        if state['contract']!=contract or state['stage']!=stage:
            raise ValueError('Blood checkpoint contract changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, history, selected = state['epoch'], state['history'], state['selected']
        if start==selected:
            _save(best,dict(contract=contract,epoch=start,stage=stage,model=state['model']))
        del state
    for e in range(start,epochs):
        lr=learning_rate(e,epochs)
        for group in optimizer.param_groups:
            group['lr']=lr
        loss=_epoch(cache,store,contract['train'],encoder,sequence,device,stage,contract['profile'],
                    contract['seed']+e,optimizer)
        dev=contract['development']
        val=(_epoch(cache,store,dev,encoder,sequence,device,stage,contract['profile'],0) if dev else None)
        entry=dict(epoch=e+1,loss=loss,validation=val,learning_rate=lr)
        history.append(entry)
        if not selected or val is None or val<history[selected-1]['validation']:
            selected=e+1
        _save(last,dict(contract=contract,epoch=e+1,stage=stage,model=model.state_dict(),
                        optimizer=optimizer.state_dict(),history=history,selected=selected))
        if selected==e+1:
            _save(best,dict(contract=contract,epoch=e+1,stage=stage,model=model.state_dict()))
        print(json.dumps(dict(stage=stage,**entry)),flush=True)
    del optimizer
    saved=torch.load(best,map_location='cpu',weights_only=True)
    if saved['contract']!=contract or saved['stage']!=stage or saved['epoch']!=selected:
        raise ValueError('Selected blood weights disagree with last committed epoch')
    model.load_state_dict(saved['model'])
    return history


def train(cache,args,out):
    import torch
    from .xu_blood_models import SliceUNet, BloodSequence
    from .xu_targets import TargetStore, blood_ids
    from .xu_training import _save
    store=TargetStore(args.targets,cache)
    supervision = getattr(args,'supervision','marginal')
    eligible = blood_cohort(cache,blood_ids(cache),supervision,store)
    if args.technical_unvalidated and not (args.profile=='smoke' and cache.scope=='technical_subset'):
        raise ValueError('Unvalidated capacity probe is restricted to technical smoke inputs')
    if not eligible['train'] or not (eligible['development'] or args.technical_unvalidated):
        raise ValueError('Require eligible blood training and development cases')
    contract = _contract(cache,store.identity,supervision,args.profile,args.seed,eligible)
    marker=out/'blood-contract.json'
    if marker.exists():
        stored = read_json(marker)
        if not args.resume or not _matches(stored,contract):
            raise ValueError('Blood training output is occupied or incompatible')
        # On certified v1 resume, existing per-epoch checkpoints retain their original identity.
        contract = stored
    else:
        if any(out.glob('*.pt')):
            raise ValueError('Unowned checkpoint in blood output')
        atomic_json(marker,contract)
    torch.manual_seed(args.seed)
    encoder=SliceUNet(args.profile).to(args.device)
    seq=BloodSequence(encoder.width).to(args.device)
    history={}
    for stage in ('encoder','sequence'):
        history[stage]=_train_stage(cache,store,contract,out,encoder,seq,stage,args.device)
    kind='best' if eligible['development'] else 'technical'
    path=out/(kind+'.pt')
    _save(path,dict(kind=kind,contract=contract,encoder=encoder.state_dict(),sequence=seq.state_dict()))
    result=dict(command='train-blood',model=path.name,supervision=supervision,
                selection='development' if kind=='best' else 'technical_last',
                train=len(eligible['train']),development=len(eligible['development']),history=history)
    atomic_json(out/'blood-training.json',result)
    return result


def load_blood_predictor(cache, checkpoint, device='cpu', profile=None, supervision=None):
    """Return a label-free predictor yielding 0/1/2 labels in requested cache-axis ROI."""
    import torch
    from .xu_blood_models import SliceUNet, BloodSequence
    from .xu_targets import blood_ids
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    c=state['contract']
    arm = supervision or c.get('supervision','marginal')
    eligible = blood_cohort(cache,blood_ids(cache),arm)
    expected = _contract(cache,c['targets'],arm,profile or c['profile'],c['seed'],eligible)
    if not _matches(c,expected):
        raise ValueError('Blood prediction checkpoint contract mismatch')
    if state['kind']!='best' and not (state['kind']=='technical' and c['profile']=='smoke'
                                      and cache.scope=='technical_subset'):
        raise ValueError('Expected selected weights or a technical smoke checkpoint')
    encoder=SliceUNet(c['profile'])
    seq=BloodSequence(encoder.width)
    encoder.load_state_dict(state['encoder'])
    seq.load_state_dict(state['sequence'])
    del state
    encoder.eval().to(device)
    seq.eval().to(device)

    @torch.inference_mode()
    def predict(name,box,shape):
        views=AxialViews(cache,name)
        zs, ys, xs=views.sampling(box,shape)
        result=np.empty((len(zs),len(ys),len(xs)),dtype=np.uint8)

        @lru_cache(maxsize=7)
        def features(z):
            frame=torch.from_numpy(views.image(z))[None,None].to(device)
            return encoder.features(frame)

        for z in np.unique(zs):
            scores=seq([features(i) for i in views.window(int(z))])[0]
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite blood predictions')
            label=scores.argmax(0).byte().cpu().numpy()
            result[zs==z]=label[np.ix_(ys,xs)]
        features.cache_clear()
        views.image.cache_clear()
        return np.ascontiguousarray(result.transpose(views.permutation))
    return predict


def run(cache,args,out):
    if args.command=='train-blood':
        return train(cache,args,out)
    if args.case not in cache.case_ids('development'):
        raise ValueError('Expected a development case')
    path=out/'blood-cache.npy'
    if path.exists():
        raise ValueError('Blood prediction requires fresh output')
    predict=load_blood_predictor(cache,args.checkpoint,args.device,args.profile)
    label=predict(args.case,None,cache.geometry(args.case).cache_shape)
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(dir=out,delete=False) as tmp:
        temp=Path(tmp.name)
        try:
            np.save(tmp,label,allow_pickle=False)
        except BaseException:
            tmp.close()
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)
    return dict(command='predict-blood',case=args.case,cache_shape=list(label.shape),path=path.name)
