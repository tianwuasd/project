"""Train the separate 2D and temporal blood-pool networks on prepared axial slices."""
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from .contracts import atomic_json, read_json, sha256_file
from .xu_blood_grid import AxialViews
from .xu_blood_supply import SliceSupply, PhaseTiming, prepare_slices, augment_slices


# Reviewed at 13ceecc2. Unscoped v1 used every case with a known blood target.
_V1_SOURCE = {
    'xu_blood.py': 'e70d0f19b0c292b4417f13a8c6d6b97160396df99fb5380b2a1861b508eb8abc',
    'xu_blood_grid.py': '3d3f354441941584e14c54160bf89722f250d60a82a8f03fb40acd8f3cde8a99',
    'xu_blood_models.py': '3e8b2508bbc470e9355eaf89490e3ffb5213d99a5e9eaaddc42d2f4f62357f8f',
    'xu_geometry.py': '4500549b63cd6dc38551a177193137c3b2da64faa582d5d182ec84be9c805d79',
}
# d0f2c45a differs here only in retaining unused encoder gradients.
_PREVIOUS_V2_SOURCE = dict(_V1_SOURCE, **{
    'xu_blood.py':'a3894bdb66686531ac768d8526a0e5d144394210c8847bec647074c300f80bb4'})
_SINGLE_GPU_V2_SOURCE = {**_PREVIOUS_V2_SOURCE,
                         'xu_blood.py':'8879aa65a0c59ec0dd698241e783642651adcedc9df0342b1e3b388a64543659'}


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
            ('xu_blood.py','xu_blood_grid.py','xu_blood_models.py','xu_geometry.py','xu_distributed.py',
             'xu_blood_supply.py','xu_supply.py','xu_supply_budget.py','xu_resume_compat.py')}


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


def _normalized_contract(stored, expected):
    if (stored.get('method') == 'xu-blood-v1' and 'supervision' not in stored
            and expected['supervision'] == 'marginal'):
        return dict(stored,method='xu-blood-v2',supervision='marginal')
    return stored


def _matches(stored, expected):
    from .xu_resume_compat import source_compatible
    actual = _normalized_contract(stored,expected)
    return (isinstance(actual.get('source'),dict) and bool(actual['source']) and
            {k:v for k,v in actual.items() if k != 'source'} ==
            {k:v for k,v in expected.items() if k != 'source'} and
            source_compatible(actual['source'],expected['source'],'src/dataset1282_analysis/'))


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
    images, labels = prepare_slices(views, centres, temporal)
    return augment_slices(images.to(device),labels.to(device),augment)


class BloodForward:
    """Shard independent slices or frozen feature frames; keep each temporal sequence on its primary."""
    def __init__(self, encoder, device, temporal, batch_size):
        import torch
        from .xu_distributed import BranchForward
        self.encoder = encoder
        self.temporal = temporal
        self.distribution = BranchForward(device,7 if temporal else batch_size)
        if temporal:
            class Features(torch.nn.Module):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def forward(self, frames):
                    return self.network.features(frames)

            self.features = Features(encoder).eval().requires_grad_(False)
            if self.distribution.max_active > 1:
                # Keep the frozen multi-gigabyte slice encoder resident for this
                # pass over the cohort, rather than broadcasting it per centre.
                self.distribution.freeze(self.features)
        else:
            self.features = None

    def __call__(self, images, sequence):
        import torch
        if not self.temporal:
            return self.distribution.forward(self.encoder,images)
        with torch.no_grad():
            if len(self.distribution.stats['assigned_devices']) == 1:
                frames = [self.distribution.forward(self.features,images[:,i:i+1]) for i in range(7)]
            else:
                if images.shape[:2] != (1,7):
                    raise ValueError('Temporal blood input must retain one ordered seven-frame case')
                frames = self.distribution.forward(self.features, images.transpose(0,1).contiguous()).split(1)
        return sequence(frames)


def _epoch(cache, store, names, encoder, sequence, device, stage, profile, seed, optimizer=None,
           epoch=None, total_epochs=None, telemetry=None):
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
    forward = BloodForward(encoder,device,temporal,batch)
    timing = PhaseTiming()
    loss_sum, count = 0., 0
    for pos, name in enumerate(order,1):
        views = AxialViews(cache,name,store)
        centres = np.arange(views.zyx[0])
        if profile=='smoke':
            centres = np.array([len(centres)//2])
        if training:
            rng.shuffle(centres)
        supply = SliceSupply(views,centres,batch,temporal)
        step_seconds = 0.
        with supply.batches() as batches:
            for _, pair in batches:
                images, labels = supply.transfer(pair,device)
                # 增强随机数在消费线程产生，保持原有病例与切片顺序。
                images, labels = augment_slices(images,labels,rng if training else None)
                if not (labels >= 0).any():
                    continue
                started = time.perf_counter()
                with torch.set_grad_enabled(training):
                    scores = forward(images,sequence)
                    loss = _objective(scores,labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite blood-pool loss')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                loss_sum += float(loss.detach()) * len(images)
                count += len(images)
                step_seconds += time.perf_counter()-started
        finished = time.perf_counter()
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        timing.add(supply,step_seconds,time.perf_counter()-finished)
        views.image.cache_clear()
        print(json.dumps(dict(stage=stage,phase='train' if training else 'development',
                              epoch=epoch,total_epochs=total_epochs,updated_at=time.time(),
                              cases_completed=pos,total=len(order),slices=count,
                              timing=timing.report(count,pos,len(order)))),flush=True)
    if not count:
        raise ValueError('No supervised blood-pool slices')
    if len(forward.distribution.stats['assigned_devices']) > 1:
        print(json.dumps(dict(stage=stage,distribution=forward.distribution.stats)),flush=True)
    if telemetry is not None:
        telemetry.update(timing.report(count,len(order),len(order)))
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
            complete_best = False
            if best.exists():
                previous = torch.load(best,map_location='cpu',weights_only=True)
                if previous['contract'] != contract or previous['stage'] != stage or previous['epoch'] > start:
                    raise ValueError('Selected blood checkpoint contract changed')
                complete_best = previous['epoch'] == start
                if complete_best and (previous['model'].keys() != state['model'].keys() or
                        any(not torch.equal(previous['model'][k],v) for k,v in state['model'].items())):
                    raise ValueError('Selected blood weights disagree with last committed epoch')
                del previous
            if not complete_best:
                _save(best,dict(contract=contract,epoch=start,stage=stage,model=state['model']))
        del state
    for e in range(start,epochs):
        epoch_started = time.perf_counter()
        lr=learning_rate(e,epochs)
        for group in optimizer.param_groups:
            group['lr']=lr
        train_timing, validation_timing = {}, {}
        loss=_epoch(cache,store,contract['train'],encoder,sequence,device,stage,contract['profile'],
                    contract['seed']+e,optimizer,e+1,epochs,train_timing)
        dev=contract['development']
        val=(_epoch(cache,store,dev,encoder,sequence,device,stage,contract['profile'],0,
                    epoch=e+1,total_epochs=epochs,telemetry=validation_timing) if dev else None)
        entry=dict(epoch=e+1,loss=loss,validation=val,learning_rate=lr)
        history.append(entry)
        if not selected or val is None or val<history[selected-1]['validation']:
            selected=e+1
        _save(last,dict(contract=contract,epoch=e+1,stage=stage,model=model.state_dict(),
                        optimizer=optimizer.state_dict(),history=history,selected=selected))
        if selected==e+1:
            _save(best,dict(contract=contract,epoch=e+1,stage=stage,model=model.state_dict()))
        elapsed = time.perf_counter()-epoch_started
        print(json.dumps(dict(stage=stage,**entry,total_epochs=epochs,remaining_epochs=epochs-e-1,
                              epoch_seconds=elapsed,updated_at=time.time(),
                              phase_eta_seconds=elapsed*(epochs-e-1),
                              train_timing=train_timing,validation_timing=validation_timing)),flush=True)
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
        if stored != contract:
            from .xu_resume_compat import preserve_contract
            normalized = _normalized_contract(stored,contract)
            preserve_contract(normalized,contract,out,field='source',path_prefix='src/dataset1282_analysis/')
            if normalized != stored:
                legacy_record = out/'.runtime/resume-compat/original-blood-contract.json'
                if legacy_record.exists() and read_json(legacy_record) != stored:
                    raise ValueError('Original blood contract evidence changed')
                if not legacy_record.exists():
                    atomic_json(legacy_record,stored)
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
        if stage == 'sequence':
            # This encoder is frozen while the temporal network trains.
            encoder.zero_grad(set_to_none=True)
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
