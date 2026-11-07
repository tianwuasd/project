"""Seven-class proximal aorta/pulmonary artery auxiliary branch."""
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import atomic_json, digest, sha256_file
from .xu_distributed import BranchForward
from .xu_models import Xu3DSegmenter, branch_spec, _conv
from .xu_targets import TargetStore
from .xu_supply import BatchSupply, check_gradients
from .xu_training import _augment, _compatible_contract, _image, _recover_best, _save, _write_best, learning_rate

PROTOCOL = 'xu-reviewed-proximal-v1'


def _model(branch, profile):
    if branch not in ('init64','init128'):
        raise ValueError('Expected init64 or init128')
    spec = branch_spec(branch.replace('init','all'),profile)
    model = Xu3DSegmenter(spec)
    model.head = nn.Conv3d(spec.width,7,1)
    model.prior = nn.Sequential(nn.MaxPool3d(4),*_conv(7,spec.prior_width),
                                *_conv(spec.prior_width,spec.prior_width),*_conv(spec.prior_width,7),
                                nn.ConvTranspose3d(7,7,4,stride=4))
    return model


def _loss(scores, target):
    valid = target != -1
    if not valid.any():
        raise ValueError('No supervised voxels inside the predicted ROI')
    p = scores.float().softmax(1)
    dice = []
    for c in range(7):
        truth = target==c
        pc = p[:,c] * valid
        dice.append((2*(pc*truth).sum()+1e-6)/(pc.sum()+truth.sum()+1e-6))
    weights = p.new_tensor([1,2,2,2,2,2,3])
    return F.cross_entropy(scores.float(),target,ignore_index=-1) + 1 - (torch.stack(dice)*weights).sum()/weights.sum()


def _epoch(model, names, cache, targets, book, device, rng=None, optimizer=None, stats=None):
    training = optimizer is not None
    model.train(training)
    spec = model.spec
    forward = BranchForward(device, spec.batch, require_all=True)
    loss_sum = 0.
    def prepare(batch):
        images, labels = [], []
        for name in batch:
            box = book['boxes'][name]
            shape = (spec.side,)*3
            array = cache.read_image(name,shape,box)
            y = targets.read(name,'init',box,shape)
            if y.dtype.kind not in 'iu' or y.min() < -1 or y.max()>6 or not (y==6).any():
                raise ValueError('Invalid auxiliary labels or proximal supervision lost in predicted ROI')
            images.append(_image(array,spec.side))
            labels.append(torch.from_numpy(y.astype('int64')))
        return torch.stack(images)[:,None], torch.stack(labels)

    supply = BatchSupply(names, spec.batch, spec.side, prepare,
                         lambda name: math.prod(cache.geometry(name).cache_shape))
    with supply.batches() as batches:
        for step, (batch, pair) in enumerate(batches, 1):
            x,y = supply.transfer(pair, device)
            if training:
                x,y = _augment(x,y,rng)
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss = _loss(forward.forward(model, x),y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite proximal loss')
            if training:
                loss.backward()
                check_gradients(model)
                optimizer.step()
            loss_sum += float(loss.detach())*len(batch)
            if step%10==0:
                print(f'Init {"training" if training else "validation"} {min(step*spec.batch,len(names))}/{len(names)}',flush=True)
            del pair
    if stats is not None:
        stats.update(supply.stats)
        stats['distribution'] = forward.stats
    return loss_sum/len(names)


def _inference_contract(cache, branch, profile, supervision, book):
    from .xu_workflow import input_id
    source = Path(__file__).parent
    return dict(protocol=PROTOCOL,branch=branch,profile=profile,supervision=supervision,
                identity=input_id(cache,supervision,book),classes=['BG','LV','RV','LA','RA','MYO','proximal_AO_PA'],
                source={s:sha256_file(source/s) for s in ('xu_init.py','xu_models.py','xu_targets.py','xu_training.py','xu_supply.py','xu_distributed.py')})


def train_init(cache,args,out):
    from .xu_sample_cache import CachedPrepared
    reader = CachedPrepared(cache,out/'.input-cache')
    from .xu_workflow import roi_book
    book = roi_book(args.boxes,cache,args)
    targets = TargetStore(args.targets,cache)
    def eligible(role):
        names = cache.case_ids(role,complete_only=(args.supervision=='complete'))
        return [name for name in names if targets.has(name,'init')]
    train = eligible('train')
    val = eligible('development')
    if not train or not val or set(train)&set(val):
        raise ValueError('Reviewed proximal targets are required for separate training and development patients')
    contract = _inference_contract(cache,args.branch,args.profile,args.supervision,book)
    contract.update(target_store=targets.identity,seed=args.seed,train_val=digest([train,val]),
                    torch_version=str(torch.__version__))
    device = torch.device(args.device)
    if device.type not in ('cuda','cpu'):
        raise ValueError('Expected CPU or CUDA')
    last,best = out/'last.pt',out/'best.pt'
    if args.resume != last.is_file() or (not args.resume and best.exists()):
        raise ValueError('Use a fresh directory or --resume a committed run')
    spec = branch_spec(args.branch.replace('init','all'),args.profile)
    replicas = BranchForward(device, spec.batch, require_all=True).max_active
    if replicas > 1:
        contract['batch_replicas'] = replicas
    if args.profile=='paper' and len(train)<spec.batch:
        raise ValueError('Insufficient eligible patients for the paper batch')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        model=_model(args.branch,args.profile).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
    history,best_epoch,best_loss=[], -1, math.inf
    if args.resume:
        state=torch.load(last,map_location='cpu',weights_only=True)
        if not _compatible_contract(state['contract'],contract):
            raise ValueError('Proximal training contract changed')
        _recover_best(best,state)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history,best_epoch,best_loss=state['history'],state['best_epoch'],state['best_loss']
        del state
    for epoch in range(len(history),spec.epochs):
        names=sorted(train)
        random.Random(args.seed+epoch).shuffle(names)
        rng=torch.Generator(device=device).manual_seed(args.seed+epoch)
        lr=learning_rate(epoch,spec.epochs)
        for g in optimizer.param_groups:
            g['lr']=lr
        train_supply, validation_supply = {}, {}
        tr=_epoch(model,names,reader,targets,book,device,rng,optimizer,train_supply)
        va=_epoch(model,val,reader,targets,book,device,stats=validation_supply)
        if va<best_loss:
            best_epoch,best_loss=epoch,va
        row=dict(epoch=epoch+1,train_loss=tr,validation_loss=va,learning_rate=lr,
                 train_supply=train_supply, validation_supply=validation_supply)
        row['input_cache_cumulative'] = dict(reader.arrays.stats)
        print(json.dumps(row),flush=True)
        history.append(row)
        state=dict(contract=contract,epoch=epoch,history=history,best_epoch=best_epoch,best_loss=best_loss,
                   model=model.state_dict(),optimizer=optimizer.state_dict())
        _save(last,state)
        if epoch==best_epoch:
            _write_best(best,state)
    report=dict(branch=args.branch,profile=args.profile,supervision=args.supervision,train_cases=len(train),
                development_cases=len(val),best_epoch=best_epoch+1,best_loss=best_loss,
                contract_digest=digest(contract),history=history)
    atomic_json(out/'training.json',report)
    return report


def load_init_predictor(checkpoint, *, cache, branch, profile, supervision, book, device='cpu'):
    fingerprint=sha256_file(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if sha256_file(checkpoint)!=fingerprint:
        raise ValueError('Proximal checkpoint changed while loading')
    expected=_inference_contract(cache,branch,profile,supervision,book)
    if not _compatible_contract({k:state['contract'].get(k) for k in expected},expected):
        raise ValueError('Incompatible proximal checkpoint')
    if 'optimizer' in state and state['best_epoch']!=state['epoch']:
        raise ValueError('Inference requires a validation-selected checkpoint')
    model=_model(branch,profile)
    model.load_state_dict(state['model'])
    model.eval().to(device)
    del state

    @torch.inference_mode()
    def predict(image):
        scores=model(_image(image,model.spec.side)[None,None].to(device))
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite proximal prediction')
        return scores.argmax(1)[0].to('cpu',torch.uint8).numpy()
    predict.checkpoint_sha256=fingerprint
    return predict
