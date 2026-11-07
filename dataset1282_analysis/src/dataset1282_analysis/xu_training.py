"""Train one Xu 3D branch on an explicitly frozen split and predict without labels."""
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch
from torch.nn import functional as F

from .contracts import atomic_json, digest, sha256_file
from .training.marginal import marginal_loss
from .xu_distributed import BranchForward
from .xu_models import BranchSpec, Xu3DSegmenter, branch_spec, build_model
from .xu_supply import BatchSupply, check_gradients


def _paper_loss(scores, target):
    valid = target != -1
    if not valid.any():
        raise ValueError('No supervised voxels')
    probabilities = scores.float().softmax(1)
    dice = []
    for c in range(11):
        p = probabilities[:, c] * valid
        y = target == c
        dice.append((2 * (p * y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6))
    weights = probabilities.new_tensor([1] + [2]*10)
    return 1 - (torch.stack(dice)*weights).sum()/weights.sum() + F.cross_entropy(
        scores.float(), target, ignore_index=-1)


def branch_loss(scores, target, unknown):
    """Full cases use paper Dice+CE; only partial cases use merged-background supervision."""
    complete = [i for i, classes in enumerate(unknown) if not classes]
    partial = [i for i, classes in enumerate(unknown) if classes]
    terms = []
    if complete:
        terms.append(len(complete) * _paper_loss(scores[complete], target[complete]))
    if partial:
        terms.append(len(partial) * marginal_loss(scores[partial], target[partial],
                                                  [unknown[i] for i in partial]))
    return sum(terms) / len(unknown)


def learning_rate(epoch, epochs):
    """Epoch index is zero based. Integer boundaries are explicit for odd run lengths."""
    fraction = epoch / epochs
    if fraction < 0.5:
        return 2e-4
    return 2e-5 if fraction < 0.75 else 2e-6


def _augment(images, labels, rng):
    """Joint 3D scaling, rotation, deformation; magnitudes follow the local CHD baseline."""
    batch = len(images)
    device = images.device
    angle = (torch.rand(batch, device=device, generator=rng) - 0.5) * (math.pi / 6)
    scale = 0.9 + 0.2 * torch.rand(batch, device=device, generator=rng)
    theta = torch.zeros(batch, 3, 4, dtype=images.dtype, device=device)
    theta[:,0,0] = theta[:,1,1] = scale * angle.cos()
    theta[:,0,1] = -scale * angle.sin()
    theta[:,1,0] = scale * angle.sin()
    theta[:,2,2] = scale
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    noise = torch.randn(batch,3,4,4,4, device=device, generator=rng) * 0.015
    grid += F.interpolate(noise, size=images.shape[2:], mode='trilinear',
                          align_corners=False).movedim(1,-1)
    images = F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=False)
    # Shift the labels so that zero padding maps back to the -1 ignore value.
    transported = F.grid_sample((labels[:,None]+1).float(), grid, mode='nearest', align_corners=False)
    return images, transported[:,0].long()-1


def _volume(array, side):
    volume = torch.as_tensor(array)
    if volume.shape == (1, side, side, side):
        volume = volume[0]
    if volume.shape != (side, side, side):
        raise ValueError(f'Expected a {side}^3 branch grid, got {tuple(volume.shape)}')
    return volume


def _image(array, side):
    image = _volume(array, side).float()
    if not torch.isfinite(image).all():
        raise ValueError('Non-finite input CT')
    return image


def _label(array, side, missing):
    label = _volume(array, side)
    if label.is_floating_point() and not torch.equal(label, label.round()):
        raise ValueError('Fractional or nonfinite label voxels')
    label = label.long()
    if label.min() < -1 or label.max() > 10 or not (label>=0).any():
        raise ValueError('Label outside -1..10 or empty supervision')
    if any((label == c).any() for c in missing):
        raise ValueError('Case metadata marks an observed foreground class unknown')
    return label


def _epoch(model, ids, groups, load, branch, device, rng, optimizer=None, stats=None):
    training = optimizer is not None
    model.train(training)
    batch_size = model.spec.batch
    forward = BranchForward(device, batch_size, require_all=True)
    loss_sum = 0.
    def prepare(batch):
        xs, ys = [], []
        for name in batch:
            x, y = load(name, branch)
            xs.append(_image(x, model.spec.side))
            ys.append(_label(y, model.spec.side, groups[name]))
        return torch.stack(xs)[:,None], torch.stack(ys)

    supply = BatchSupply(ids, batch_size, model.spec.side, prepare, getattr(load, 'source_voxels', None))
    with supply.batches() as batches:
        for step, (batch, pair) in enumerate(batches, 1):
            images, target = supply.transfer(pair, device)
            if training:
                images, target = _augment(images, target, rng)
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss = branch_loss(forward.forward(model, images), target, [groups[name] for name in batch])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite branch loss')
            if training:
                loss.backward()
                check_gradients(model)
                optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            if step%10 == 0:
                print(json.dumps(dict(branch=branch, phase='train' if training else 'validation',
                                      cases_completed=min(step*batch_size,len(ids)), total=len(ids))), flush=True)
            del pair
    if stats is not None:
        stats.update(supply.stats)
        stats['distribution'] = forward.stats
    return loss_sum/len(ids)


def _save(path, value):
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(value,f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _groups(groups):
    result = {}
    for name, unknown in groups.items():
        if not isinstance(name,str) or not name:
            raise ValueError('Expected a nonempty case ID')
        if any(type(c) is not int or c < 1 or c > 10 for c in unknown):
            raise ValueError('Expected integral foreground class IDs 1..10')
        result[name] = sorted(set(unknown))
    return result


def _write_best(best, state):
    _save(best,{k:state[k] for k in ('contract','epoch','model','best_loss')})


def _compatible_contract(actual, expected):
    if actual == expected:
        return True
    # The epoch pause is an execution boundary only; it leaves the previous
    # multi-device optimizer, schedule, loss and model definitions unchanged.
    previous_pause = dict(expected['source'])
    previous_pause['xu_training.py'] = '54b970cb34279c6e94392d4632aa90a3173c4f3bf0026660ac1fa6a1111cd247'
    if actual == dict(expected, source=previous_pause):
        return True
    # Exact prior single-device versions. Data, split, loss and model fingerprints
    # still need to match; new multi-device training adds its own replica-count field.
    previous = dict(expected['source'])
    if 'xu_supply.py' not in previous:
        return False
    if 'xu_distributed.py' in previous:
        del previous['xu_distributed.py']
        previous['xu_training.py'] = '91418394469d70b9f5f1e3b16b2bc1309c40303c1c92ee12068343f77382cffc'
        previous['xu_supply.py'] = 'b66bf7ac0598dd54b33ba0930f26ab21c76daa72b08915eb6e85be1016642841'
        if 'xu_init.py' in previous:
            previous['xu_init.py'] = '4127bfc94a5d5d31057f97bb44b84faf60e95d65949b8c43a981df9186359c4e'
        if actual == dict(expected, source=previous):
            return True
    del previous['xu_supply.py']
    previous['xu_training.py'] = '2ed34c4688840f93c08992c9c623f4b88fb2b674ec0a259e12e30b3f076a38ed'
    if 'xu_init.py' in previous:
        previous['xu_init.py'] = '715f576c65fe60d5472ce08cdbbb468c35158e6de94d722c4db429b84a9d61d6'
    return actual == dict(expected, source=previous)


def _recover_best(best, state):
    if state['epoch'] == state['best_epoch']:
        # last.pt commits first so a crash cannot promote an uncommitted best.
        _write_best(best,state)
        return
    actual = torch.load(best,map_location='cpu',weights_only=True)
    if (actual['epoch'] != state['best_epoch'] or actual['best_loss'] != state['best_loss']
            or not _compatible_contract(actual['contract'], state['contract'])):
        raise ValueError('Best checkpoint differs from committed epoch history')


def _training_model(branch, profile, technical_fit):
    if not technical_fit or technical_fit.get('width', 2) == 2:
        return build_model(branch, profile)
    width = technical_fit['width']
    if profile != 'smoke' or width not in (4, 8):
        raise ValueError('Invalid technical pilot width')
    base = branch_spec(branch, profile)
    return Xu3DSegmenter(BranchSpec(base.side, width, 4, 2*width, 1, technical_fit['epochs']))


def execution_contract(replicas, batch, elastic):
    if not 1 <= replicas <= batch:
        raise ValueError('设备数量必须适合固定 batch')
    if elastic:
        return dict(execution_policy='fixed-batch-reallocation-v1', global_batch=batch)
    return {'batch_replicas': replicas} if replicas > 1 else {}


def train_branch(*, branch, output_dir, train_cases, validation_cases, load_sample,
                 data_identity, profile='paper', supervision='complete', device='cpu',
                 resume=False, seed=2023, pilot_epochs=None, pilot_width=2, pause_after_epoch=None):
    """load_sample(case_id, branch) returns aligned image/label cubes on the branch grid.

    Callers bind preprocessing, cohort, geometry, and predicted ROI identities in
    data_identity and hold an exclusive output-directory lock for the run.
    """
    spec = branch_spec(branch,profile)
    if pause_after_epoch is not None and (pause_after_epoch not in (1, 2) or profile != 'paper'
                                          or branch not in ('crop64', 'crop128') or pilot_epochs is not None):
        raise ValueError('Epoch pause supports only paper locators after epoch 1 or 2')
    if pilot_epochs is not None and (profile != 'smoke' or not 1 <= pilot_epochs <= 500):
        raise ValueError('Pilot fit requires 1..500 technical-profile epochs')
    if pilot_width not in (2, 4, 8) or (pilot_width != 2 and pilot_epochs is None):
        raise ValueError('Pilot width needs an explicitly bounded technical fit')
    epochs = pilot_epochs or spec.epochs
    train, val = _groups(train_cases), _groups(validation_cases)
    if not train or not val or train.keys() & val.keys():
        raise ValueError('Need nonempty disjoint training and validation IDs')
    if not isinstance(data_identity,str) or not data_identity.strip():
        raise ValueError('Missing data identity')
    if supervision not in ('complete','marginal'):
        raise ValueError('Expected complete or marginal supervision')
    if supervision=='complete' and (any(train.values()) or (profile=='paper' and any(val.values()))):
        raise ValueError('Complete paper arm requires cases with all foreground labels')
    if device != 'cpu' and torch.device(device).type != 'cuda':
        raise ValueError('Expected cpu or cuda device')
    if profile=='paper' and len(train)<spec.batch:
        raise ValueError('Insufficient cases for paper training batch')
    root = Path(output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    last, best = root/'last.pt', root/'best.pt'
    if resume != last.is_file() or (not resume and best.exists()):
        raise ValueError('Use a fresh output or resume a committed matching run')
    source = Path(__file__).parent
    contract = dict(protocol='xu-3d-v1', branch=branch, profile=profile,
                    data_identity=data_identity, supervision=supervision, seed=seed,
                    split_digest=digest(dict(train=train,validation=val)), torch_version=str(torch.__version__),
                    source={name:sha256_file(source/name) for name in
                            ('xu_models.py','xu_training.py','xu_supply.py','xu_distributed.py','training/marginal.py')})
    replicas = BranchForward(device, spec.batch, require_all=True).max_active
    contract.update(execution_contract(replicas, spec.batch, os.environ.get('XU_ELASTIC_DEVICES') == '1'))
    if pilot_epochs is not None:
        contract['technical_fit'] = dict(epochs=epochs, initial_lr=0.002, width=pilot_width)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = _training_model(branch,profile,contract.get('technical_fit')).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate(0,spec.epochs))
    epoch_start, best_epoch, best_loss, history = 0, -1, float('inf'), []
    if resume:
        state = torch.load(last,map_location='cpu',weights_only=True)
        if not _compatible_contract(state['contract'], contract):
            raise ValueError('Checkpoint contract changed')
        _recover_best(best,state)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_start = state['epoch']+1
        best_epoch,best_loss,history = state['best_epoch'],state['best_loss'],state['history']
        print(f'Restored {branch} after epoch {epoch_start}: {last}', flush=True)
        del state
    # Keep the planned 15-epoch schedule and contract when stopping after a
    # complete train/development epoch. Resume reconstructs the next epoch RNG.
    stop_epoch = min(epochs, pause_after_epoch or epochs)
    for epoch in range(epoch_start,stop_epoch):
        ids=sorted(train)
        random.Random(seed+epoch).shuffle(ids)
        rng=torch.Generator(device=device).manual_seed(seed+epoch)
        lr=learning_rate(epoch,epochs) * (10 if pilot_epochs else 1)
        for params in optimizer.param_groups:
            params['lr']=lr
        train_supply, validation_supply = {}, {}
        train_loss=_epoch(model,ids,train,load_sample,branch,device,rng,optimizer,train_supply)
        validation_loss=_epoch(model,sorted(val),val,load_sample,branch,device,rng,stats=validation_supply)
        if validation_loss < best_loss:
            best_loss,best_epoch=validation_loss,epoch
        entry=dict(epoch=epoch+1,learning_rate=lr,train_loss=train_loss,validation_loss=validation_loss,
                   train_supply=train_supply, validation_supply=validation_supply)
        if hasattr(load_sample,'cache_stats'):
            entry['input_cache_cumulative'] = dict(load_sample.cache_stats)
        history.append(entry)
        state=dict(contract=contract,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   epoch=epoch,best_epoch=best_epoch,best_loss=best_loss,history=history)
        _save(last,state)
        if best_epoch==epoch:
            _write_best(best,state)
        print(json.dumps(dict(branch=branch,profile=profile,**entry)),flush=True)
    completed_epochs = len(history)
    summary=dict(branch=branch,profile=profile,supervision=supervision,train_cases=len(train),
                 state='complete' if completed_epochs == epochs else 'paused', completed_epochs=completed_epochs,
                 validation_cases=len(val),batch=spec.batch,epochs=epochs,
                 validation_partial_cases=sum(bool(v) for v in val.values()),best_epoch=best_epoch+1,
                 best_loss=best_loss,history=history,best_checkpoint=str(best),last_checkpoint=str(last),
                 contract_digest=digest(contract))
    atomic_json(root/'training.json',summary)
    return summary


def load_predictor(checkpoint, *, branch, profile, data_identity, device='cpu', supervision=None):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    c=state['contract']
    if ((c['branch'],c['profile'],c['data_identity']) != (branch,profile,data_identity)
            or c.get('protocol')!='xu-3d-v1'
            or (supervision is not None and c['supervision']!=supervision)):
        raise ValueError('Prediction checkpoint contract mismatch')
    # A last.pt is equivalent to best.pt only when validation selected its epoch.
    if 'optimizer' in state and state.get('best_epoch') != state['epoch']:
        raise ValueError('Inference requires the validation-selected best checkpoint')
    if c['source']['xu_models.py']!=sha256_file(Path(__file__).with_name('xu_models.py')):
        raise ValueError('Prediction model implementation changed')
    model=_training_model(branch,profile,c.get('technical_fit'))
    model.load_state_dict(state['model'])
    model.eval().to(device)
    del state

    @torch.inference_mode()
    def predict(image):
        x=_image(image,model.spec.side)[None,None].to(device)
        scores=model(x)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Non-finite predicted scores')
        return scores.argmax(1)[0].to('cpu',torch.uint8).numpy()
    return predict
