import gc
import os
from time import perf_counter

import numpy as np
import torch

from .contracts import sha256_file
from .training.marginal import marginal_loss
from .xu_models import build_model
from .xu_training import _save


def check_models(cache, name, workspace, device, sam_weights, stunet_weights, event, models):
    from .xu_resources import apply_env_limit, combined_environment, stage_resources
    plan = stage_resources([device], threads=4, policy='v2', require_pid_evidence=device!='cpu',
                           host_demand_bytes=4*1024**3, cuda_demand_bytes=3*1024**3 if device!='cpu' else 0)
    os.environ.update(combined_environment(plan, [device]))
    limits = apply_env_limit(device)
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    event(stage='model-admission', limits=limits, device=str(device))

    def finish(model, optimizer, loss, forward, label, started):
        if not torch.isfinite(loss):
            raise ValueError('短测损失非有限值')
        parameter = next(p for p in model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12., error_if_nonfinite=True)
        optimizer.step()
        if torch.equal(before, parameter):
            raise ValueError('短测没有更新模型参数')
        model.eval()
        with torch.no_grad():
            expected = forward().detach().cpu()
        path = workspace/(label+'-last.pt')
        _save(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=1))
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            torch.testing.assert_close(forward().cpu(), expected, rtol=0, atol=0)
        event(stage='model-update-reload', model=label, updates=1, loss=float(loss.detach()),
              seconds=perf_counter()-started, checkpoint_bytes=path.stat().st_size,
              checkpoint_sha256=sha256_file(path), prediction_exact=True,
              peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        del state, expected
        path.unlink()

    for branch in ('crop64', 'crop128', 'all64', 'all128') if 'xu' in models else ():
        started = perf_counter()
        size = 128 if branch.endswith('128') else 64
        x = torch.from_numpy(cache.read_image(name, (size,)*3))[None,None].to(device)
        y = torch.from_numpy(cache.read_label(name, (size,)*3).astype(np.int64))[None].to(device)
        model = build_model(branch, 'smoke').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        loss = marginal_loss(model(x), y, [cache.supervision[name]['unknown_classes']])
        finish(model, optimizer, loss, lambda:model(x), 'xu-'+branch+'-smoke', started)
        del model, optimizer, loss, x, y
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if 'stunet' in models:
        from .independent import stunet
        started = perf_counter()
        patch = (32,)*3
        model = stunet.network(stunet.topology((1.,)*3, patch), False)
        stunet.load_pretrained(model, stunet_weights)
        model.to(device)
        reader = stunet.PatchReader(cache, patch)
        image, target = reader.read(name, np.random.default_rng(2023), True)
        x = torch.from_numpy(image)[None,None].to(device)
        y = torch.from_numpy(target.astype(np.int64))[None].to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.99)
        loss = marginal_loss(model(x), y, [cache.supervision[name]['unknown_classes']])
        finish(model, optimizer, loss, lambda:model(x), 'stunet-small', started)
        del model, optimizer, loss, x, y, reader, image, target
        cache.clear()
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if 'sam' not in models:
        return

    from .independent.sam_med3d import load_model
    from .independent.sam_train import PromptHead, forward_patch, update_model, fingerprint
    started = perf_counter()
    model = load_model(sam_weights, device)
    model.image_encoder.requires_grad_(False)
    head = PromptHead(model).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    case = (name, cache.supervision[name]['present_classes'][0], 2023)
    before = fingerprint(head)
    value = update_model(model.image_encoder, head, cache, case, device, optimizer,
                         dict(dice_smoothing=1., gradient_clip=12.))
    if before == fingerprint(head):
        raise ValueError('SAM 提示解码器没有更新')
    head.eval()
    with torch.no_grad():
        expected, target = forward_patch(model.image_encoder, head, cache, *case, device)
        expected = expected.cpu()
    path = workspace/'sam-last.pt'
    _save(path, dict(model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(), steps=1))
    saved = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    with torch.no_grad():
        actual, _ = forward_patch(model.image_encoder, head, cache, *case, device)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    event(stage='model-update-reload', model='sam-med3d-turbo', updates=1, loss=value,
          seconds=perf_counter()-started, checkpoint_bytes=path.stat().st_size,
          checkpoint_sha256=sha256_file(path), prediction_exact=True,
          peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    path.unlink()
