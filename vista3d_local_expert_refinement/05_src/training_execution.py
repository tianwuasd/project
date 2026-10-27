"""Rank-local execution setup and capacity facts, separate from stage decisions."""
import math
import os
import platform
import time


class EagerRetryRequired(RuntimeError):
    """Abort the whole group; retry eager from the last complete checkpoint."""


def apply_execution(torch, options=None, *, main_threads=1):
    options = options or {}
    mode = options.get('mode', 'performance')
    if mode not in ('performance', 'strict'):
        raise ValueError('execution mode must be performance or strict')
    if int(main_threads) < 1:
        raise ValueError('compiler needs a positive rank CPU allocation')
    threads = int(main_threads)
    # Only this fresh rank process is changed; no persistent user configuration.
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = str(threads)
    os.environ['MAX_JOBS'] = '1'
    torch.backends.cudnn.benchmark = mode == 'performance'
    torch.backends.cudnn.deterministic = mode == 'strict'
    torch.use_deterministic_algorithms(mode == 'strict')
    import torch._inductor.config as config
    config.compile_threads = threads
    requested = bool(options.get('compile', True)) and mode != 'strict'
    return dict(mode=mode, compile_requested=requested, compile_applied=False,
        compile_backend='inductor', compile_mode='default', compile_threads=threads,
        nested_build_jobs=1, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        amp='upstream CUDA float16 autocast and GradScaler; CPU disabled',
        loss_mode='pinned upstream eager Dice+CE/BCE and eager deep-supervision weights',
        compute_timing_scope='host wall step and CUDA current-stream event elapsed time; not sum of kernel times',
        compile_warmup_seconds=0., fallback_reason=None,
        cache_identity=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            python=platform.python_version(), platform=platform.platform(),
            cache_dir=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
            triton_cache_dir=os.environ.get('TRITON_CACHE_DIR')))


def _leaves(value):
    if isinstance(value, (tuple, list)):
        for item in value: yield from _leaves(item)
    elif isinstance(value, dict):
        for item in value.values(): yield from _leaves(item)
    else:
        yield value


def preflight_compile(trainer, batch, report):
    """Local network-only forward/backward, NEVER DDP or distributed Dice.

    No optimizer update; restore buffers and RNG before formal training. All
    ranks agree on fallback before entering any formal DDP/loss collective.
    """
    import torch
    import torch.distributed as dist
    from checkpoint_rng import capture, restore
    if report.get('preflight_complete'): return
    report['preflight_complete'] = True
    if not report['compile_requested']: return
    eager = trainer.network.module if trainer.is_ddp else trainer.network
    rng = capture()
    # Copy to CPU so the recovery snapshot does not distort GPU capacity peaks.
    state = {key: value.detach().cpu().clone() for key, value in eager.state_dict().items()}
    report['compile_snapshot_cpu_bytes'] = sum(v.numel()*v.element_size() for v in state.values())
    old_grads = [p.grad for p in eager.parameters()]
    compiled, error = None, None
    started = time.perf_counter()
    try:
        if any(isinstance(module,torch.nn.SyncBatchNorm) for module in eager.modules()):
            raise RuntimeError('SyncBatchNorm forward can enter collectives; isolated compile preflight unavailable')
        # Setting env is insufficient if Inductor config was already imported.
        import torch._inductor.config as config
        with config.patch(compile_threads=report['compile_threads']):
            compiled = torch.compile(eager, backend='inductor', mode='default')
            data = batch['data'].to(trainer.device, non_blocking=True)
            from contextlib import nullcontext
            with torch.autocast('cuda') if trainer.device.type == 'cuda' else nullcontext():
                outputs = compiled(data)
                # Compile backward without invoking Dice all-gather or DDP hooks.
                scalar = sum(item.float().sum() for item in _leaves(outputs)
                             if isinstance(item, torch.Tensor) and item.requires_grad)
            torch.autograd.grad(scalar, tuple(eager.parameters()), allow_unused=True)
            if trainer.device.type == 'cuda': torch.cuda.synchronize(trainer.device)
    except Exception as failure:
        error = f'{type(failure).__name__}: {failure}'
    finally:
        eager.load_state_dict(state)
        for parameter, grad in zip(eager.parameters(), old_grads): parameter.grad = grad
        restore(rng)
        report['compile_warmup_seconds'] = time.perf_counter() - started
        if trainer.device.type == 'cuda':
            report['compile_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(trainer.device)
            report['compile_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(trainer.device)
    outcomes = [error]
    if dist.is_available() and dist.is_initialized():
        outcomes = [None] * dist.get_world_size()
        dist.all_gather_object(outcomes, error)
    errors = [f'rank {i}: {value}' for i, value in enumerate(outcomes) if value]
    if errors:
        report['fallback_reason'] = '; '.join(errors)
        trainer.print_to_log_file('Compile preflight failed; group eager fallback: ' + report['fallback_reason'])
        return
    if trainer.is_ddp: trainer.network.module = compiled
    else: trainer.network = compiled
    report['compile_applied'] = True
    report['cache_identity']['input_shape'] = list(batch['data'].shape)
    report['cache_identity']['input_dtype'] = str(batch['data'].dtype)
    report['cache_identity']['network'] = type(eager).__qualname__
    trainer.print_to_log_file(f'Network compile local preflight passed; eager loss unchanged; warmup={report["compile_warmup_seconds"]:.3f}s')


def capacity_batch(*, proven_batch, total_bytes, fixed_bytes, peak_allocated_bytes,
                   peak_reserved_bytes, optimizer_initialized, slope_bytes=None,
                   margin_fraction=.1):
    """One reservation margin, never infer an unmeasured activation slope.

    fixed_bytes is quiescent allocated storage after optimizer initialization. Reserved
    minus allocated is allocator slack, counted ONCE alongside fixed storage.
    A candidate still requires the coordinator's bounded capacity confirmation.
    """
    if proven_batch < 1 or not 0 <= margin_fraction < 1: raise ValueError('invalid capacity input')
    result = dict(candidate_batch=proven_batch, proven_batch=proven_batch,
                  status='retain_proven', reservation_margin_bytes=None)
    values = (total_bytes, fixed_bytes, peak_allocated_bytes, peak_reserved_bytes)
    if not optimizer_initialized:
        result['reason'] = 'optimizer state not initialized'; return result
    if (any(v is None or not math.isfinite(v) or v < 0 for v in values)
            or total_bytes <= 0 or not isinstance(slope_bytes, (int, float))
            or not math.isfinite(slope_bytes) or slope_bytes <= 0):
        result['reason'] = 'unknown or nonpositive capacity/slope'; return result
    slack = max(0, peak_reserved_bytes - peak_allocated_bytes)
    margin = total_bytes * margin_fraction
    candidate = math.floor((total_bytes - margin - fixed_bytes - slack) / slope_bytes)
    result.update(candidate_batch=max(proven_batch, candidate),
                  reservation_margin_bytes=margin, allocator_slack_bytes=slack,
                  status='capacity_candidate', reason='confirmation required; not speed ranking')
    return result


def memory_facts(trainer, *, quiescent=False):
    import torch
    seen = set()
    def size(value):
        total = 0
        for tensor in _leaves(value):
            if isinstance(tensor, torch.Tensor):
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                if key not in seen:
                    seen.add(key); total += storage.nbytes()
        return total
    parameters = list(trainer.network.parameters())
    fixed = size(parameters) + size(list(trainer.network.buffers()))
    fixed += size([p.grad for p in parameters]) + size(dict(trainer.optimizer.state))
    result = dict(fixed_bytes=fixed, unique_storage_bytes=fixed,
                  optimizer_initialized=bool(trainer.optimizer.state), quiescent=quiescent,
                  scope='unique model/buffer/gradient/optimizer storage lower bound; excludes DDP/compiler/context/allocator storage')
    if trainer.device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(trainer.device)
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(trainer.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(trainer.device),
            allocated_bytes=torch.cuda.memory_allocated(trainer.device),
            reserved_bytes=torch.cuda.memory_reserved(trainer.device), total_bytes=total,
            free_bytes=free)
        if quiescent:
            result['fixed_bytes'] = result['allocated_bytes']
            result['scope'] = ('quiescent CUDA allocated bytes after input release and zero_grad; '
                'includes Torch-tracked DDP/compiler tensors, excludes CUDA context and allocator slack')
        execution = getattr(trainer, '_vista_execution', {})
        result['train_validation_peak_allocated_bytes'] = result['peak_allocated_bytes']
        result['train_validation_peak_reserved_bytes'] = result['peak_reserved_bytes']
        for kind in ('allocated','reserved'):
            compile_peak = execution.get(f'compile_peak_{kind}_bytes',0)
            result[f'compile_peak_{kind}_bytes'] = compile_peak
            result[f'peak_{kind}_bytes'] = max(compile_peak,result[f'peak_{kind}_bytes'])
    else:
        result.update(peak_allocated_bytes=None, peak_reserved_bytes=None, total_bytes=None)
    return result
