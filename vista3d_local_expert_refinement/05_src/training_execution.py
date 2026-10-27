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


def _probe_network(network, data, parameters, device_type):
    """One local autograd graph at a time; only detached CPU references survive."""
    import torch
    from contextlib import nullcontext
    with torch.autocast('cuda') if device_type == 'cuda' else nullcontext():
        outputs = network(data.clone())
        scalar = sum(item.float().sum() for item in _leaves(outputs)
                     if isinstance(item, torch.Tensor) and item.requires_grad)
    tensors = []
    def snapshot(value):
        if isinstance(value, torch.Tensor):
            cpu = value.detach().to(device='cpu', copy=True)
            if not torch.isfinite(cpu).all():
                raise ValueError('compile compatibility: nonfinite output')
            tensors.append(cpu)
            return ('tensor', tuple(cpu.shape), str(cpu.dtype))
        if type(value) in (list, tuple):
            return (type(value).__name__, tuple(snapshot(item) for item in value))
        if type(value) is dict:
            return ('dict', tuple((key,snapshot(item)) for key,item in value.items()))
        if value is None: return ('none',)
        raise ValueError('compile compatibility: unsupported output structure')
    structure = snapshot(outputs)
    gradients = torch.autograd.grad(scalar, parameters, allow_unused=True)
    cpu_gradients = []
    for gradient in gradients:
        cpu = gradient.detach().to(device='cpu', copy=True) if gradient is not None else None
        if cpu is not None and not torch.isfinite(cpu).all():
            raise ValueError('compile compatibility: nonfinite gradient')
        cpu_gradients.append(cpu)
    return structure, tensors, cpu_gradients


def _compare_probes(reference, actual, *, rtol, atol):
    import torch
    if reference[0] != actual[0]:
        raise ValueError('compile compatibility: output structure/shape/dtype mismatch')
    for name, expected, observed in (('output',reference[1],actual[1]),
                                      ('gradient',reference[2],actual[2])):
        if len(expected) != len(observed):
            raise ValueError(f'compile compatibility: {name} count mismatch')
        for index, (left,right) in enumerate(zip(expected,observed)):
            if left is None or right is None:
                if left is not right:
                    raise ValueError(f'compile compatibility: {name}[{index}] None mismatch')
            elif (left.shape != right.shape or left.dtype != right.dtype
                  or not torch.allclose(left,right,rtol=rtol,atol=atol,equal_nan=False)):
                raise ValueError(f'compile compatibility: {name}[{index}] differs from eager (rtol={rtol}, atol={atol})')


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
    # Float16 AMP allows 1% relative / 1e-3 absolute differences; full precision
    # uses 1e-4 / 1e-5. These are explicit acceptance bounds, not bitwise claims.
    report['compatibility'] = dict(passed=False, local_passed=False,
        rtol=1e-2 if trainer.device.type == 'cuda' else 1e-4,
        atol=1e-3 if trainer.device.type == 'cuda' else 1e-5,
        reference_storage='detached CPU outputs and parameter gradients',
        scope='same-input local network outputs and sum-output gradients; eager loss unchanged')
    compiled, error = None, None
    started = time.perf_counter()
    try:
        if any(isinstance(module,torch.nn.SyncBatchNorm) for module in eager.modules()):
            raise RuntimeError('SyncBatchNorm forward can enter collectives; isolated compile preflight unavailable')
        # Setting env is insufficient if Inductor config was already imported.
        import torch._inductor.config as config
        with config.patch(compile_threads=report['compile_threads']):
            data = batch['data'].to(trainer.device, non_blocking=True)
            parameters = tuple(p for p in eager.parameters() if p.requires_grad)
            reference = _probe_network(eager,data,parameters,trainer.device.type)
            eager.load_state_dict(state); restore(rng)
            compiled = torch.compile(eager, backend='inductor', mode='default')
            # Constructor work may consume RNG; both executions start identically.
            eager.load_state_dict(state); restore(rng)
            actual = _probe_network(compiled,data,parameters,trainer.device.type)
            _compare_probes(reference,actual,rtol=report['compatibility']['rtol'],
                            atol=report['compatibility']['atol'])
            report['compatibility']['local_passed'] = True
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
    report['compatibility']['passed'] = True
    report['cache_identity']['input_shape'] = list(batch['data'].shape)
    report['cache_identity']['input_dtype'] = str(batch['data'].dtype)
    report['cache_identity']['network'] = type(eager).__qualname__
    trainer.print_to_log_file(f'Network compile local preflight passed; eager loss unchanged; warmup={report["compile_warmup_seconds"]:.3f}s')


def capacity_batch(*, proven_batch, usable_bytes, fixed_bytes, peak_allocated_bytes,
                   peak_reserved_bytes, optimizer_initialized, slope_bytes=None,
                   evidence_reliable=True):
    """U is already usable capacity; do not reserve percentage/cache space again.

    Derive (peak-fixed)/proven only from reliable initialized measurements.
    Reserved memory is observational. Any proposal still needs confirmation.
    """
    if isinstance(proven_batch,bool) or not isinstance(proven_batch,int) or proven_batch < 1:
        raise ValueError('invalid proven batch')
    result = dict(candidate_batch=proven_batch, proven_batch=proven_batch,
                  status='retain_proven', usable_bytes=usable_bytes,
                  peak_reserved_bytes=peak_reserved_bytes, slope_bytes=None)
    if not optimizer_initialized or not evidence_reliable:
        result['reason'] = 'optimizer uninitialized or evidence unreliable'; return result
    def valid(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0
    if not all(valid(v) for v in (usable_bytes,fixed_bytes,peak_allocated_bytes)) or usable_bytes<=0:
        result['reason'] = 'unknown or invalid capacity evidence'; return result
    if slope_bytes is None:
        slope_bytes = (peak_allocated_bytes-fixed_bytes)/proven_batch
        result['slope_source'] = '(peak_allocated_bytes-fixed_bytes)/proven_batch'
    else:
        result['slope_source'] = 'supplied measured slope'
    if not valid(slope_bytes) or slope_bytes<=0:
        result['reason'] = 'unknown or nonpositive capacity/slope'; return result
    candidate = math.floor((usable_bytes-fixed_bytes)/slope_bytes)
    result.update(candidate_batch=max(proven_batch, candidate),
                  slope_bytes=slope_bytes,
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
