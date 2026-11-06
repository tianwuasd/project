"""Fresh admission for independent Xu stages. CUDA indices always refer to the job's visible devices."""
import json
import os
import subprocess
import sys
from uuid import UUID

import psutil

from .resources import GIB, budget, snapshot

_applied_group = None
_applied_host = None


def applied_host_budget():
    """Last successful effective host budget in this worker, never in a forked child."""
    return _applied_host[1] if _applied_host and _applied_host[0] == os.getpid() else None


def require_applied_group(names):
    """A forked or direct low-level caller cannot inherit a successful GPU admission."""
    ordered = tuple(str(n) for n in names)
    if len(ordered) > 1 and _applied_group != (os.getpid(), ordered):
        raise ResourceUnavailable('This process has not applied allocator limits for its full GPU group')


class ResourceUnavailable(RuntimeError):
    """A requested stage cannot be admitted with the current resource evidence."""


def gpu_budget(free, total, policy):
    if policy == 'v2':
        reserve = max(4 * GIB, (total + 9) // 10)
        return max(0, free - reserve)
    if policy == 'legacy':
        return max(0, min(4 * free // 5, free - GIB))
    raise ValueError(f'Unknown Xu resource policy: {policy}')


def _device_info(selection):
    import torch
    device = torch.device(selection)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ResourceUnavailable(f'Requested CUDA device unavailable: {selection}')
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ResourceUnavailable(f'Device outside the visible CUDA set: {selection}')
    props = torch.cuda.get_device_properties(index)
    # torch reports an unprefixed CUDA UUID. Match only the same physical UUID
    # in nvidia-smi; physical indices may differ under CUDA_VISIBLE_DEVICES.
    raw = str(getattr(props, 'uuid', ''))
    try:
        uuid = 'GPU-' + str(UUID(raw.removeprefix('GPU-')))
    except ValueError:
        uuid = None
    free, total = torch.cuda.mem_get_info(index)
    return dict(device=f'cuda:{index}', uuid=uuid, name=props.name,
                free_bytes=int(free), total_bytes=int(total))


def _compute_pids(uuid):
    if uuid is None:
        return None
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--id='+uuid, '--query-compute-apps=gpu_uuid,pid',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            check=True, timeout=5)
        pids = set()
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            row_uuid, pid = (part.strip() for part in line.split(','))
            if row_uuid.casefold() != uuid.casefold() or int(pid) <= 0:
                return None
            pids.add(int(pid))
        return pids
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _identities(pids):
    identities = {}
    for pid in pids:
        try:
            identities[str(pid)] = psutil.Process(pid).create_time()
        except psutil.Error:
            continue
    return identities


def _owned_identities(pids):
    """Whitelist the scheduler and its current direct children, with birth checks."""
    parent = os.getpid()
    kept = _identities([parent])
    for key in pids:
        try:
            pid = int(key)
            if pid == parent:
                continue
            p = psutil.Process(pid)
            birth = p.create_time()
            if isinstance(pids, dict) and pids[key] != birth:
                continue
            if p.ppid() == parent and p.is_running():
                kept[str(pid)] = birth
        except (ValueError, psutil.Error):
            continue
    return kept


def _peer_credit(peers, parent, outstanding):
    """Only verified direct children can repay the peer's reserved RAM share."""
    if not isinstance(peers, dict) or outstanding <= 0:
        return 0
    credited = 0
    for key, birth in peers.items():
        try:
            p = psutil.Process(int(key))
            if p.pid == os.getpid() or p.ppid() != parent or p.create_time() != birth:
                continue
            uss = p.memory_full_info().uss
            if p.is_running() and p.ppid() == parent:
                credited += max(0, int(uss))
        except (psutil.Error, ValueError, TypeError, AttributeError, NotImplementedError):
            continue
    return min(credited, outstanding)


def _check_device(info, policy, owned, required):
    pids = _compute_pids(info['uuid']) if policy == 'v2' else None
    if required and pids is None:
        raise ResourceUnavailable(f"No GPU PID evidence for {info['device']} ({info['uuid']})")
    if pids is not None:
        live = _identities(pids)
        # Treat unmappable host PIDs, including those outside a container's PID
        # namespace, as external. Never act on or signal any observed process.
        external = [pid for pid in pids if str(pid) not in live or
                    owned.get(str(pid)) != live[str(pid)]]
        if external:
            raise ResourceUnavailable(f"External compute processes on {info['device']}: {sorted(external)}")
    limit = gpu_budget(info['free_bytes'], info['total_bytes'], policy)
    if limit < 1:
        raise ResourceUnavailable(f"No CUDA budget left on {info['device']}")
    return dict(info, allocator_budget_bytes=limit, pid_evidence=pids is not None,
                compute_pids=sorted(pids or ()))


def _host_snapshot(policy):
    facts = snapshot()
    if policy == 'v2' and sys.platform.startswith('linux'):
        # NUMA and disk-pressure telemetry can be absent. The affinity, idle,
        # quota and remaining RAM readings used in admission must be evidenced.
        required = [m for m in facts.get('missing_evidence', ()) if any(
            word in m.casefold() for word in ('affinity','permitted-cpu','quota','cgroup',
                                              'cpuset','memory limit','memavailable','ram'))]
        if (not facts.get('cgroup_evidence') or facts.get('affinity_source') != 'sched_getaffinity'
                or not facts.get('cpu_availability_source','').startswith('permitted_cpu_idle_sample')):
            required.append('missing measured CPU affinity/availability or cgroup hierarchy')
        if required:
            raise ResourceUnavailable(f'Cannot confirm Linux CPU/RAM admission: {required}')
    return facts


def _demand(request, key):
    n = request[key] if isinstance(request, dict) else request
    if type(n) is not int or n < 0:
        raise ValueError('Expected a nonnegative per-stage working set')
    return n


def reserve_wave(threads, policy, lanes=2):
    """Freeze equal CPU/RAM shares before either independent branch starts."""
    if lanes != 2 or policy != 'v2' or (threads is not None and (type(threads) is not int or threads < 1)):
        raise ValueError('A parallel Xu wave requires two v2 lanes and positive threads')
    initial = budget(_host_snapshot(policy), {})
    slots, memory = initial['cpu_slots'], initial['memory_bytes']
    if slots is None or slots < lanes or not memory:
        raise ResourceUnavailable(f'Insufficient CPU/RAM to reserve a pair: {initial}')
    ceiling = slots // lanes
    return dict(lanes=lanes, policy=policy, parent=os.getpid(),
                threads=min(threads, ceiling) if threads else ceiling,
                host_per_stage_bytes=memory // lanes)


def stage_resources(devices, concurrent=1, threads=None, policy='v2', *, owned_pids=(),
                    require_pid_evidence=None, host_demand_bytes=0, cuda_demand_bytes=0,
                    wave_reservation=None, peer_active_pids=(), peer_finished=False):
    """Plan per-stage shares using one fresh affinity/cgroup/host sample."""
    if (not devices or isinstance(devices, str) or type(concurrent) is not int or concurrent < 1
            or (threads is not None and (type(threads) is not int or threads < 1))
            or policy not in ('v2','legacy')):
        raise ValueError('Expected devices, positive concurrency/threads and a known policy')
    wave_evidence = None
    if wave_reservation is not None:
        w = wave_reservation
        if w['parent'] != os.getpid() or w['policy'] != policy or w['lanes'] != concurrent:
            raise ValueError('Wave reservation belongs to another scheduler or resource policy')
        fair = w['host_per_stage_bytes']
        reserved = 0 if peer_finished else fair
        before = _peer_credit(peer_active_pids, w['parent'], reserved)
    facts = _host_snapshot(policy)
    current = budget(facts, {})
    slots, host = current['cpu_slots'], current['memory_bytes']
    minimum_slots = concurrent if wave_reservation is None else 1
    if slots is None or slots < minimum_slots or not host:
        raise ResourceUnavailable(f'Insufficient fresh CPU/RAM for {concurrent} stages: {current}')
    per_host = host // concurrent
    per_cpu = slots // concurrent
    if wave_reservation is not None:
        # Bracket the host reading: pages allocated after that reading cannot
        # be credited against the earlier, still larger available-memory value.
        credited = min(before, _peer_credit(peer_active_pids, w['parent'], reserved))
        pending = reserved - credited
        per_host = min(fair, max(0, host - pending))
        per_cpu = min(w['threads'], slots)
        wave_evidence = dict(parent=w['parent'],fair_share_bytes=fair,peer_reserved_bytes=reserved,
                             peer_credited_bytes=credited,peer_outstanding_bytes=pending,
                             peer_finished=peer_finished, peer_identities=(
                                 dict(peer_active_pids) if isinstance(peer_active_pids,dict) else {}))
        if per_host <= 0:
            raise ResourceUnavailable(f'No fresh host RAM after peer reservation: {wave_evidence}')
    devices = list(devices)
    required = bool(require_pid_evidence) or (policy == 'v2' and sys.platform.startswith('linux') and
                                              (len(devices) > 1 or concurrent > 1))
    owned = _owned_identities(owned_pids)
    cards = {}
    for selection in devices:
        if selection == 'cpu':
            info = dict(device='cpu', uuid=None, allocator_budget_bytes=None)
        else:
            info = _check_device(_device_info(selection), policy, owned, required)
        key = info['device']
        if key in cards or (info['uuid'] and any(card['uuid'] == info['uuid'] for card in cards.values())):
            raise ValueError(f'Duplicate allocated device: {selection}')
        host_need, cuda_need = _demand(host_demand_bytes,key), _demand(cuda_demand_bytes,key)
        if host_need > per_host:
            raise ResourceUnavailable(f'{key} needs {host_need} host bytes, available share {per_host}')
        if cuda_need > (info['allocator_budget_bytes'] or 0):
            raise ResourceUnavailable(f'{key} needs {cuda_need} CUDA bytes, budget {info["allocator_budget_bytes"]}')
        cards[key] = dict(info, host_demand_bytes=host_need, cuda_demand_bytes=cuda_need)
    return dict(policy=policy, threads=min(threads, per_cpu) if threads else per_cpu,
                host_per_stage_bytes=per_host, host_budget_bytes=host, cpu_slots=slots,
                cpu_permitted=facts.get('cpu_permitted'), cpu_available=facts.get('cpu_available'),
                missing_evidence=facts.get('missing_evidence', []), concurrent=concurrent,
                require_pid_evidence=required, owned=owned, devices=cards, wave=wave_evidence)


def limit_environment(plan, device):
    """Build the child environment overlay. Keep all scheduler visibility restrictions."""
    card = plan['devices'][device]
    overlay = {k: str(plan['threads']) for k in
               ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','VECLIB_MAXIMUM_THREADS')}
    overlay.update(XU_RESOURCE_POLICY=plan['policy'],XU_HOST_BUDGET_BYTES=str(plan['host_per_stage_bytes']),
                   XU_HOST_DEMAND_BYTES=str(card['host_demand_bytes']))
    if plan.get('wave') is not None:
        overlay['XU_WAVE_RESERVATION'] = json.dumps(plan['wave'])
    if device != 'cpu':
        overlay['XU_CUDA_ADMISSION'] = json.dumps(dict(
            device=device, uuid=card['uuid'], limit_bytes=card['allocator_budget_bytes'],
            cuda_demand_bytes=card['cuda_demand_bytes'],
            policy=plan['policy'], require_pid_evidence=plan['require_pid_evidence'],
            owned=plan['owned']))
    return overlay


def combined_environment(plan, devices):
    """Give one worker exactly the freshly admitted devices for this stage."""
    names = list(devices)
    if not names or len(set(names)) != len(names) or any(n not in plan['devices'] for n in names):
        raise ValueError('Stage device list is empty, duplicated or lacks a reservation')
    overlay = limit_environment(plan,names[0])
    if len(names) > 1:
        if 'cpu' in names or plan['policy'] != 'v2':
            raise ValueError('Distributed stages require v2 CUDA admission on every device')
        tickets = [json.loads(limit_environment(plan,n)['XU_CUDA_ADMISSION']) for n in names]
        if any(not t['uuid'] for t in tickets):
            raise ResourceUnavailable('A visible GPU has no verifiable UUID')
        overlay['XU_CUDA_ADMISSIONS'] = json.dumps(tickets)
        overlay['XU_BRANCH_DEVICES'] = json.dumps(names)
        overlay['XU_HOST_DEMAND_BYTES'] = str(max(plan['devices'][n]['host_demand_bytes'] for n in names))
    return overlay


def apply_env_limit(device):
    """Recheck admission *inside* the worker and install its allocator ceiling."""
    global _applied_group, _applied_host
    _applied_group = None
    _applied_host = None
    ticket = os.environ.get('XU_CUDA_ADMISSION')
    group = os.environ.get('XU_CUDA_ADMISSIONS')
    members = os.environ.get('XU_BRANCH_DEVICES')
    if bool(group) != bool(members) or (group and not ticket):
        # A single allocated device can also be listed explicitly by callers.
        if group or not ticket or json.loads(members) != [str(device)]:
            raise ResourceUnavailable('Missing or unmatched device group admission')
    share = os.environ.get('XU_HOST_BUDGET_BYTES')
    if not ticket and not share:
        return {}
    wave = json.loads(os.environ['XU_WAVE_RESERVATION']) if os.environ.get('XU_WAVE_RESERVATION') else None
    if wave:
        reserved = wave['peer_reserved_bytes']
        # Windows venv launcher indirection may hide the immediate scheduler.
        # Without a known shared parent, keep the entire peer reservation.
        peers = wave['peer_identities'] if wave['parent'] == os.getppid() else {}
        before = _peer_credit(peers, wave['parent'], reserved)
    real_host = budget(_host_snapshot(os.environ.get('XU_RESOURCE_POLICY','legacy')), {})['memory_bytes']
    if not real_host or not share:
        raise ResourceUnavailable('Missing fresh host RAM budget or stage allocation')
    if wave:
        credited = min(before, _peer_credit(peers, wave['parent'], reserved))
        real_host = max(0, real_host - (reserved - credited))
    host = min(real_host, int(share))
    if host <= 0 or host < int(os.environ.get('XU_HOST_DEMAND_BYTES','0')):
        raise ResourceUnavailable(f'Insufficient fresh host RAM: {host} bytes')
    result = dict(host_budget_bytes=host)
    if not ticket:
        if str(device) != 'cpu':
            raise ResourceUnavailable('CUDA worker missing GPU admission ticket')
        _applied_host = (os.getpid(),host)
        return result
    import torch
    spec = json.loads(ticket)
    requested = torch.device(device)
    if requested.type != 'cuda' or requested.index is None or str(requested) != spec['device']:
        raise ResourceUnavailable(f'Expected explicit assigned CUDA device {spec["device"]}, got {device}')
    if not group:
        result.update(_apply_card(device,spec))
        _applied_host = (os.getpid(),host)
        return result
    tickets = json.loads(group)
    names = json.loads(members)
    if (not isinstance(tickets,list) or not tickets or tickets[0] != spec or not isinstance(names,list)
            or [t['device'] for t in tickets] != names or len(set(names)) != len(names)
            or any(not t['uuid'] or t['policy'] != 'v2' for t in tickets)
            or len({t['uuid'] for t in tickets}) != len(tickets)):
        raise ResourceUnavailable('GPU group tickets do not match the allocated stage')
    try:
        cards = {t['device']:_apply_card(t['device'],t) for t in tickets}
    finally:
        torch.cuda.set_device(requested)
    _applied_group = (os.getpid(),tuple(names))
    _applied_host = (os.getpid(),host)
    return dict(result, **cards[str(requested)], per_device=cards)


def _apply_card(device, spec):
    import torch
    # Select the card before any property or memory access can initialize a context.
    torch.cuda.set_device(torch.device(device))
    info = _device_info(device)
    if spec['device'] != info['device'] or spec['uuid'] != info['uuid']:
        raise ResourceUnavailable(f'Allocated CUDA identity changed for {device}')
    owned = {**spec['owned'], **_identities([os.getpid()])}
    info = _check_device(info, spec['policy'], owned, spec['require_pid_evidence'])
    limit = min(info['allocator_budget_bytes'], int(spec['limit_bytes']))
    if limit <= 0 or limit < spec['cuda_demand_bytes']:
        raise ResourceUnavailable(f'CUDA demand {spec["cuda_demand_bytes"]} exceeds fresh budget {limit}')
    torch.cuda.set_per_process_memory_fraction(limit/info['total_bytes'], device)
    info.update(allocator_budget_bytes=limit, cuda_demand_bytes=spec['cuda_demand_bytes'])
    return dict(info, policy=spec['policy'])
