from dataclasses import replace
import math
from pathlib import Path
from time import monotonic

from filelock import FileLock

from ..artifacts.store import read_json, write_json
from ..observability.events import event
from .memory import alive
from .limits import memory_ceiling, project_usage
from .host import host_capacity


def memory_floor(state):
    idle = max(0, state['workspace_capacity'] - state['workspace_reserved'])
    used = state['task_rss'] + state['unrealized_bytes']
    required = used + idle + 1
    waiting = {}
    for token, row in state.get('waiting', {}).items():
        if not alive(row):
            continue
        waiting[token] = row
        if row['kind'] == 'case':
            if (state['active_readers'] >= state['reader_slots'] or
                    state['workspace_reserved'] + row['bytes'] > state['workspace_capacity']):
                continue
            candidate = used + row['bytes']
        else:
            candidate = used + idle + row['bytes']
        required = max(required, candidate)
    return required, waiting


def grow_memory(session, spare):
    h = session.handle
    with h.condition:
        if h.cancel.is_set() or h.state.get('replanning', False):
            return
        required, waiting = memory_floor(h.state)
        h.state['waiting'] = waiting
        if required <= session.capacity:
            return
    root, r = Path(session.request.registry_root), session.request.resources
    with FileLock(str(root / '.resource.lock')), h.condition:
        if h.cancel.is_set() or h.state.get('replanning', False):
            return
        required, waiting = memory_floor(h.state)
        if required <= session.capacity:
            return
        rows = [row for row in read_json(root / 'active.json') if alive(row)]
        own = next(row for row in rows if row['pid'] == session.owner['pid'] and row['birth'] == session.owner['birth'])
        peers = [row for row in rows if row is not own]
        limit = memory_ceiling(r, host_capacity())
        rss = h.state['task_rss']
        peer_used = project_usage(peers, r)
        peer_pending = sum(max(0, row['ram_bytes'] - row.get('rss_bytes', 0)) for row in peers)
        budget = min(limit - peer_used, rss + spare - peer_pending)
        target = math.ceil(required * (1 + r.get('memory_margin_fraction', .5)))
        fields = dict(required_bytes=required, requested_bytes=target, available_budget_bytes=budget,
                      previous_bytes=session.capacity, waiting=list(waiting.values()))
        if target > budget:
            if monotonic() - getattr(session, 'memory_wait_log', -float('inf')) >= 15:
                event(session.request.run_dir, 'memory_capacity_waiting', **fields)
                session.memory_wait_log = monotonic()
            return
        generation = h.state['plan_generation'] + 1
        own.update(ram_bytes=target, rss_bytes=rss, plan_generation=generation)
        write_json(root / 'active.json', rows, durable=False)
        session.rows, session.capacity = rows, target
        session.applied_demand = replace(session.applied_demand,
            fixed_bytes=session.applied_demand.fixed_bytes + target - h.state['capacity'])
        h.state.update(capacity=target, plan_generation=generation, project_bytes=peer_used+target,
            headroom=max(0, min(target-rss, spare-peer_pending)),
            paused=spare-peer_pending < 0 or rss > target or peer_used+target > limit)
        session.allocation.memory_bytes = target
        session.allocation.applied = session._applied()
        h.condition.notify_all()
    event(session.request.run_dir, 'memory_capacity_increased', memory_bytes=target,
          plan_generation=generation, **fields)
