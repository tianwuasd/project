import hashlib
import os
from time import monotonic

from ..observability.events import event


def reservation_snapshot(state, amount, purpose, free, began, limit):
    fields = ('capacity','task_rss','fixed_rss','reserved','unrealized_bytes','workspace_capacity',
              'workspace_reserved','active_readers','reader_slots','paused','replanning','monitor_failed',
              'headroom','project_bytes','system_headroom_bytes','pending_bytes','limited_project_bytes',
              'project_limit_bytes','sampled_at','sampling_seconds')
    value = {key:state.get(key) for key in fields}
    claims,waiting = state.get('claims',{}),state.get('waiting',{})
    aggregates = {}
    for kind in ('ordinary','case'):
        rows = [row for row in claims.values() if row['kind']==kind]
        requests = [row for row in waiting.values() if row['kind']==kind]
        aggregates[kind] = dict(claim_count=len(rows),claim_bytes=sum(row['amount'] for row in rows),
            realized_bytes=sum(row['realized'] for row in rows),waiting_count=len(requests),
            waiting_bytes=sum(row['bytes'] for row in requests),max_request_bytes=max((row['bytes'] for row in requests),default=0))
    value.update(requested_bytes=amount,available_bytes=free,pid=os.getpid(),
        purpose_digest=hashlib.blake2s(purpose.encode('utf-8'),digest_size=10).hexdigest(),
        elapsed_seconds=monotonic()-began,timeout_seconds=limit,
        sample_age_seconds=monotonic()-state['sampled_monotonic'],
        idle_pool_bytes=max(0,state.get('workspace_capacity',0)-state.get('workspace_reserved',0)),
        aggregates=aggregates,memory_scopes=state.get('memory_scopes',[]))
    return state.get('diagnostics_dir'),value


def emit_reservation_wait(record, timed_out):
    directory,fields = record
    if directory is not None:
        event(directory,'resource_reservation_timeout' if timed_out else 'resource_reservation_waiting',**fields)
