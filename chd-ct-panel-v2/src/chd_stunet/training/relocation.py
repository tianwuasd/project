"""GPU contention changes placement, never the global batch or data cursor."""
import json
from .fleet import eligible


def choose_members(previous,rows,maximum):
    """Keep eligible survivors in rank order, fill from fresh idle UUIDs."""
    available=[row['uuid'] for row in eligible(rows)]
    survivors=[uuid for uuid in previous if uuid in available][:maximum]
    return (survivors+[uuid for uuid in available if uuid not in survivors])[:maximum]


def foreign_on_members(members,rows,owned_pids):
    return {row['uuid']:sorted(set(row['external_pids'])-owned_pids) for row in rows
            if row['uuid'] in members and set(row['external_pids'])-owned_pids}


def read_handoff(output,generation):
    path=output/'control/handoff.json'
    if not path.is_file():return None
    request=json.loads(path.read_text(encoding='utf-8'))
    return request if request.get('generation')==generation else None


def retryable_failure(fatals,external):
    """Retry proven resource loss, never conceal a data/contract error."""
    if not fatals:return False
    def communication(row):
        return row.get('error_type') in ('OutOfMemoryError','DistBackendError','DistNetworkError') or (
            row.get('error_type')=='RuntimeError' and any(token in row.get('error','') for token in ('NCCL','Connection closed by peer')))
    return all(row.get('resource_unavailable') or communication(row) for row in fatals) and (
        bool(external) or any(row.get('resource_unavailable') for row in fatals))
