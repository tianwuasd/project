import json
import os
import time

from .contracts import atomic_json, read_json
from .resources import GIB, snapshot
from .xu_resources import ResourceUnavailable, HostResourcesChanged, GpuResourcesChanged, _identities


def external_processes(pids, owned):
    live = _identities(pids)
    return sorted(pid for pid in pids if str(pid) not in live or owned.get(str(pid)) != live[str(pid)])


def isolation_violations(inventory, processes, stage, allowed):
    violations = {}
    for uuid,card in inventory.items():
        live = _identities(card['compute_pids'])
        found = [pid for pid in card['compute_pids'] if str(pid) in processes
                 and live.get(str(pid)) == processes[str(pid)]['create_time']
                 and (processes[str(pid)]['stage'] is None or
                      (processes[str(pid)]['stage'] == stage and uuid not in allowed))]
        if found:
            violations[uuid] = found
    return violations


class RuntimeGuard:
    def __init__(self, output, stage, overlay, children):
        self.output = output
        self.output.mkdir(parents=True,exist_ok=True)
        self.stage = stage
        self.children = children
        self.tickets = json.loads(overlay.get('XU_CUDA_ADMISSIONS','[]'))
        if not self.tickets and overlay.get('XU_CUDA_ADMISSION'):
            self.tickets = [json.loads(overlay['XU_CUDA_ADMISSION'])]
        available = snapshot()['available_memory_bytes']
        if available is None:
            raise ResourceUnavailable('无法核实运行期间主机内存')
        self.reserve_bytes = max(4*GIB,available//5)
        self.host_limit = int(overlay.get('XU_HOST_BUDGET_BYTES','0'))
        self.started = time.monotonic()

    def check(self):
        self.children.ensure_running()
        processes = self.children.owned_processes()
        owned = {pid:row['create_time'] for pid,row in processes.items()}
        available = snapshot()['available_memory_bytes']
        row = dict(state='running',stage=self.stage,checked_at=time.time(),
                   seconds=time.monotonic()-self.started,owned=owned,
                   available_memory_bytes=available,reserve_bytes=self.reserve_bytes,
                   host_limit_bytes=self.host_limit,gpus=[])
        reason = None
        gpu_changed = []
        if available is None or available < self.reserve_bytes:
            reason = '实时主机或 cgroup 内存低于保留值'
        stage_owned = [p for p in processes.values() if p['stage'] == self.stage]
        if any(p['memory_full_info'] is None for p in stage_owned):
            raise ResourceUnavailable('无法核实本阶段进程内存')
        row['stage_uss_bytes'] = sum(p['memory_full_info'].uss for p in stage_owned)
        if self.host_limit and row['stage_uss_bytes'] > self.host_limit:
            reason = '本阶段内存超过分配预算'
        host_reason = reason
        isolation_error = None
        if self.tickets or json.loads(os.environ.get('XU_COORDINATOR_DEVICES','[]')):
            from .xu_gpu_probe import gpu_snapshot
            inventory = gpu_snapshot()
            observed = {pid for ticket in self.tickets for pid in inventory[ticket['uuid']]['compute_pids']}
            if external_processes(observed, owned):
                # 显卡查询后重新核实本组进程，涵盖并发启动的分支和后代。
                inventory = gpu_snapshot()
                processes = self.children.owned_processes()
                owned = {pid:p['create_time'] for pid,p in processes.items()}
                row['owned'] = owned
            violations = isolation_violations(inventory,processes,self.stage,{t['uuid'] for t in self.tickets})
            row['isolation_violations'] = violations
            if violations:
                isolation_error = f'GPU 隔离违规，本任务进程出现在未分配设备: {violations}'
            for ticket in self.tickets:
                card = inventory[ticket['uuid']]
                external = external_processes(card['compute_pids'], owned)
                floor = max(4*GIB,card['total_bytes']//10)
                row['gpus'].append(dict(card,device=ticket['device'],external_pids=external,
                                         reserve_bytes=floor))
                if external:
                    reason = f"{ticket['device']} 出现外部计算进程 {external}"
                    gpu_changed.append(ticket['device'])
                elif card['free_bytes'] < floor:
                    reason = f"{ticket['device']} 显存低于保留值"
                    gpu_changed.append(ticket['device'])
        reason = isolation_error or host_reason or reason
        if reason:
            row.update(state='denied',reason=reason)
        atomic_json(self.output/(self.stage+'-runtime.json'),row)
        if reason:
            atomic_json(self.output/(self.stage+f'-interrupted-{time.time_ns()}.json'),row)
            if isolation_error:
                raise RuntimeError(isolation_error)
            if gpu_changed and not host_reason and not isolation_error:
                raise GpuResourcesChanged(reason, gpu_changed)
            raise HostResourcesChanged(reason)
        return row

    def finish(self, code):
        path = self.output/(self.stage+'-runtime.json')
        row = read_json(path) if path.exists() else dict(stage=self.stage)
        if row.get('state') != 'denied':
            row['state'] = 'finished' if code == 0 else 'interrupted'
        row.update(exit_code=code,monitor_ended_at=time.time())
        atomic_json(path,row)
