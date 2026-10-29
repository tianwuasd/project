"""All-rank rolling supply feedback inside a long epoch; no GPU membership change."""
import copy
import json
import time
from pathlib import Path


def compact_resource(resource):
    return {key:resource[key] for key in ('schema_version','rank_count','cpu_budget','memory_budget_bytes',
        'total_cpu_slots','estimated_ram_bytes','rank_settings','local_batch_sizes') if key in resource}


def slowest_window(rows):
    if not rows or any(r['steps']<20 or r['seconds']<30 for r in rows):return None
    slow=max(rows,key=lambda r:r['seconds'])
    supply=max(rows,key=lambda r:r['wait_fraction'])
    # DDP can make a healthy peer's total wall time slightly longer while it
    # waits for the rank whose data supply stalled. Preserve that supply signal.
    pressure=max((r.get('io_pressure') or {}).get('avg10',0) for r in rows)
    return dict(slow,wait_fraction=supply['wait_fraction'],io_pressure={'avg10':pressure} if any(r.get('io_pressure') is not None for r in rows) else None,slowest_wall_rank=slow['rank'],
                supply_wait_rank=supply['rank'],rank_windows=rows,
                selection='maximum rank supply wait fraction; slowest wall metadata retained')


def source_pool(loader):
    while hasattr(loader,'source'):loader=loader.source
    return loader


def apply_pools(trainer,resource,rank):
    row=resource['rank_settings'][rank]
    loaders=[trainer.dataloader_train,trainer.dataloader_val]
    if not all(getattr(getattr(source_pool(l),'loader',source_pool(l)),'replayable',False) for l in loaders):return False
    from read_admission import create_gate,update_gate
    sources=[getattr(source_pool(loader),'loader',source_pool(loader)) for loader in loaders]
    for source in sources:
        if hasattr(source,'spec'):
            gate=source.spec.get('read_gate')
            limit=row.get('read_concurrency',max(1,row['train_workers'],row['val_workers']))
            if gate:update_gate(gate,limit)
            else:source.spec['read_gate']=str(create_gate(limit))
    # Both outer wrappers rewind to model consumption before inner pool close.
    for loader in loaders:loader.pause()
    from rank_affinity import bind
    affinity=bind(rank,trainer._vista_gpu_uuids,resource)
    for loader,workers,threads,queue,shared in [
        (loaders[0],'train_workers','worker_threads','prefetch','max_shared_bytes'),
        (loaders[1],'val_workers','val_worker_threads','val_prefetch','val_max_shared_bytes')]:
        loader.reconfigure(workers=row[workers],worker_threads=row[threads] if row[workers] else row['main_threads'],
            prefetch=max(1,row[queue]),max_shared_bytes=row.get(shared,0))
    trainer._vista_resource_config=resource
    return affinity


class RollingFeedback:
    def __init__(self,resource,rank,world,output,clock=time.monotonic):
        self.resource=copy.deepcopy(resource);self.capacity=copy.deepcopy(resource)
        self.rank=rank;self.world=world;self.output=Path(output);self.clock=clock
        self.windows=[];self.events=[];self.reset()
        self.pending=None
        self.run_id=f'{time.time_ns()}-{rank}'

    def reset(self):
        self.steps=0;self.wait=0;self.started=self.clock()

    def observe(self,timing,trainer,dist):
        self.steps+=1;self.wait+=timing['batch_wait_seconds']
        if self.steps%20:return
        seconds=self.clock()-self.started
        pressure=None
        try:
            raw=Path('/proc/pressure/io').read_text().splitlines()[0]
            pressure={k:float(v) for k,v in (item.split('=') for item in raw.split()[1:])}
        except (OSError,ValueError,IndexError):pass
        local=dict(rank=self.rank,steps=self.steps,seconds=seconds,wait_fraction=self.wait/max(seconds,1e-9),
            io_pressure=pressure,pressure_scope='host PSI; not attribution to this loader',
            bottleneck='unknown',classification_status='storage/IPC causal classification unavailable')
        rows=[None]*self.world
        # Every rank reaches exactly the same update boundary. A failed worker
        # raises locally; the parent supervisor terminates the blocked peers.
        dist.all_gather_object(rows,local)
        window=slowest_window(rows)
        if window is None:return
        self.windows.append(window)
        self.windows=self.windows[-2:]
        from live_resources import feedback_policy
        from server_data import write_json
        packet=[None]
        if self.rank==0:
            packet[0]=feedback_policy(self.resource,self.windows,self.capacity)
            packet[0]['capacity_scope']='retained stage allocation; no own-busy idle/RAM subtraction; extra external capacity unmeasured'
        dist.broadcast_object_list(packet,src=0)
        decision=packet[0]
        decision['requested_settings']=copy.deepcopy(decision['settings'])
        if decision['action'] in ('expand','recover','limit_reads','restore_reads'):
            applied=apply_pools(trainer,decision['settings'],self.rank)
            if applied is False:self.pending=decision['settings'];decision['applied_boundary']='deferred_full_epoch'
            else:
                self.resource=decision['settings'];decision['applied_boundary']='replay_safe_batch'
                decision['affinity']=applied
        else:decision['applied_boundary']='none'
        decision['applied_settings']=copy.deepcopy(self.resource)
        self.events.append(dict(schema_version=2,epoch=trainer.current_epoch,window=window,decision=decision))
        self.events=self.events[-2:]
        # The capacity receipt already contains topology and full host probes.
        # Keep the latest state bounded; append detailed changes once, instead
        # of rewriting a growing history from every rank on the training path.
        summary=compact_resource(self.resource)
        latest=dict(schema_version=3,epoch=trainer.current_epoch,recorded_at=time.time(),run_id=self.run_id,rank=self.rank,window=window,
            decision={key:decision[key] for key in ('action','reasons','applied_boundary','capacity_scope') if key in decision},
            applied_settings=summary)
        if decision['action']!='keep':latest['requested_settings']=compact_resource(decision['requested_settings'])
        history=copy.deepcopy(latest)
        if decision['action']=='keep':history.pop('applied_settings')
        with (self.output/f'feedback.rank{self.rank}.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(history,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        write_json(self.output/f'feedback.rank{self.rank}.json',latest)
        self.reset()

    def epoch_boundary(self,trainer):
        if self.pending is not None:
            # Unsupported source does not claim safe mutation. The coordinator
            # can restart the group from the complete checkpoint to apply it.
            return self.pending
        return None
