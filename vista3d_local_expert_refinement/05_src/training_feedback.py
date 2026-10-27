"""All-rank rolling supply feedback inside a long epoch; no GPU membership change."""
import copy
import time
from pathlib import Path


def slowest_window(rows):
    if not rows or any(r['steps']<20 or r['seconds']<30 for r in rows):return None
    slow=max(rows,key=lambda r:r['seconds'])
    return dict(slow,rank_windows=rows,selection='slowest rank wall time')


def source_pool(loader):
    while hasattr(loader,'source'):loader=loader.source
    return loader


def apply_pools(trainer,resource,rank):
    row=resource['rank_settings'][rank]
    loaders=[trainer.dataloader_train,trainer.dataloader_val]
    if not all(getattr(getattr(source_pool(l),'loader',source_pool(l)),'replayable',False) for l in loaders):return False
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

    def reset(self):
        self.steps=0;self.wait=0;self.started=self.clock()

    def observe(self,timing,trainer,dist):
        self.steps+=1;self.wait+=timing['batch_wait_seconds']
        if self.steps%20:return
        seconds=self.clock()-self.started
        local=dict(rank=self.rank,steps=self.steps,seconds=seconds,wait_fraction=self.wait/max(seconds,1e-9))
        rows=[None]*self.world
        # Every rank reaches exactly the same update boundary. A failed worker
        # raises locally; the parent supervisor terminates the blocked peers.
        dist.all_gather_object(rows,local)
        window=slowest_window(rows)
        if window is None:return
        self.windows.append(window)
        from live_resources import feedback_policy
        from server_data import write_json
        packet=[None]
        if self.rank==0:
            packet[0]=feedback_policy(self.resource,self.windows,self.capacity)
            packet[0]['capacity_scope']='retained stage allocation; no own-busy idle/RAM subtraction; extra external capacity unmeasured'
        dist.broadcast_object_list(packet,src=0)
        decision=packet[0]
        if decision['action'] in ('expand','recover'):
            applied=apply_pools(trainer,decision['settings'],self.rank)
            if applied is False:self.pending=decision['settings'];decision['applied_boundary']='deferred_full_epoch'
            else:
                self.resource=decision['settings'];decision['applied_boundary']='replay_safe_batch'
                decision['affinity']=applied
        else:decision['applied_boundary']='none'
        self.events.append(dict(schema_version=2,epoch=trainer.current_epoch,window=window,decision=decision))
        write_json(self.output/f'feedback.rank{self.rank}.json',self.events)
        self.reset()

    def epoch_boundary(self,trainer):
        if self.pending is not None:
            # Unsupported source does not claim safe mutation. The coordinator
            # can restart the group from the complete checkpoint to apply it.
            return self.pending
        return None
