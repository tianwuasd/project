"""Natural full-epoch expansion regression guard; no trial ranking or extra work."""
import math
import statistics


def expansion_baseline(result,global_batch,previous_gpus):
    rows=result.get('epoch_performance',[])
    # Require one warmup plus three completed observations in the old group.
    if len(rows)<4:return None
    recent=rows[-3:]
    if any(r.get('global_batch')!=global_batch or not _valid(r.get('seconds')) for r in recent):return None
    return dict(global_batch=global_batch,reference_seconds=statistics.median(r['seconds'] for r in recent),
        previous_gpu_uuids=list(previous_gpus),baseline_epochs=[r['epoch'] for r in recent],
        regression_ratio=1.25,consecutive_epochs=3,warmup_epochs=1,
        scope='same frozen patch, global batch and 250 train/50 validation attempts; natural full-epoch wall time')


def _valid(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0


class ExpansionGuard:
    def __init__(self,baseline,global_batch):
        self.baseline=baseline or {};self.global_batch=global_batch;self.observed=0;self.samples=[]

    def observe(self,epoch,seconds):
        self.observed+=1
        result=dict(rollback=False,epoch=epoch,seconds=seconds,baseline=self.baseline,samples=list(self.samples))
        if (not _valid(self.baseline.get('reference_seconds')) or self.baseline.get('global_batch')!=self.global_batch
                or not _valid(seconds)):
            return dict(result,status='not_comparable')
        if self.observed<=self.baseline.get('warmup_epochs',1):return dict(result,status='warmup')
        self.samples.append(seconds);self.samples=self.samples[-self.baseline.get('consecutive_epochs',3):]
        threshold=self.baseline['reference_seconds']*self.baseline.get('regression_ratio',1.25)
        rollback=len(self.samples)>=self.baseline.get('consecutive_epochs',3) and all(v>threshold for v in self.samples)
        return dict(result,status='regression' if rollback else 'observing',rollback=rollback,
            samples=list(self.samples),threshold_seconds=threshold)
