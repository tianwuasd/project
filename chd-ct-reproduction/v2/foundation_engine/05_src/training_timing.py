"""Observe batch acquisition and step wall time without changing training work."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import time


def _stats(values):
    values = sorted(values)
    if not values:
        raise ValueError('No measured timing samples')
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid timing sample')
    return dict(count=len(values), median=statistics.median(values),
                mean=statistics.mean(values), p95=values[math.ceil(.95*len(values))-1],
                sum=sum(values))


def measure_step(loader, step, synchronize, clock=time.perf_counter):
    # Same two CUDA synchronization boundaries as the original smoke loop.
    synchronize()
    started = clock()
    batch = next(loader)
    loaded = clock()
    result = step(batch)
    synchronize()
    ended = clock()
    return batch, result, dict(batch_wait_seconds=loaded-started,
                              step_seconds=ended-loaded, total_seconds=ended-started)


def summarize_steps(records, warmup=0):
    if not isinstance(warmup, int) or not 0 <= warmup < len(records):
        raise ValueError('Warmup must leave at least one measured step')
    measured = records[warmup:]
    result = dict(warmup_steps=warmup, measured_steps=len(measured))
    for name in ['batch_wait_seconds', 'step_seconds', 'total_seconds']:
        result[name] = _stats([record[name] for record in measured])
    result['scope'] = 'batch acquisition; step includes transfer and device computation, not CUDA kernel time alone'
    return result


class EpochTiming:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.active = False

    def start(self, epoch):
        self.epoch, self.started = epoch, self.clock()
        self.samples = {}
        self.active = True

    @contextmanager
    def measure(self, phase):
        active, started = self.active, self.clock()
        yield
        if active and self.active:
            self.samples.setdefault(phase, []).append(self.clock()-started)

    def finish(self):
        record = dict(schema_version=1, epoch=self.epoch,
                      observed_wall_seconds=self.clock()-self.started,
                      scope='host wall clock; step includes transfer and computation; batch wait includes serial augmentation when workers=0',
                      phases={name: dict(count=len(values), sum_seconds=sum(values),
                                        median_seconds=statistics.median(values),
                                        p95_seconds=_stats(values)['p95'])
                              for name, values in self.samples.items()})
        self.active = False
        return record

    @staticmethod
    def publish(path, record):
        try:
            with Path(path).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        except OSError as error:
            return str(error)
        return None


class TimedIterator:
    def __init__(self, source, timing, stage):
        self.source, self.timing, self.stage = source, timing, stage

    def __iter__(self):
        return self

    def __next__(self):
        with self.timing.measure(self.stage+'_batch_wait'):
            return next(self.source)

    def __getattr__(self, name):
        return getattr(self.source, name)
