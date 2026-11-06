"""Epoch-boundary adjustments to CPU supply for a synchronous training group.

Keep two short windows from each rank. DDP peers that got their batch first
can spend the missing rank's entire wait inside the measured step body. Plan
from each rank's queue wait, then judge the trial by whole-epoch wall time.
"""
from collections import deque
import math
from statistics import mean, median

from ..acceleration.preprocess_cpu import GIB, snapshot
from .collectives import all_objects


class RankSupplyWindow:
    def __init__(self):
        self.recent = deque(maxlen=2)
        self.pending = []

    def observe(self, wait, body, supply):
        self.pending.append((wait, body, supply))
        if len(self.pending) == 20:
            wait, body, supply = map(mean, zip(*self.pending))
            self.recent.append((wait, wait / max(wait + body, 1e-9), supply))
            self.pending.clear()

    def view(self):
        if len(self.recent) < 2:
            return None
        waits, fractions, service = zip(*self.recent)
        return dict(wait=mean(waits), fraction=min(fractions), service=mean(service))


def _admit(current, proposed, estimates, stage, fresh, hard_cap, step_limit):
    if hard_cap is not None and sum(proposed) > hard_cap:
        return False
    if max(proposed) > step_limit or min(proposed) < 0:
        return False
    growing = sum(proposed) > sum(current)
    # The idle sample already excludes busy workers. Test the incremental
    # CPU requirement against that sample, plus the absolute stage ceiling.
    permitted = fresh.get('cpu_permitted')
    quota = fresh.get('cgroup_cpu_quota')
    if quota is not None and permitted is not None:
        permitted = min(permitted, quota)
    if permitted is None or sum(proposed) + len(proposed) + 1 > math.floor(.8 * permitted):
        return False
    if growing and (fresh.get('cpu_budget') or 0) < 1:
        return False
    free = fresh.get('available_memory_bytes')
    if free is None:
        return False
    reserve = max(stage['reserve_bytes'], 4 * GIB, math.ceil(.2 * free))
    # MemAvailable/cgroup remaining already excludes live private memory. Only
    # credit verified direct workers up to each reservation; a missing birth
    # or USS observation leaves the full estimate outstanding. Drained donors
    # finish teardown before receivers start submitting work.
    credits = fresh.get('worker_private_credit', [0] * len(current))
    outstanding = sum(max(0, target * size - min(max(0, credit), count * size))
                      for target, size, count, credit in zip(proposed, estimates, current, credits))
    return outstanding + GIB <= min(fresh.get('memory_budget_bytes') or 0, free - reserve)


def next_worker_layout(workers, measures, estimates, stage, fresh, hard_cap, step_limit):
    """Admit at most one new worker, or transfer one under a fixed ceiling."""
    hold = dict(workers=list(workers), reason='stable_or_insufficient_evidence')
    if any(row is None for row in measures):
        return hold
    starving = [i for i, row in enumerate(measures) if row['fraction'] >= .15 and row['wait'] >= .02]
    if not starving:
        return hold
    slow = max(starving, key=lambda i: measures[i]['wait'])
    candidate = list(workers)
    candidate[slow] += 1
    if _admit(workers, candidate, estimates, stage, fresh, hard_cap, step_limit):
        return dict(workers=candidate, reason='sustained_queue_wait_with_headroom')

    # At a fixed limit, move a worker only if the donor was consistently
    # supplied and its predicted per-worker service time leaves a margin.
    demand = [row['service'] for row in measures]
    def bottleneck(layout):
        return max(t / max(1, count) for t, count in zip(demand, layout))
    donors = [i for i, n in enumerate(workers) if i != slow and n >= 2
              and measures[i]['wait'] < .2 * measures[slow]['wait']]
    options = []
    for donor in donors:
        trial = candidate.copy()
        trial[donor] -= 1
        if _admit(workers, trial, estimates, stage, fresh, hard_cap, step_limit):
            options.append(trial)
    if options:
        best = min(options, key=bottleneck)
        if bottleneck(best) < .9 * bottleneck(workers):
            return dict(workers=best, reason='transfer_one_worker_to_waiting_rank')
    return dict(workers=list(workers), reason='retain_resource_limit')


class SupplyTrials:
    """Require measured benefit before accepting further speculative growth."""
    def __init__(self):
        self.trial = None
        self.cooldown = 0
        self.defer_trials = {}

    def decide(self, workers, measures, estimates, stage, fresh, hard_cap, step_limit, seconds):
        def hold(reason):
            return dict(workers=list(workers), reason=reason)
        self.defer_trials = {pair: n - 1 for pair, n in self.defer_trials.items() if n > 1}
        if self.cooldown:
            self.cooldown -= 1
            return hold('post_trial_cooldown')
        if self.trial is not None:
            trial = self.trial
            trial['epochs'].append(seconds)
            # First epoch includes creation of any resized CPU executors.
            if len(trial['epochs']) < 3:
                return hold('measuring_trial')
            self.trial = None
            if median(trial['epochs'][1:]) > .95 * trial['baseline']:
                self.cooldown = 4
                previous = trial['previous']
                self.defer_trials[(tuple(previous), tuple(workers))] = 24
                if all(a <= b for a,b in zip(previous,workers)) or _admit(
                        workers, previous, estimates, stage, fresh, hard_cap, step_limit):
                    return dict(workers=previous, reason='trial_gain_below_five_percent')
                return hold('retain_trial_due_to_headroom')
        decision = next_worker_layout(workers, measures, estimates, stage, fresh, hard_cap, step_limit)
        if decision['workers'] != workers:
            if (tuple(workers), tuple(decision['workers'])) in self.defer_trials:
                return hold('trial_previously_unhelpful')
            self.trial = dict(previous=list(workers), baseline=seconds, epochs=[])
        return decision


def agree_on_layout(rank, window, private_credit, trials, workers, supply, hard_cap, step_limit, seconds):
    """One rank samples the host and decides; every DDP rank uses that decision."""
    reports = all_objects((window.view(), private_credit))
    decision = None
    if rank == 0:
        measures, credits = zip(*reports)
        fresh = dict(snapshot(), worker_private_credit=credits)
        decision = trials.decide(workers, measures, supply['worker_estimate_bytes'],
                                 supply['stage'], fresh, hard_cap, step_limit, seconds)
        decision.update(previous=list(workers), windows=measures, cpu_budget=fresh.get('cpu_budget'),
                        available_memory_bytes=fresh.get('available_memory_bytes'), private_credit=credits)
    return all_objects(decision)[0]
