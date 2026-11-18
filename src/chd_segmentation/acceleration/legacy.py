from contextlib import contextmanager

from .runtime import attach


class SessionGuard:
    def __init__(self, handle):
        self.client = attach(handle)
        self.owner = handle.owner

    def __enter__(self):
        return self

    def snapshot(self):
        value = self.client.snapshot()
        h = self.client.handle
        with h.condition:
            capacity, reserved = h.state['capacity'], h.state['reserved']+h.state.get('fixed_rss',0)
            rss = h.state.get('task_rss', 0)
        free = max(0, min(capacity-reserved, value.headroom_bytes))
        own = dict(accounted_bytes=rss, rss_bytes=rss, pss_bytes=None, uss_bytes=None, process_count=len(h.members))
        budget = dict(headroom_bytes=free, own_pending_bytes=0, own_claim_pending_bytes=0,
                      memory_limit_bytes=capacity, own=own, reason='memory_headroom' if value.paused else 'eligible')
        return dict(time_utc=value.sampled_at, monotonic=value.sampled_monotonic, host=value.host,
                    project=dict(accounted_bytes=value.project_accounted_bytes, complete=True),
                    budget=budget, paused=value.paused, stop_requested=self.client.boundary().action=='stop', peaks={})

    boundary = snapshot
    observed_boundary = snapshot

    def checkpoint(self):
        if self.client.boundary().action == 'stop':
            raise InterruptedError('当前任务已取消或达到显式等待时限')

    @contextmanager
    def admit(self, amount, label='allocation', wait=True, reserve=False, release=None):
        if release is not None and self.client.boundary().action == 'wait':
            release()
        with self.client.reserve(int(amount), purpose=label, wait=wait) as ticket:
            yield None if ticket is None else self.snapshot()

    def __exit__(self, *_):
        # 驻留worker在病例之间仍占内存，交由任务监督器按进程存活情况计入。
        return False
