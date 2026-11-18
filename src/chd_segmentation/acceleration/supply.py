from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event


class OrderedSupply:
    def __init__(self, client, *, capacity, build, collate):
        if capacity not in (1, 2):
            raise ValueError('供数容量必须为当前批或当前批加下一批')
        self.client, self.capacity, self.build, self.collate = client, capacity, build, collate
        self.queue, self.current, self.last = deque(), None, None

    def __enter__(self):
        self.stopping = Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ordered-supply')
        return self

    def _build(self, requests):
        # 仅取消此供数器的等待，不改变同一会话中其他消费者的取消状态。
        while not self.stopping.is_set():
            decision = self.client.boundary()
            if decision.action == 'stop':
                if decision.reason == 'cancelled':
                    raise InterruptedError('供数任务已取消')
                raise TimeoutError('已达到显式批次资源等待时限')
            with self.client.reserve(0, purpose='batch_ready', wait=False) as allowance:
                if allowance is not None:
                    if self.stopping.is_set():
                        break
                    return self.collate([self.build(request) for request in requests])
            self.stopping.wait(.1)
        raise InterruptedError('供数器已经退出，停止等待新批次额度')

    def submit(self, index, requests):
        if len(self.queue) + int(self.current is not None) >= self.capacity:
            raise BufferError('供数容量已经用完')
        if self.last is not None and index != self.last + 1:
            raise ValueError('仅允许派发下一顺序批次')
        self.queue.append((index, self.pool.submit(self._build, requests)))
        self.last = index

    def take(self, index):
        if self.current is not None or not self.queue or self.queue[0][0] != index:
            raise ValueError('取用批次与派发顺序不一致')
        self.current, future = self.queue.popleft()
        return future.result()

    def consumed(self, index):
        if index != self.current:
            raise ValueError('消费批次与当前批次不一致')
        self.current = None

    def __exit__(self, *_):
        self.stopping.set()
        for _, future in self.queue:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.queue.clear()
