from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class TimingBook:
    def __init__(self):
        self.values, self.lock = {}, Lock()

    @contextmanager
    def measure(self, name):
        began = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - began
            with self.lock:
                count, total, maximum = self.values.get(name, (0, 0., 0.))
                self.values[name] = count + 1, total + elapsed, max(maximum, elapsed)

    def snapshot(self, reset=False):
        with self.lock:
            result = {name: dict(count=count, total_seconds=total,
                      mean_seconds=total/count, max_seconds=maximum)
                      for name, (count, total, maximum) in self.values.items()}
            if reset:
                self.values.clear()
            return result
