"""Cooperative cancellation and bounded simultaneous filesystem traffic."""
from contextlib import contextmanager
import threading


class StopRequested(Exception):
    pass


class Control:
    def __init__(self, read_slots=1, stop=None, event=None):
        self.readers = threading.BoundedSemaphore(read_slots)
        self.stop = stop or (lambda: False)
        self.event = event or (lambda *args, **kwargs: None)

    def check(self):
        if self.stop():
            raise StopRequested('cooperative stop requested')

    def stage(self, case_id, name, **fields):
        self.event(case_id, name, **fields)
        self.check()

    @contextmanager
    def io(self):
        while not self.readers.acquire(timeout=.1):
            self.check()
        try:
            self.check()
            yield
        finally:
            self.readers.release()
