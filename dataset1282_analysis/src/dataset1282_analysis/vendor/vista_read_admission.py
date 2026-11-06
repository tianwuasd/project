"""Separate payload-read admission using local temporary OS locks, not NAS polling.

The lock covers only read/decompression; resampling, statistics and compression
run after release. OS close/death releases the slot. Shrinking the limit affects
new readers; it never interrupts readers already admitted. Scope is this run.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import time
import threading

_LOCAL=threading.local()
from ..contracts import read_json, atomic_json as write_json


def create_gate(limit):
    root=Path(tempfile.mkdtemp(prefix='vista-reads-'))
    update_gate(root,limit)
    return root


def update_gate(root,limit):
    limit=max(1,int(limit))
    root=Path(root)
    for slot in range(limit):
        file=root/f'{slot}.lock'
        if not file.exists():file.write_bytes(b'0')
    write_json(root/'policy.json',dict(read_concurrency=limit,scope='local temporary disk; actual payload reads'))


def _try_lock(stream):
    try:
        if os.name=='posix':
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        return True
    except (BlockingIOError,PermissionError):return False
    except OSError as error:
        if error.errno in (11,13,36):return False
        raise


@contextmanager
def read_slot():
    location=os.environ.get('VISTA_READ_GATE')
    if not location or getattr(_LOCAL,'depth',0):
        yield;return
    root=Path(location);stream=None
    while stream is None:
        count=read_json(root/'policy.json')['read_concurrency']
        for slot in range(count):
            candidate=(root/f'{slot}.lock').open('r+b')
            if _try_lock(candidate):stream=candidate;break
            candidate.close()
        if stream is None:time.sleep(.02)
    _LOCAL.depth=1
    try:yield
    finally:_LOCAL.depth=0;stream.close()


class ReaderFactory:
    """Picklable callable passed through official fingerprint spawn arguments."""
    def __init__(self,reader):self.reader=reader
    def __call__(self):return AdmittedReader(self.reader())


class AdmittedReader:
    def __init__(self,reader):self.reader=reader
    def read_images(self,*args,**kwargs):
        with read_slot():return self.reader.read_images(*args,**kwargs)
    def read_seg(self,*args,**kwargs):
        with read_slot():return self.reader.read_seg(*args,**kwargs)
    def __getattr__(self,key):return getattr(self.reader,key)


@contextmanager
def reader_scope(reader_class):
    """Only in a dedicated single-case process, never concurrent Python threads."""
    originals={name:getattr(reader_class,name) for name in ('read_images','read_seg')}
    def wrapped(fn):
        def read(*args,**kwargs):
            with read_slot():return fn(*args,**kwargs)
        return read
    for name,fn in originals.items():setattr(reader_class,name,wrapped(fn))
    try:yield
    finally:
        for name,fn in originals.items():setattr(reader_class,name,fn)


@contextmanager
def fingerprint_admission():
    """Keep official sampler/statistics unchanged; only wrap its reader factory."""
    import nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor as module
    original=module.determine_reader_writer_from_dataset_json
    module.determine_reader_writer_from_dataset_json=lambda *a,**k:ReaderFactory(original(*a,**k))
    try:yield
    finally:module.determine_reader_writer_from_dataset_json=original


@contextmanager
def gate_scope(location):
    previous=os.environ.get('VISTA_READ_GATE')
    if location:os.environ['VISTA_READ_GATE']=str(location)
    else:os.environ.pop('VISTA_READ_GATE',None)
    try:yield
    finally:
        if previous is None:os.environ.pop('VISTA_READ_GATE',None)
        else:os.environ['VISTA_READ_GATE']=previous


def admitted_call(function,*args,**kwargs):
    with read_slot():return function(*args,**kwargs)
