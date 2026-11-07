"""Exclusive operation lock, process birth identity, and atomic status snapshots."""
import contextlib
import ctypes
import json
import os
from pathlib import Path
import socket
import time
import uuid
from ..contracts import atomic_json, read_json


def process_birth(pid):
    if os.name=='nt':
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            error=ctypes.get_last_error()
            if error==87: return None
            raise OSError(error,'Cannot inspect process identity')
        try:
            values=[wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle,*(ctypes.byref(v) for v in values)):
                raise OSError(ctypes.get_last_error(),'Cannot inspect process birth')
            if values[1].dwHighDateTime or values[1].dwLowDateTime:
                return None
            return str((values[0].dwHighDateTime<<32)+values[0].dwLowDateTime)
        finally: kernel.CloseHandle(handle)
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        fields=stat[stat.rfind(')')+2:].split()
        if fields[0]=='Z': return None
        start=fields[19]
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return boot+':'+start
    except FileNotFoundError: return None


def lock_active(path):
    if not Path(path).exists(): return False
    row=read_json(path)
    if row.get('hostname')!=socket.gethostname():
        raise RuntimeError('Cannot verify an operation lock owned by another host')
    if process_birth(row['pid'])==row['birth']: return True
    return any(process_birth(child['pid'])==child['birth'] for child in row.get('workers',[]))


@contextlib.contextmanager
def _exclusive_guard(path):
    """OS locking closes stale-file takeover races; kernel releases it on process death."""
    stream=Path(path).open('a+b')
    try:
        stream.seek(0,2)
        if not stream.tell(): stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e: raise RuntimeError('Another project operation holds the OS lock') from e
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    finally: stream.close()


def register_workers(home,owner,pids):
    workers=[]
    for pid in pids:
        born=process_birth(pid)
        if born is not None: workers.append({'pid':pid,'birth':born})
    owner['workers']=workers
    atomic_json(Path(home)/'.runtime/operation.lock',owner)


def owner_alive(home,owner):
    try:
        if owner.get('hostname')!=socket.gethostname(): return False
        current=read_json(Path(home)/'.runtime/operation.lock')
        return current.get('nonce')==owner.get('nonce') and process_birth(owner['pid'])==owner['birth']
    except (OSError,ValueError,KeyError): return False


@contextlib.contextmanager
def operation_lock(home,kind,run_dir=None):
    path=Path(home)/'.runtime/operation.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    row={'pid':os.getpid(),'birth':process_birth(os.getpid()),'hostname':socket.gethostname(),
         'kind':kind,'run_dir':str(run_dir) if run_dir else None,'started_at':time.time(),
         'nonce':uuid.uuid4().hex,'workers':[],'lock_protocol':2}
    with _exclusive_guard(path.with_name('operation.guard')):
        if path.exists():
            try:
                old=read_json(path)
                # Give orphan self-watchdogs a bounded chance to exit before takeover.
                if old.get('hostname')==socket.gethostname() and old.get('pid') and process_birth(old['pid'])!=old.get('birth'):
                    deadline=time.monotonic()+3
                    while lock_active(path) and time.monotonic()<deadline: time.sleep(.1)
                if lock_active(path): raise RuntimeError('Another project operation or owned worker is active')
            except (ValueError,KeyError):
                # Exclusive kernel lock proves no protocol-v2 writer is still in flight.
                pass
            path.rename(path.with_name('stale-'+str(time.time_ns())+'.json'))
        atomic_json(path,row)
        try: yield row
        finally:
            if path.exists() and read_json(path)==row: path.unlink()


def write_status(directory,**values):
    atomic_json(Path(directory)/'status.json',{'updated_at':time.time(),**values})
