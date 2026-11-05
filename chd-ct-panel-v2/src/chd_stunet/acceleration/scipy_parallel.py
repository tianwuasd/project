"""Partition the ORIGINAL SciPy zoom kernel over disjoint shared output views.

SciPy still computes the complete spline prefilter and global zoom factors.
Only its final C kernel is dispatched in output partitions: local integer
indices + an exact integer origin reproduce each original global coordinate.
Input/prefilter/output arrays are shared, with no extra volume per thread.

This owned-process adapter temporarily replaces one private callable. It is
gated by a reviewed SciPy version and exact runtime probes, restored on exit,
and never changes installed package files. Unknown APIs use the serial path.
Source rationale: SciPy v1.17.1 ni_interpolation.c, NI_ZoomShift lines739-760
and NPY_BEGIN_THREADS; https://github.com/scipy/scipy/blob/v1.17.1/scipy/ndimage/src/ni_interpolation.c
"""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from contextlib import contextmanager
import math
import threading
from types import BuiltinFunctionType
import numpy as np

_LOCK=threading.Lock()
_PROVEN=set()
REVIEWED_SCIPY={'1.17.1'}


class _Dispatcher:
    def __init__(self,original,executor,threads,stats,min_voxels,progress=None):
        self.original,self.executor,self.threads=original,executor,threads
        self.stats,self.min_voxels,self.progress=stats,min_voxels,progress
        self.owner=threading.get_ident()

    def kernel(self,input,zoom,shift,output,order,mode,cval,npad,grid_mode):
        if threading.get_ident()!=self.owner:
            raise RuntimeError('Concurrent unowned SciPy call in preprocessing adapter')
        if (zoom is None or shift is not None or output.size<self.min_voxels or output.size==0
                or output.ndim not in (2,3) or order not in (0,1,3)
                or np.shares_memory(input,output)):
            self.stats['serial_calls']+=1
            return self.original(input,zoom,shift,output,order,mode,cval,npad,grid_mode)
        self.stats['parallel_calls']+=1
        axis=0 if output.shape[0]>=self.threads else int(np.argmax(output.shape))
        block=max(1,math.ceil(output.shape[axis]/(self.threads*8)))
        starts=iter(range(0,output.shape[axis],block));pending={};completed=0

        def submit():
            start=next(starts,None)
            if start is None:return False
            slices=[slice(None)]*output.ndim
            slices[axis]=slice(start,min(start+block,output.shape[axis]))
            view=output[tuple(slices)]
            offset=np.zeros(output.ndim,dtype=np.float64);offset[axis]=start
            future=self.executor.submit(self.original,input,zoom,offset,view,order,mode,cval,npad,grid_mode)
            pending[future]=int(view.size)
            self.stats['peak_pending']=max(self.stats['peak_pending'],len(pending))
            return True

        for _ in range(self.threads):submit()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                future.result();completed+=pending.pop(future)
                self.stats['completed_tiles']+=1
                if self.progress:self.progress(completed,int(output.size))
            # Fill after draining completed results; pending never exceeds the
            # admitted CPU threads, and no work waits behind a slow first tile.
            while len(pending)<self.threads and submit():pass
        if completed!=output.size:raise RuntimeError('Interpolation partition coverage differs')


def _stats(threads):
    return dict(requested_threads=threads,applied_threads=1,parallel_calls=0,serial_calls=0,
        completed_tiles=0,peak_pending=0,additional_full_volume_buffers=0)


def _probe(original,ndimage,module):
    """Validate this installed binary/ABI, not merely a version label."""
    x=np.random.default_rng(934).normal(size=(7,9,5))
    with ThreadPoolExecutor(max_workers=2) as executor:
        dispatcher=_Dispatcher(original,executor,2,_stats(2),0)
        for order in (0,1,3):
            for grid in (False,True):
                for mode in ('nearest','reflect'):
                    expected=ndimage.zoom(x,[1.4,.7,1.3],order=order,mode=mode,grid_mode=grid)
                    module.zoom_shift=dispatcher.kernel
                    try:actual=ndimage.zoom(x,[1.4,.7,1.3],order=order,mode=mode,grid_mode=grid)
                    finally:module.zoom_shift=original
                    if not np.array_equal(actual,expected):raise ValueError('Native zoom partition probe differs')


@contextmanager
def parallel_zooms(threads=1,*,min_voxels=500_000,progress=None):
    """One serial preprocessing caller, in its own process, per context.

    Native BLAS/OpenMP remain one thread; this pool's exact admitted thread
    count is separately charged by the V2 supervisor. Do not call this from a
    thread or run overlapping contexts in a shared application process.
    """
    if type(threads) is not int or threads<1:raise ValueError('Positive integer CPU threads required')
    stats=_stats(threads)
    if threads==1:
        stats['reason']='one admitted thread';yield stats;return
    import scipy
    from scipy import ndimage
    from scipy.ndimage import _nd_image
    stats['scipy_version']=scipy.__version__
    if scipy.__version__ not in REVIEWED_SCIPY:
        stats['reason']='unreviewed SciPy version; original serial kernel retained';yield stats;return
    if threading.current_thread() is not threading.main_thread():raise RuntimeError('Owned main process caller required')
    if not _LOCK.acquire(blocking=False):raise RuntimeError('Overlapping SciPy execution contexts refused')
    original=_nd_image.zoom_shift;executor=None
    try:
        if not isinstance(original,BuiltinFunctionType):
            stats['reason']='unrecognized SciPy kernel callable; original retained';yield stats;return
        key=(scipy.__version__,id(original),np.__version__)
        try:
            if key not in _PROVEN:_probe(original,ndimage,_nd_image);_PROVEN.add(key)
        except Exception as error:
            stats['reason']='native probe failed; serial fallback: '+str(error);yield stats;return
        executor=ThreadPoolExecutor(max_workers=threads,thread_name_prefix='native-zoom')
        dispatcher=_Dispatcher(original,executor,threads,stats,min_voxels,progress)
        _nd_image.zoom_shift=dispatcher.kernel
        stats.update(applied_threads=threads,reason='reviewed API and exact installed-kernel probe passed')
        yield stats
    finally:
        # Join only this pool. Its C calls use the captured original directly,
        # and the shared output remains alive until all writes finish.
        if executor:executor.shutdown(wait=True,cancel_futures=True)
        _nd_image.zoom_shift=original
        _LOCK.release()
