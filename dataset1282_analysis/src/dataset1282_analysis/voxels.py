"""Sequential NIfTI slabs: one gzip pass, all-voxel moments, fixed sample percentiles."""
import gzip
import hashlib
import math
from pathlib import Path
import numpy as np
import nibabel as nib


class HashReader:
    def __init__(self, stream): self.stream, self.hash = stream, hashlib.sha256()
    def read(self, size=-1):
        value = self.stream.read(size); self.hash.update(value); return value
    def tell(self): return self.stream.tell()
    def seekable(self): return False


class NiftiStream:
    def __init__(self, path, chunk_slices=8):
        self.path = Path(path); self.image = nib.load(str(path)); self.proxy = self.image.dataobj
        self.shape = self.image.shape
        if len(self.shape)!=3 or any(d<1 for d in self.shape): raise ValueError('Expected a nonempty 3D image')
        if self.proxy.order != 'F': raise ValueError('Unsupported NIfTI storage order')
        self.chunk = max(1, int(chunk_slices)); self.sha256 = None

    def blocks(self):
        with self.path.open('rb') as raw:
            hashed = HashReader(raw)
            stream = gzip.GzipFile(fileobj=hashed, mode='rb') if self.path.name.endswith('.gz') else hashed
            try:
                offset = int(self.proxy.offset)
                if len(stream.read(offset)) != offset: raise EOFError('Truncated NIfTI header')
                for start in range(0,self.shape[2],self.chunk):
                    depth = min(self.chunk,self.shape[2]-start)
                    count = self.shape[0]*self.shape[1]*depth
                    payload = stream.read(count*self.proxy.dtype.itemsize)
                    if len(payload) != count*self.proxy.dtype.itemsize: raise EOFError('Truncated NIfTI voxel payload')
                    block = np.frombuffer(payload,dtype=self.proxy.dtype).reshape((*self.shape[:2],depth),order='F')
                    if self.proxy.slope != 1 or self.proxy.inter != 0:
                        block = block.astype(np.float64)*self.proxy.slope+self.proxy.inter
                    yield start, block
                # Reach EOF so gzip CRC/trailer checks and the compressed-file digest are complete.
                while stream.read(8*1024*1024): pass
                self.sha256 = hashed.hash.hexdigest()
            finally:
                if hasattr(stream,'close'): stream.close()


def analyze_image(path, policy):
    source = NiftiStream(path, policy.get('chunk_slices',8))
    total = math.prod(source.shape)
    sample_count = min(total,int(policy.get('percentile_samples',100000)))
    positions = np.linspace(0,total-1,sample_count,dtype=np.int64)
    samples, slices = [], []
    n, mean, m2, bad = 0, 0., 0., 0
    minimum, maximum, stored_minimum, stored_maximum = math.inf,-math.inf,math.inf,-math.inf
    for z, block in source.blocks():
        flat = block.ravel(order='F')
        finite = np.isfinite(flat)
        count = int(finite.sum()); bad += len(flat)-count
        values = flat if count==len(flat) else flat[finite]
        if count:
            local_mean=float(np.mean(values,dtype=np.float64)); local_m2=float(np.var(values,dtype=np.float64))*count
            delta=local_mean-mean
            combined=n+count
            m2 += local_m2 + delta*delta*n*count/combined
            mean += delta*count/combined; n=combined
            minimum=min(minimum,float(values.min())); maximum=max(maximum,float(values.max()))
        first=z*source.shape[0]*source.shape[1]
        lo,hi=np.searchsorted(positions,[first,first+len(flat)])
        samples.append(flat[positions[lo:hi]-first].astype(np.float64))
        for local_z in range(block.shape[2]):
            plane=block[:,:,local_z]; valid=plane[np.isfinite(plane)]
            slices.append({'z':z+local_z,'finite_count':int(valid.size),
                           'min':float(valid.min()) if valid.size else None,
                           'max':float(valid.max()) if valid.size else None,
                           'mean':float(valid.mean(dtype=np.float64)) if valid.size else None,
                           'std':float(valid.std(dtype=np.float64)) if valid.size else None})
    sample=np.concatenate(samples); sample=sample[np.isfinite(sample)]
    percentiles = {str(p):float(np.percentile(sample,p)) if len(sample) else None for p in (0.5,1,5,50,95,99,99.5)}
    slope,inter=float(source.proxy.slope),float(source.proxy.inter)
    stored_range=sorted([(minimum-inter)/slope,(maximum-inter)/slope]) if n and slope else [None,None]
    return {'voxel_count':total,'finite_count':n,'nonfinite_count':bad,
            'min':minimum if n else None,'max':maximum if n else None,
            'mean':mean if n else None,'std':math.sqrt(max(0,m2/n)) if n else None,
            'constant':bool(n and minimum==maximum),'stored_min':stored_range[0],'stored_max':stored_range[1],
            'proxy_slope':slope,'proxy_intercept':inter,'intensity_basis':'proxy-scaled; HU unverified',
            'percentiles':percentiles,'percentiles_approximate':sample_count<total,
            'percentile_sample_count':len(sample),'sample_rule':'linspace linear Fortran voxel indices; nonfinite samples omitted',
            'slices':slices,'sha256':source.sha256,'complete_voxel_read':True}
