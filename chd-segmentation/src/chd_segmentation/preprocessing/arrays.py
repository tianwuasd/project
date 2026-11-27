from itertools import product
import math
from pathlib import Path

import blosc2
import numpy as np


NPY_FORMAT = 'chd-offline-npy-v1'
B2ND_FORMAT = 'chd-offline-b2nd-v1'
FORMATS = (NPY_FORMAT, B2ND_FORMAT)


def storage_format(settings):
    return {'npy': NPY_FORMAT, 'b2nd': B2ND_FORMAT}[settings.get('storage_format', 'b2nd')]


class CompressedArray:
    def __init__(self, path):
        self.filename = str(path)
        self.array = blosc2.open(path, mode='r', dparams={'nthreads': 1})
        self.shape, self.dtype = tuple(self.array.shape), np.dtype(self.array.dtype)
        self.ndim, self.size = len(self.shape), math.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize
        self.decode_workspace_bytes = 2 * math.prod(self.array.chunks) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        value = self.array[...]
        return np.array(value, dtype=dtype, copy=True) if copy else np.asarray(value, dtype=dtype)

    def __getitem__(self, selection):
        items = selection if isinstance(selection, tuple) else (selection,)
        if not any(isinstance(item, np.ndarray) for item in items):
            return self.array[selection]
        if len(items) != self.ndim:
            raise IndexError('压缩数组高级索引需要显式指定全部维度')
        region, local = [], []
        for item, size in zip(items, self.shape):
            if isinstance(item, np.ndarray):
                if item.dtype.kind not in 'iu' or not item.size:
                    raise IndexError('压缩数组高级索引需要非空整数数组')
                values = np.where(item < 0, item + size, item)
                if values.min() < 0 or values.max() >= size:
                    raise IndexError('压缩数组索引越界')
                low, high = int(values.min()), int(values.max()) + 1
                region.append(slice(low, high))
                local.append(values-low)
            elif isinstance(item, slice):
                region.append(item)
                local.append(slice(None))
            else:
                index = int(item)
                index = index + size if index < 0 else index
                if not 0 <= index < size:
                    raise IndexError('压缩数组索引越界')
                region.append(slice(index, index+1))
                local.append(0)
        return self.array[tuple(region)][tuple(local)]


def array_name(kind, format):
    if format not in FORMATS:
        raise ValueError('不支持的离线数组格式：' + str(format))
    return kind + ('.b2nd' if format == B2ND_FORMAT else '.npy')


def open_array(directory, kind, format):
    path = Path(directory) / array_name(kind, format)
    return CompressedArray(path) if format == B2ND_FORMAT else np.load(path, mmap_mode='r', allow_pickle=False)


def save_compressed(path, array, settings, boundary=None):
    budget = settings.get('compression_chunk_bytes', 4*1024**2)
    chunks = [min(int(n), 128) for n in array.shape]
    while math.prod(chunks)*array.dtype.itemsize > budget:
        axis = max(range(len(chunks)), key=lambda i: chunks[i])
        chunks[axis] = max(1, (chunks[axis]+1)//2)
    output = blosc2.empty(array.shape, dtype=array.dtype, urlpath=str(path), mode='w', chunks=tuple(chunks),
                         cparams={'codec': blosc2.Codec.ZSTD, 'clevel': settings.get('compression_level', 3), 'nthreads': 1},
                         dparams={'nthreads': 1})
    for start in product(*(range(0, n, c) for n, c in zip(array.shape, chunks))):
        if boundary is not None:
            boundary()
        section = tuple(slice(s, min(n, s+c)) for s, n, c in zip(start, array.shape, chunks))
        values = np.ascontiguousarray(array[section])
        output[section] = values
    del output
    reopened = blosc2.open(str(path), mode='r', dparams={'nthreads': 1})
    if tuple(reopened.shape) != tuple(array.shape) or np.dtype(reopened.dtype) != array.dtype:
        raise ValueError('压缩数组头与待保存数据不同：' + str(path))
    del reopened
    return 2*math.prod(chunks)*array.dtype.itemsize
