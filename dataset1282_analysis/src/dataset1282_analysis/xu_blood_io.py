from contextlib import contextmanager
from threading import RLock

import blosc2
import numpy as np

from .contracts import digest, read_json, sha256_file
from .xu_geometry import _box_in, _resize


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _validate_grid(array, *, label):
    shape = tuple(array.shape)
    if (label and len(shape) != 3) or (not label and (len(shape) != 4 or shape[0] != 1)):
        raise ValueError('Compressed artifact has invalid spatial dimensions')
    if label:
        if array.dtype.kind not in 'iu':
            raise ValueError('Invalid segmentation dtype')
    elif array.dtype != np.float32:
        raise ValueError('Invalid prepared intensity dtype')
    return shape


class _CaseReader:
    def __init__(self, cache, store, name):
        self.cache = cache
        self.store = store
        self.name = name
        self._geometry = cache.geometry(name)
        self.lock = RLock()
        self._arrays = {}
        self._stamps = {}
        self._paths_by_kind = {}
        self.closed = False
        self.stats = {'opens': 0, 'hash_checks': 0}
        self._open_artifact('image')
        if store.has(name, 'blood'):
            self._open_artifact('blood')

    def _paths(self, kind):
        if kind == 'image':
            relative = f'{self.cache._folder}/{self.name}.b2nd'
            path = self.cache._artifact(f'process-{self.name}', relative)
            expected_shape = (1, *self._geometry.cache_shape)
            expected_dtype = np.dtype(np.float32)
        else:
            if not self.store.has(self.name, 'blood'):
                raise ValueError('No eligible blood supervision for this case')
            path = self.store.root/'cases'/self.name/'blood.b2nd'
            record = self.store.manifest['cases'][self.name]['available']['blood']
            stamp = _stamp(path)
            marker = read_json(path.parent/'targets.json')
            self.stats['hash_checks'] += 1
            if (marker['contract'] != digest(self.store.manifest['contract'])
                    or marker['result'] != self.store.manifest['cases'][self.name]
                    or sha256_file(path) != record):
                raise ValueError('Derived label digest mismatch')
            if _stamp(path) != stamp:
                raise ValueError('Derived target changed while opening')
            expected_shape = self._geometry.cache_shape
            expected_dtype = np.dtype(np.int8)
        if path.resolve() != path or not path.is_file():
            raise ValueError('Missing committed blood artifact')
        return path, expected_shape, expected_dtype

    def _open_artifact(self, kind):
        path, expected_shape, expected_dtype = self._paths(kind)
        stamp = _stamp(path)
        array = blosc2.open(str(path), mode='r')
        self.stats['opens'] += 1
        if tuple(array.shape) != expected_shape or array.dtype != expected_dtype:
            raise ValueError('Compressed array disagrees with committed geometry or dtype')
        _validate_grid(array, label=(kind == 'blood'))
        if _stamp(path) != stamp:
            raise ValueError('Committed artifact changed while opening')
        self._arrays[kind] = array
        self._stamps[kind] = stamp
        self._paths_by_kind[kind] = path

    def geometry(self, name):
        if name != self.name:
            raise ValueError('Case reader is scoped to one case')
        return self._geometry

    def _read(self, kind, box, shape):
        with self.lock:
            if self.closed:
                raise RuntimeError('Case reader is closed')
            path = self._paths_by_kind[kind]
            if _stamp(path) != self._stamps[kind]:
                if kind == 'blood':
                    self.stats['hash_checks'] += 1
                    sha256_file(path)
                raise ValueError('Committed artifact changed during case read session')
            array = self._arrays[kind]
            region = _box_in(box, self._geometry.cache_shape)
            if kind == 'image':
                volume = np.asarray(array[(slice(0, 1), *(slice(a, b) for a, b in region))])[0]
                if not np.isfinite(volume).all():
                    raise ValueError('Invalid prepared intensities')
            else:
                volume = np.asarray(array[tuple(slice(a, b) for a, b in region)])
                if volume.size and (volume.min() < -1 or volume.max() > 2):
                    raise ValueError('Invalid blood target classes')
            return _resize(volume, shape, 1 if kind == 'image' else 0)

    def read_image(self, name, shape=None, box=None):
        if name != self.name:
            raise ValueError('Case reader is scoped to one case')
        return self._read('image', box, shape)

    def read(self, name, kind, box=None, shape=None):
        if name != self.name or kind != 'blood':
            raise ValueError('Case target reader is scoped to one blood target')
        return self._read('blood', box, shape)

    def has(self, name, kind):
        return name == self.name and kind == 'blood' and self.store.has(name, kind)

    def close(self):
        with self.lock:
            self._arrays.clear()
            self.closed = True


@contextmanager
def case_readers(cache, store, name):
    reader = _CaseReader(cache, store, name)
    try:
        yield reader, reader
    finally:
        reader.close()
