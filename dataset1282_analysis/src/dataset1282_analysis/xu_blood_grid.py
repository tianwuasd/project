"""Physical-aspect axial views of the full prepared CT, independent of ROI/labels."""
from functools import lru_cache

import numpy as np
from scipy.ndimage import zoom

from .xu_geometry import _box_in, _triple


class AxialViews:
    def __init__(self, cache, name, targets=None):
        self.cache, self.name, self.targets = cache, name, targets
        geo = cache.geometry(name)
        # Cache axes are a permutation of reader ZYX, not necessarily axial-first.
        self.axes = tuple(geo.transpose_forward.index(i) for i in range(3))
        self.zyx = tuple(geo.cache_shape[i] for i in self.axes)
        self.cache_shape = geo.cache_shape
        self.permutation = geo.transpose_forward
        field_mm = [(geo.bbox_transposed[a][1] - geo.bbox_transposed[a][0]) *
                    geo.spacing_xyz[2-i] for i, a in enumerate(self.axes)]
        self.scaled = tuple(max(1, round(512 * v / max(field_mm[1:]))) for v in field_mm[1:])
        self.offset = tuple((512-n)//2 for n in self.scaled)
        self.image = lru_cache(maxsize=8)(self._image)

    def _plane(self, z, target):
        box = [(0, n) for n in self.cache_shape]
        box[self.axes[0]] = (z, z+1)
        if target:
            data = self.targets.read(self.name, 'blood', box=box)
        else:
            data = self.cache.read_image(self.name, box=box)
        plane = data.transpose(self.axes)[0]
        resized = zoom(plane, np.divide(self.scaled, plane.shape), order=0 if target else 1,
                       mode='nearest', grid_mode=True, prefilter=False)
        canvas = np.full((512,512), -1 if target else 0, dtype=np.int8 if target else np.float32)
        y, x = self.offset
        h, w = self.scaled
        canvas[y:y+h, x:x+w] = resized
        return canvas

    def _image(self, z):
        return self._plane(z, False)

    def label(self, z):
        return self._plane(z, True)

    def window(self, z):
        return [min(max(z+i, 0), self.zyx[0]-1) for i in range(-3,4)]

    def sampling(self, box, shape):
        """Where each requested output centre lands in the padded prediction canvas."""
        bounds = _box_in(box, self.cache_shape)
        shape = _triple(shape, 'blood output shape')
        indices = []
        for i, a in enumerate(self.axes):
            lo, hi = bounds[a]
            coordinates = lo + (np.arange(shape[a])+0.5)*(hi-lo)/shape[a]
            n = self.zyx[i]
            if i == 0:
                ids = np.minimum(coordinates.astype(np.int64), n-1)
            else:
                size = self.scaled[i-1]
                ids = self.offset[i-1] + np.minimum((coordinates*size/n).astype(np.int64), size-1)
            indices.append(ids)
        return indices
